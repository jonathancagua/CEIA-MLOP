# fastapi/grpc_server.py
from concurrent import futures
import grpc
import predict_pb2
import predict_pb2_grpc

import pandas as pd
import numpy as np
from predictor import model, check_model


class PredictorServicer(predict_pb2_grpc.PredictorServicer):
    def Predict(self, request, context):
        # Convertimos request a DataFrame
        features_df = pd.DataFrame(
            np.array([request.speechiness, request.energy,
                     request.danceability, request.acousticness]).reshape(1, -1),
            columns=['speechiness', 'energy', 'danceability', 'acousticness']
        )

        # Predicción
        prediction = model.predict(features_df)
        str_pred = "Successful" if prediction[0] > 0 else "Not Successful"

        # Chequear modelo
        check_model()

        return predict_pb2.PredictResponse(
            int_output=bool(prediction[0].item()),
            str_output=str_pred
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    predict_pb2_grpc.add_PredictorServicer_to_server(
        PredictorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server running on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
