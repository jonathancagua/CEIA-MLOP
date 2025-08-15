import streamlit as st
import requests
import grpc
import predict_pb2
import predict_pb2_grpc
import time

# Título de la app
st.title("Análisis de canciones: Predicción de éxito")

st.markdown("""
Esta aplicación permite analizar las características de una canción y predecir si será exitosa o no. 
Por favor, ingresa los valores de las siguientes características musicales y elige el protocolo de conexión.
""")

# Entrada de datos
speechiness = st.number_input("Speechiness (0.0 - 1.0)", min_value=0.0,
                              max_value=1.0, value=0.5, step=0.0001, format="%.4f")
energy = st.number_input("Energy (0.0 - 1.0)", min_value=0.0,
                         max_value=1.0, value=0.5, step=0.0001, format="%.4f")
danceability = st.number_input("Danceability (0.0 - 1.0)", min_value=0.0,
                               max_value=1.0, value=0.5, step=0.0001, format="%.4f")
acousticness = st.number_input("Acousticness (0.0 - 1.0)", min_value=0.0,
                               max_value=1.0, value=0.5, step=0.0001, format="%.4f")

# Selección de protocolo
protocol = st.selectbox("Selecciona el protocolo de comunicación", [
                        "REST", "GraphQL", "gRPC", "Todos"])

if st.button("Analizar canción"):
    prediction = None
    inference_time = None

    protocol_results = []

    if protocol == "REST" or protocol == "Todos":
        API_URL = "http://fastapi:8800/predict/"
        song_features = {
            "features": {
                "speechiness": speechiness,
                "energy": energy,
                "danceability": danceability,
                "acousticness": acousticness
            }
        }
        try:
            start = time.time()
            response = requests.post(API_URL, json=song_features)
            end = time.time()
            inference_time = end - start
            if response.status_code == 200:
                prediction_data = response.json()
                prediction = prediction_data.get(
                    "str_output", "Sin información")
                protocol_results.append(("REST", prediction, inference_time))
            else:
                st.error(f"Error REST: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión REST: {e}")

    if protocol == "GraphQL" or protocol == "Todos":
        API_URL = "http://fastapi:8800/graphql"

        # Consulta GraphQL como query
        query = """
        query Predict($speechiness: Float!, $energy: Float!, $danceability: Float!, $acousticness: Float!) {
            predict(speechiness: $speechiness, energy: $energy, danceability: $danceability, acousticness: $acousticness) {
                intOutput
                strOutput
            }
        }
        """

        # Variables a enviar
        variables = {
            "speechiness": speechiness,
            "energy": energy,
            "danceability": danceability,
            "acousticness": acousticness
        }

        try:
            start = time.time()
            response = requests.post(
                API_URL,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"}
            )
            end = time.time()
            inference_time = end - start

            # Parseo de la respuesta
            data = response.json()

            if "errors" in data:
                raise Exception(data["errors"])

            prediction = data["data"]["predict"]["strOutput"]
            protocol_results.append(("GRAPHQL", prediction, inference_time))

        except Exception as e:
            st.error(f"Error GraphQL: {e}")

    if protocol == "gRPC" or protocol == "Todos":
        try:
            start = time.time()
            channel = grpc.insecure_channel('fastapi:50051')
            stub = predict_pb2_grpc.PredictorStub(channel)

            # Usar el nombre correcto del request generado desde el proto
            request = predict_pb2.PredictRequest(
                speechiness=speechiness,
                energy=energy,
                danceability=danceability,
                acousticness=acousticness
            )

            response = stub.Predict(request)
            end = time.time()
            inference_time = end - start

            # Acceder a los campos correctos
            prediction = response.str_output
            protocol_results.append(("GRPC", prediction, inference_time))

        except Exception as e:
            st.error(f"Error gRPC: {e}")

    for proto_name, pred, inf_time in protocol_results:
        if pred:
            st.write(f"### Protocolo: {proto_name}")
            st.write(f"Predicción: **{pred}**")
            st.write(f"Tiempo de inferencia: **{inf_time:.4f} segundos**")
            if pred.lower() == "successful":
                st.success(
                    "¡La canción tiene características prometedoras para ser exitosa!")
            else:
                st.warning(
                    "La canción puede no ser exitosa según las características proporcionadas.")
