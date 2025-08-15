import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["mean"] = np.array(data_dictionary["mean"])
        data_dictionary["std"] = np.array(data_dictionary["std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "spotify_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Input schema for the music success prediction model.

    This class defines the input fields required by the music success prediction model along with their descriptions
    and validation constraints.

    :param speechiness: Speechiness of the track (0.0 to 1.0).
    :param energy: Energy level of the track (0.0 to 1.0).
    :param danceability: Danceability of the track (0.0 to 1.0).
    :param acousticness: Acousticness of the track (0.0 to 1.0).
    """

    speechiness: float = Field(
        description="Speechiness of the track",
        ge=0.0,
        le=1.0,
    )
    energy: float = Field(
        description="Energy level of the track",
        ge=0.0,
        le=1.0,
    )
    danceability: float = Field(
        description="Danceability of the track",
        ge=0.0,
        le=1.0,
    )
    acousticness: float = Field(
        description="Acousticness of the track",
        ge=0.0,
        le=1.0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "speechiness": 0.0444,
                    "energy": 0.521,
                    "danceability": 0.514,
                    "acousticness": 0.713,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the music success prediction model.

    This class defines the output fields returned by the music success prediction model along with their descriptions
    and possible values.

    :param int_output: Output of the model. True if the track is predicted to be successful.
    :param str_output: Output of the model in string form. Can be "Not Successful" or "Successful".
    """

    int_output: bool = Field(
        description="Output of the model. True if the track is predicted to be successful",
    )
    str_output: Literal["Not Successful", "Successful"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "Successful",
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model("spotify_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Spotify Success Prediction API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Spotify Success Prediction API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting music success.

    This endpoint receives features related to a track's characteristics and predicts whether the track will be
    successful or not using a trained model. It returns the prediction result in both integer and string formats.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # Scale the data using standard scaler
    # features_df = (features_df - data_dict["mean"]) / data_dict["std"]

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "Not Successful"
    if prediction[0] > 0:
        str_pred = "Successful"

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
