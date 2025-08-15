# core/predictor.py
import json
import pickle
import boto3
import mlflow
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import Annotated


# Carga de modelo y utilidades
def load_model(model_name: str, alias: str):
    try:
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(
            model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        with open('/app/files/model.pkl', 'rb') as file_ml:
            model_ml = pickle.load(file_ml)
        version_model_ml = 0

    try:
        s3 = boto3.client('s3')
        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["mean"] = np.array(data_dictionary["mean"])
        data_dictionary["std"] = np.array(data_dictionary["std"])
    except:
        with open('/app/files/data.json', 'r') as file_s3:
            data_dictionary = json.load(file_s3)

    return model_ml, version_model_ml, data_dictionary


# Variables globales
model, version_model, data_dict = load_model("spotify_model_prod", "champion")


def check_model():
    global model, data_dict, version_model
    try:
        model_name = "spotify_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        if new_version_model != version_model:
            model, version_model, data_dict = load_model(model_name, alias)
    except:
        pass


# Modelos Pydantic
class ModelInput(BaseModel):
    speechiness: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    danceability: float = Field(ge=0.0, le=1.0)
    acousticness: float = Field(ge=0.0, le=1.0)

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
    int_output: bool
    str_output: Literal["Not Successful", "Successful"]

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


# Lógica de predicción

def predict(features: ModelInput):
    features_list = list(features.dict().values())
    features_key = list(features.dict().keys())

    features_df = pd.DataFrame(
        np.array(features_list).reshape([1, -1]),
        columns=features_key
    )

    prediction = model.predict(features_df)

    str_pred = "Not Successful"
    if prediction[0] > 0:
        str_pred = "Successful"

    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
