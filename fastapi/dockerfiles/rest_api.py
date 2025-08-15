# fastapi/rest_api.py
from fastapi import APIRouter, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing_extensions import Annotated

from predictor import model, data_dict, version_model, ModelInput, ModelOutput, check_model

router = APIRouter()


@router.get("/")
async def read_root():
    """
    Root endpoint of the Spotify Success Prediction API.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Spotify Success Prediction API"}))


@router.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting music success.
    """
    import pandas as pd
    import numpy as np

    # Convert features into a pandas DataFrame
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]
    features_df = pd.DataFrame(
        np.array(features_list).reshape([1, -1]), columns=features_key)

    # Make prediction using the trained model
    prediction = model.predict(features_df)

    str_pred = "Not Successful"
    if prediction[0] > 0:
        str_pred = "Successful"

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
