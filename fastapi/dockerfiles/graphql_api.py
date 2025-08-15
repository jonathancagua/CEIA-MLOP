import strawberry
from fastapi import APIRouter
from strawberry.fastapi import GraphQLRouter
from predictor import model, check_model
import pandas as pd
import numpy as np

# Definimos el output de GraphQL


@strawberry.type
class ModelOutputGQL:
    int_output: bool
    str_output: str

# Query


@strawberry.type
class Query:
    @strawberry.field
    def predict(self, speechiness: float, energy: float, danceability: float, acousticness: float) -> ModelOutputGQL:
        # Convertimos input a DataFrame
        features_df = pd.DataFrame(
            np.array([speechiness, energy, danceability,
                     acousticness]).reshape(1, -1),
            columns=['speechiness', 'energy', 'danceability', 'acousticness']
        )

        prediction = model.predict(features_df)
        str_pred = "Successful" if prediction[0] > 0 else "Not Successful"
        check_model()
        return ModelOutputGQL(int_output=bool(prediction[0].item()), str_output=str_pred)


# Creamos schema y router
graphql_schema = strawberry.Schema(query=Query)
graphql_router = GraphQLRouter(graphql_schema)

router = APIRouter()
router.include_router(graphql_router, prefix="/graphql")
