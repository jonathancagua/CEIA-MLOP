# fastapi/app.py
from fastapi import FastAPI
from rest_api import router as rest_router
from graphql_api import router as graphql_router

app = FastAPI()
app.include_router(rest_router)
app.include_router(graphql_router)
