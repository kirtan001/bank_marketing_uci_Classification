from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from predict import predict


# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predict whether a customer will subscribe to a term deposit",
    version="1.0.0"
)


# -----------------------------
# Input data schema
# -----------------------------
class CustomerInput(BaseModel):
    age: int
    marital: Literal["single", "married", "divorced"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["yes", "no"]
    balance: float
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]
    contact: Literal["cellular", "telephone", "not_recorded"]
    day: int
    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]
    campaign: int
    pdays: int
    previous: int
    poutcome: Literal[
        "success", "failure", "other", "no_previous_campaign"
    ]


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def make_prediction(customer: CustomerInput):
    """
    Generate prediction for a single customer.
    """
    input_data = customer.dict()
    result = predict(input_data)
    return result
