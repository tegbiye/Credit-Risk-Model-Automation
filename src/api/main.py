import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import mlflow.sklearn
import logging
import os

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create console handler and set level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Create a logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Create file handler and set level
file_handler = logging.FileHandler('logs/app.log')  # Log file name
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add handlers to the logger (avoid duplicate handlers)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Load the model from MLflow Model Registry
try:
    model = mlflow.sklearn.load_model("models:/FraudDetectionModel/Production")
    logger.info("Model loaded successfully from MLflow Model Registry")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

# Define expected feature columns (updated to include CountryCode and PricingStrategy)
FEATURE_COLUMNS = [
    'Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth',
    'TransactionYear', 'Amount_TotalAmount', 'Amount_AvgAmount',
    'Amount_TransactionCount', 'Amount_StdAmount',
    'ProductCategory_data_bundles', 'ProductCategory_financial_services',
    'ProductCategory_movies', 'ProductCategory_other', 'ProductCategory_ticket',
    'ProductCategory_transport', 'ProductCategory_tv', 'ProductCategory_utility_bill',
    'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5',
    'ProviderId_ProviderId_2', 'ProviderId_ProviderId_3', 'ProviderId_ProviderId_4',
    'ProviderId_ProviderId_5', 'ProviderId_ProviderId_6', 'CountryCode', 'PricingStrategy'
]


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        data_dict = request.dict()
        input_data = pd.DataFrame([data_dict], columns=FEATURE_COLUMNS)
        if set(input_data.columns) != set(FEATURE_COLUMNS):
            logger.error(
                f"Input data columns do not match expected: {list(input_data.columns)}")
            raise HTTPException(
                status_code=400, detail="Input data must match expected feature columns")
        risk_proba = model.predict_proba(input_data)[:, 1][0]
        threshold = 0.5
        is_high_risk = risk_proba > threshold
        logger.info(
            f"Prediction made: risk_proba={risk_proba}, is_high_risk={is_high_risk}")
        return PredictionResponse(risk_probability=risk_proba, is_high_risk=is_high_risk)
    except ValidationError as e:
        logger.error(f"Input validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
