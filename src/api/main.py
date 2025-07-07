import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import mlflow.sklearn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Load the model from MLflow Model Registry
try:
    model = mlflow.sklearn.load_model("models:/FraudDetectionModel/Production")
    logger.info("Model loaded successfully from MLflow Model Registry")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

# Define expected feature columns (27 numeric columns from data processing)
FEATURE_COLUMNS = [
    'Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth',
    'TransactionYear', 'Amount_TotalAmount', 'Amount_AvgAmount',
    'Amount_TransactionCount', 'Amount_StdAmount', 'ProductCategory_data_bundles',
    'ProductCategory_financial_services', 'ProductCategory_movies',
    'ProductCategory_other', 'ProductCategory_ticket', 'ProductCategory_transport',
    'ProductCategory_tv', 'ProductCategory_utility_bill', 'ChannelId_ChannelId_2',
    'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 'ProviderId_ProviderId_2',
    'ProviderId_ProviderId_3', 'ProviderId_ProviderId_4', 'ProviderId_ProviderId_5',
    'ProviderId_ProviderId_6'
]


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict fraud risk probability for a single customer data point.

    Parameters:
    - request: PredictionRequest object containing feature values

    Returns:
    - PredictionResponse with risk probability
    """
    try:
        # Convert request to DataFrame
        data_dict = request.dict()
        input_data = pd.DataFrame([data_dict], columns=FEATURE_COLUMNS)

        # Verify all required columns are present
        if set(input_data.columns) != set(FEATURE_COLUMNS):
            logger.error(
                f"Input data columns do not match expected: {list(input_data.columns)}")
            raise HTTPException(
                status_code=400, detail="Input data must match expected feature columns")

        # Predict probability
        risk_proba = model.predict_proba(input_data)[:, 1][0]
        logger.info(f"Prediction made: risk_proba={risk_proba}")

        return PredictionResponse(risk_probability=risk_proba)

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
