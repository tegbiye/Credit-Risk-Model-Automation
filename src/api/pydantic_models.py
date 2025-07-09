from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Pydantic model for input data to the /predict endpoint.
    Matches the 29 numeric feature columns expected by the model.
    """
    Amount: float = Field(..., description="Transaction amount")
    Value: float = Field(..., description="Transaction value")
    TransactionHour: int = Field(..., ge=0, le=23,
                                 description="Hour of transaction")
    TransactionDay: int = Field(..., ge=1, le=31,
                                description="Day of transaction")
    TransactionMonth: int = Field(..., ge=1, le=12,
                                  description="Month of transaction")
    TransactionYear: int = Field(..., ge=2000,
                                 description="Year of transaction")
    Amount_TotalAmount: float = Field(...,
                                      description="Total amount per customer")
    Amount_AvgAmount: float = Field(...,
                                    description="Average amount per customer")
    Amount_TransactionCount: float = Field(
        ..., description="Number of transactions per customer")
    Amount_StdAmount: Optional[float] = Field(
        None, description="Standard deviation of amount per customer")
    ProductCategory_data_bundles: float = Field(
        ..., ge=0, le=1, description="One-hot encoded ProductCategory_data_bundles")
    ProductCategory_financial_services: float = Field(
        ..., ge=0, le=1, description="One-hot encoded ProductCategory_financial_services")
    ProductCategory_movies: float = Field(..., ge=0, le=1,
                                          description="One-hot encoded ProductCategory_movies")
    ProductCategory_other: float = Field(..., ge=0, le=1,
                                         description="One-hot encoded ProductCategory_other")
    ProductCategory_ticket: float = Field(..., ge=0, le=1,
                                          description="One-hot encoded ProductCategory_ticket")
    ProductCategory_transport: float = Field(
        ..., ge=0, le=1, description="One-hot encoded ProductCategory_transport")
    ProductCategory_tv: float = Field(..., ge=0, le=1,
                                      description="One-hot encoded ProductCategory_tv")
    ProductCategory_utility_bill: float = Field(
        ..., ge=0, le=1, description="One-hot encoded ProductCategory_utility_bill")
    ChannelId_ChannelId_2: float = Field(..., ge=0, le=1,
                                         description="One-hot encoded ChannelId_ChannelId_2")
    ChannelId_ChannelId_3: float = Field(..., ge=0, le=1,
                                         description="One-hot encoded ChannelId_ChannelId_3")
    ChannelId_ChannelId_5: float = Field(..., ge=0, le=1,
                                         description="One-hot encoded ChannelId_ChannelId_5")
    ProviderId_ProviderId_2: float = Field(..., ge=0, le=1,
                                           description="One-hot encoded ProviderId_ProviderId_2")
    ProviderId_ProviderId_3: float = Field(..., ge=0, le=1,
                                           description="One-hot encoded ProviderId_ProviderId_3")
    ProviderId_ProviderId_4: float = Field(..., ge=0, le=1,
                                           description="One-hot encoded ProviderId_ProviderId_4")
    ProviderId_ProviderId_5: float = Field(..., ge=0, le=1,
                                           description="One-hot encoded ProviderId_ProviderId_5")
    ProviderId_ProviderId_6: float = Field(..., ge=0, le=1,
                                           description="One-hot encoded ProviderId_ProviderId_6")
    CountryCode: float = Field(...,
                               description="Country code of the transaction")
    PricingStrategy: float = Field(...,
                                   description="Pricing strategy of the transaction")


class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0,
                                    description="Probability of high fraud risk")
    is_high_risk: bool = Field(
        ..., description="Whether the transaction is classified as high risk")
