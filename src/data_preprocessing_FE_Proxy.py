import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from data_preprocessing_FE import process_data

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom transformer for calculating RFM metrics


class RFMCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', time_col='TransactionStartTime', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.time_col = time_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        if X.empty:
            logger.error("Input DataFrame is empty")
            raise ValueError("Input DataFrame is empty")

        for col in [self.customer_id_col, self.time_col, self.amount_col]:
            if col not in X.columns:
                logger.error(
                    f"Column '{col}' not found in DataFrame. Available columns: {list(X.columns)}")
                raise KeyError(f"Column '{col}' not found in DataFrame")

        X[self.time_col] = pd.to_datetime(X[self.time_col], errors='coerce')
        if X[self.time_col].isna().all():
            logger.error(
                f"All values in '{self.time_col}' are invalid or missing")
            raise ValueError(
                f"All values in '{self.time_col}' are invalid or missing")
        self.snapshot_date_ = X[self.time_col].max() + pd.Timedelta(days=1)
        logger.info(f"Snapshot date set to {self.snapshot_date_}")
        return self

    def transform(self, X):
        logger.info("Calculating RFM metrics")
        X_copy = X.copy()
        X_copy[self.time_col] = pd.to_datetime(
            X_copy[self.time_col], errors='coerce')

        rfm = X_copy.groupby(self.customer_id_col).agg({
            self.time_col: lambda x: (self.snapshot_date_ - x.max()).days,
            self.customer_id_col: 'count',
            self.amount_col: 'sum'
        }).rename(columns={
            self.time_col: 'Recency',
            self.customer_id_col: 'Frequency',
            self.amount_col: 'Monetary'
        }).reset_index()

        if rfm.empty:
            logger.error("RFM calculation resulted in an empty DataFrame")
            raise ValueError("RFM calculation resulted in an empty DataFrame")
        logger.info(f"RFM DataFrame shape: {rfm.shape}")
        logger.info(f"RFM DataFrame preview:\n{rfm.head().to_string()}")
        return rfm

# Custom transformer for K-Means clustering


class RFMClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        logger.info("Fitting K-Means clustering")
        if X.shape[0] < self.n_clusters:
            logger.error(
                f"Number of samples ({X.shape[0]}) is less than number of clusters ({self.n_clusters})")
            raise ValueError(
                f"Number of samples is less than number of clusters")

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(
            X[['Recency', 'Frequency', 'Monetary']])

        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state)
        self.kmeans_.fit(X_scaled)
        return self

    def transform(self, X):
        logger.info("Assigning cluster labels")
        X_copy = X.copy()
        X_scaled = self.scaler_.transform(
            X_copy[['Recency', 'Frequency', 'Monetary']])
        X_copy['Cluster'] = self.kmeans_.predict(X_scaled)
        return X_copy

# Custom transformer for assigning high-risk labels


class HighRiskLabeler(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId'):
        self.customer_id_col = customer_id_col

    def fit(self, X, y=None):
        logger.info("Identifying high-risk cluster")
        cluster_stats = X.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        })
        logger.info(f"Cluster statistics:\n{cluster_stats.to_string()}")

        if cluster_stats.empty:
            logger.error("Cluster statistics DataFrame is empty")
            raise ValueError("Cluster statistics DataFrame is empty")

        high_risk_candidates = cluster_stats[
            (cluster_stats['Recency'] == cluster_stats['Recency'].max()) &
            (cluster_stats['Frequency'] == cluster_stats['Frequency'].min()) &
            (cluster_stats['Monetary'] == cluster_stats['Monetary'].min())
        ]

        if not high_risk_candidates.empty:
            self.high_risk_cluster_ = high_risk_candidates.index[0]
            logger.info(
                f"High-risk cluster identified: Cluster {self.high_risk_cluster_}")
        else:
            cluster_stats['Combined_Score'] = cluster_stats['Frequency'] + \
                cluster_stats['Monetary']
            self.high_risk_cluster_ = cluster_stats['Combined_Score'].idxmin()
            logger.warning(
                f"No cluster meets all criteria. Selected Cluster {self.high_risk_cluster_} with lowest Frequency + Monetary")

        return self

    def transform(self, X):
        logger.info("Assigning high-risk labels")
        X_copy = X.copy()
        X_copy['is_high_risk'] = (
            X_copy['Cluster'] == self.high_risk_cluster_).astype(int)
        return X_copy[[self.customer_id_col, 'is_high_risk']]

# Main pipeline for proxy target variable engineering


def create_rfm_pipeline(customer_id_col='CustomerId'):
    """
    Creates a sklearn Pipeline for RFM-based target variable engineering.

    Parameters:
    - customer_id_col: Name of the customer ID column

    Returns:
    - Pipeline object
    """
    logger.info("Creating RFM pipeline")
    pipeline = Pipeline([
        ('rfm_calculator', RFMCalculator(customer_id_col=customer_id_col)),
        ('rfm_clusterer', RFMClusterer(n_clusters=3, random_state=42)),
        ('high_risk_labeler', HighRiskLabeler(customer_id_col=customer_id_col))
    ])
    return pipeline

# Function to process data and add proxy target variable


def process_data_with_proxy_target(input_data, numerical_columns=None, categorical_columns=None, customer_id_col='CustomerId'):
    """
    Processes raw data and adds proxy target variable (is_high_risk).

    Parameters:
    - input_data: Input DataFrame
    - numerical_columns: List of numerical column names
    - categorical_columns: List of categorical column names
    - customer_id_col: Name of the customer ID column

    Returns:
    - X_processed: Processed feature matrix
    - y: Original target (FraudResult)
    - y_proxy: Proxy target (is_high_risk)
    - feature_names: Names of the processed features
    """
    logger.info("Starting data processing with proxy target engineering")

    if customer_id_col not in input_data.columns:
        logger.error(
            f"Column '{customer_id_col}' not found in input data. Available columns: {list(input_data.columns)}")
        raise KeyError(f"Column '{customer_id_col}' not found in input data")

    X_processed, y, feature_names = process_data(
        input_data,
        target_column='FraudResult',
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        customer_id_col=customer_id_col
    )

    if customer_id_col not in X_processed.columns:
        logger.error(
            f"Column '{customer_id_col}' missing in processed data. Available columns: {list(X_processed.columns)}")
        raise KeyError(f"Column '{customer_id_col}' missing in processed data")

    rfm_pipeline = create_rfm_pipeline(customer_id_col=customer_id_col)
    high_risk_df = rfm_pipeline.fit_transform(input_data)

    X_processed = X_processed.merge(
        high_risk_df, on=customer_id_col, how='left')
    X_processed['is_high_risk'] = X_processed['is_high_risk'].fillna(
        0).astype(int)
    feature_names.append('is_high_risk')

    logger.info(
        f"Proxy target variable added. Final shape: {X_processed.shape}")
    return X_processed, y, X_processed['is_high_risk'], feature_names
