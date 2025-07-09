import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom transformer for time-based feature extraction
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_column='TransactionStartTime'):
        self.time_column = time_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Extracting time-based features")
        X_copy = X.copy()
        X_copy[self.time_column] = pd.to_datetime(X_copy[self.time_column], errors='coerce')
        X_copy['TransactionHour'] = X_copy[self.time_column].dt.hour
        X_copy['TransactionDay'] = X_copy[self.time_column].dt.day
        X_copy['TransactionMonth'] = X_copy[self.time_column].dt.month
        X_copy['TransactionYear'] = X_copy[self.time_column].dt.year
        logger.info(f"Columns after time feature extraction: {list(X_copy.columns)}")
        return X_copy.drop(columns=[self.time_column])

# Custom transformer for aggregating features by CustomerId
class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, group_by='CustomerId', value_column='Amount'):
        self.group_by = group_by
        self.value_column = value_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info(f"Aggregating features by {self.group_by}")
        agg_features = X.groupby(self.group_by).agg({
            self.value_column: [
                ('TotalAmount', 'sum'),
                ('AvgAmount', 'mean'),
                ('TransactionCount', 'count'),
                ('StdAmount', 'std')
            ]
        }).reset_index()
        agg_features.columns = [self.group_by] + [f"{self.value_column}_{col}" for col in ['TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount']]
        X_merged = X.merge(agg_features, on=self.group_by, how='left')
        logger.info(f"Columns after aggregation: {list(X_merged.columns)}")
        return X_merged

# Custom transformer for handling outliers using IQR
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, factor=1.5):
        self.columns = columns
        self.factor = factor
    
    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in self.columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.bounds_[col] = (Q1 - self.factor * IQR, Q3 + self.factor * IQR)
        return self
    
    def transform(self, X):
        logger.info("Handling outliers")
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                lower, upper = self.bounds_[col]
                X_copy[col] = X_copy[col].clip(lower=lower, upper=upper)
        logger.info(f"Columns after outlier handling: {list(X_copy.columns)}")
        return X_copy

# Main data processing pipeline
def create_data_pipeline(numerical_columns, categorical_columns, customer_id_col='CustomerId', time_column='TransactionStartTime'):
    """
    Creates a sklearn Pipeline for data processing.
    
    Parameters:
    - numerical_columns: List of numerical column names
    - categorical_columns: List of categorical column names
    - customer_id_col: Name of the customer ID column
    - time_column: Name of the time column
    
    Returns:
    - Pipeline object
    """
    logger.info("Creating data processing pipeline")
    
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ], remainder='passthrough')
    
    pipeline = Pipeline([
        ('time_features', TimeFeatureExtractor(time_column=time_column)),
        ('aggregator', CustomerAggregator(group_by=customer_id_col, value_column='Amount')),
        ('outlier_handler', OutlierHandler(columns=['Amount', 'Amount_TotalAmount', 'Amount_AvgAmount', 'Amount_StdAmount'])),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

# Function to process data and preserve target
def process_data(input_data, target_column='FraudResult', numerical_columns=None, categorical_columns=None, customer_id_col='CustomerId', time_column='TransactionStartTime'):
    """
    Processes raw data into model-ready format.
    
    Parameters:
    - input_data: Input DataFrame
    - target_column: Name of the target column
    - numerical_columns: List of numerical column names
    - categorical_columns: List of categorical column names
    - customer_id_col: Name of the customer ID column
    - time_column: Name of the time column
    
    Returns:
    - X_processed: Processed feature matrix
    - y: Target variable
    - feature_names: Names of the processed features
    """
    logger.info("Starting data processing")
    
    # Check if required columns exist
    required_columns = [customer_id_col, target_column, time_column] + numerical_columns + categorical_columns
    for col in required_columns:
        if col not in input_data.columns:
            logger.error(f"Column '{col}' not found in DataFrame. Available columns: {list(input_data.columns)}")
            raise KeyError(f"Column '{col}' not found in DataFrame")
    
    # Separate features and target
    X = input_data.drop(columns=[target_column])
    y = input_data[target_column].copy()
    
    # Update numerical_columns to include generated features
    generated_columns = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                        'Amount_TotalAmount', 'Amount_AvgAmount', 'Amount_TransactionCount', 'Amount_StdAmount']
    numerical_columns = numerical_columns + [col for col in generated_columns if col not in numerical_columns]
    
    # Create and fit pipeline
    pipeline = create_data_pipeline(numerical_columns, categorical_columns, customer_id_col, time_column)
    X_processed = pipeline.fit_transform(X)
    
    # Get feature names
    preprocessor = pipeline.named_steps['preprocessor']
    num_features = numerical_columns
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_columns)
    
    # Identify passthrough columns, excluding time_column
    input_columns = X.columns.tolist()
    transformed_columns = numerical_columns + categorical_columns + [time_column]
    passthrough_columns = [col for col in input_columns if col not in transformed_columns]
    
    # Combine feature names
    feature_names = list(num_features) + list(cat_features) + passthrough_columns
    
    # Remove duplicates in feature_names
    feature_names = list(dict.fromkeys(feature_names))
    
    # Log shapes and feature names
    logger.info(f"Transformed data shape: {X_processed.shape}")
    logger.info(f"Expected feature names count: {len(feature_names)}")
    logger.info(f"Feature names: {feature_names}")
    
    # Check for shape mismatch
    if X_processed.shape[1] != len(feature_names):
        logger.error(f"Shape mismatch: Transformed data has {X_processed.shape[1]} columns, but feature_names has {len(feature_names)}")
        raise ValueError(f"Shape mismatch: Transformed data has {X_processed.shape[1]} columns, but feature_names has {len(feature_names)}")
    
    # Convert to DataFrame
    X_processed = pd.DataFrame(X_processed, columns=feature_names)
    
    logger.info(f"Data processing complete. Output shape: {X_processed.shape}")
    return X_processed, y, feature_names

# Example usage
if __name__ == "__main__":
    try:
        data = pd.read_csv('data.csv')
        logger.info("Data loaded successfully")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {list(data.columns)}")
        
        # Define columns
        numerical_columns = ['Amount', 'Value']
        categorical_columns = ['ProductCategory', 'ChannelId', 'ProviderId']
        customer_id_col = 'CustomerId'
        time_column = 'TransactionStartTime'
        
        # Process data
        X_processed, y, feature_names = process_data(
            data,
            target_column='FraudResult',
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            customer_id_col=customer_id_col,
            time_column=time_column
        )
        
        # Save processed data
        X_processed.to_csv('processed_data.csv', index=False)
        y.to_csv('target.csv', index=False)
        
        logger.info("Processed data and target saved")
        print("Feature names:", feature_names)
        print("Processed data preview:")
        print(X_processed.head())
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")