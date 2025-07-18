import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model performance using multiple metrics.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_pred_proba: Predicted probabilities (optional, for ROC-AUC)

    Returns:
    - Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    return metrics


def train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, model_name, experiment_name):
    """
    Train a model with GridSearchCV, evaluate it, and log to MLflow.

    Parameters:
    - model: Scikit-learn model instance
    - param_grid: Hyperparameter grid for GridSearchCV
    - X_train, X_test: Training and testing features
    - y_train, y_test: Training and testing labels
    - model_name: Name of the model for logging
    - experiment_name: MLflow experiment name

    Returns:
    - Best model and its metrics
    """
    logger.info(f"Training {model_name}")

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_run"):
        # Log parameters
        mlflow.log_param("model_name", model_name)

        # Perform Grid Search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        logger.info(
            f"Best parameters for {model_name}: {grid_search.best_params_}")

        # Log hyperparameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(
            best_model, 'predict_proba') else None
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        logger.info(f"Metrics for {model_name}: {metrics}")

        # Log model
        mlflow.sklearn.log_model(best_model, f"{model_name}_model")

        return best_model, metrics


def main():
    """
    Main function to load data, train models, and register the best model.
    """
    try:
        # Load processed data
        data = pd.read_csv('data/processed/processed_data_with_proxy.csv')
        y = pd.read_csv('data/processed/target_proxy.csv')
        logger.info("Data loaded successfully")
        logger.info(f"Data shape: {data.shape}, Target shape: {y.shape}")

        # Identify numeric columns
        numeric_columns = data.select_dtypes(
            include=[np.number]).columns.tolist()
        if 'is_high_risk' in numeric_columns:
            numeric_columns.remove('is_high_risk')  # Exclude target
        logger.info(f"Numeric columns for training: {numeric_columns}")

        # Prepare features and target
        X = data[numeric_columns]  # Use only numeric columns for training
        y = y['is_high_risk']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(
            f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Define models and hyperparameter grids
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'param_grid': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }

        # Train and evaluate models
        best_model = None
        best_metrics = None
        best_model_name = None
        best_f1 = -1

        experiment_name = "Fraud_Detection_Experiment"

        for model_name, config in models.items():
            model, metrics = train_and_evaluate_model(
                config['model'],
                config['param_grid'],
                X_train,
                X_test,
                y_train,
                y_test,
                model_name,
                experiment_name
            )
            # Track best model based on F1 score
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_model = model
                best_metrics = metrics
                best_model_name = model_name

        # Register best model in MLflow Model Registry
        with mlflow.start_run(run_name=f"Best_Model_{best_model_name}"):
            mlflow.log_param("model_name", best_model_name)
            for metric_name, value in best_metrics.items():
                mlflow.log_metric(metric_name, value)
            model_info = mlflow.sklearn.log_model(
                best_model, f"{best_model_name}_best_model")
            mlflow.register_model(model_info.model_uri, "FraudDetectionModel")
            logger.info(
                f"Registered best model: {best_model_name} with F1 score: {best_f1}")

        # Save processed data with CustomerId for downstream use
        data.to_csv(
            'data/processed/processed_data_with_proxy_and_customerid.csv', index=False)

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
