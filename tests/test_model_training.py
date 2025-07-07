import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.model_training import evaluate_model

def test_evaluate_model_balanced():
    """Test evaluate_model with a balanced dataset."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_pred_proba = np.array([0.2, 0.6, 0.8, 0.4])
    
    metrics = evaluate_model(y_true, y_pred, y_pred_proba)
    
    assert metrics['accuracy'] == accuracy_score(y_true, y_pred)
    assert metrics['precision'] == precision_score(y_true, y_pred, zero_division=0)
    assert metrics['recall'] == recall_score(y_true, y_pred, zero_division=0)
    assert metrics['f1'] == f1_score(y_true, y_pred, zero_division=0)
    assert metrics['roc_auc'] == roc_auc_score(y_true, y_pred_proba)
    assert len(metrics) == 5
    assert all(isinstance(v, float) for v in metrics.values())

def test_evaluate_model_zero_division():
    """Test evaluate_model with no positive predictions."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.1, 0.3])
    
    metrics = evaluate_model(y_true, y_pred, y_pred_proba)
    
    assert metrics['accuracy'] == 1.0
    assert metrics['precision'] == 0.0  # No positive predictions
    assert metrics['recall'] == 0.0     # No true positives
    assert metrics['f1'] == 0.0         # No positive predictions
    assert metrics['roc_auc'] == pytest.approx(0.0, abs=1e-5)  # No positive class
    assert len(metrics) == 5