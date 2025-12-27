"""
Evaluation utilities (metrics) used across notebooks.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))
