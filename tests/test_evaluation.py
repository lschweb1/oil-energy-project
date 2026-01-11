import numpy as np
from src.evaluation import rmse, r2


def test_rmse_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_r2_one_when_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == 1.0
