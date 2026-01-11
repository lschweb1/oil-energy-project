import numpy as np
import pandas as pd

from src.models import fit_linear_regression, fit_random_forest


def _toy_data(n=200, random_state=0):
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    y = 0.5 * X["f1"] - 0.2 * X["f2"] + rng.normal(scale=0.1, size=n)
    return X, y


def test_fit_linear_regression_predict_shape():
    X, y = _toy_data()
    model = fit_linear_regression(X, y)
    preds = model.predict(X.head(10))
    assert preds.shape == (10,)


def test_fit_random_forest_reproducible_with_seed():
    X, y = _toy_data()
    m1 = fit_random_forest(X, y, random_state=42)
    m2 = fit_random_forest(X, y, random_state=42)
    p1 = m1.predict(X.head(20))
    p2 = m2.predict(X.head(20))
    assert np.allclose(p1, p2)
