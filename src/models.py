"""
Model training helpers.

Note: Core experimentation and validation are performed in the notebooks.
These functions provide simple constructors used for reproducibility.
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def fit_linear_regression(X, y) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def fit_random_forest(X, y, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model
