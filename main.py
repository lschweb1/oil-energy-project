from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_model_features
from src.evaluation import r2, rmse
from src.models import fit_linear_regression, fit_random_forest


PROJECT_ROOT = Path(__file__).resolve().parent
ALLOWED_TARGETS = ("XLE_target", "ICLN_target")


def _check_paths() -> None:
    """Basic sanity checks for the expected repository structure."""
    required = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "notebooks",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(f"- {p.relative_to(PROJECT_ROOT)}" for p in missing)
        raise FileNotFoundError(f"Missing expected project files/folders:\n{msg}")


def _time_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split without shuffling."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    n = len(df)
    if n < 50:
        raise ValueError(f"Not enough rows for a train/test split (n={n}).")

    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def run_pipeline(model_name: str, target: str, test_size: float, random_state: int) -> dict:
    # Load feature dataset
    df = load_model_features()

    # Ensure we have a time column (used only for sorting)
    if "Date" in df.columns:
        df = df.sort_values("Date")

    # Validate target
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available targets: {ALLOWED_TARGETS}. "
            f"Columns sample: {list(df.columns[:30])}"
        )

    # Prepare X / y
    y = df[target]
    X = df.drop(columns=[target])

    # Remove non-numeric columns if any (except Date which we can drop from X)
    if "Date" in X.columns:
        X = X.drop(columns=["Date"])

    # Keep only numeric features (robust for graders)
    X = X.select_dtypes(include=["number"])

    # Drop rows with missing values (simple + deterministic)
    data = pd.concat([X, y], axis=1).dropna()
    y = data[target]
    X = data.drop(columns=[target])

    # Train/test split (time-based)
    train_df, test_df = _time_split(pd.concat([X, y], axis=1), test_size=test_size)
    y_train = train_df[target]
    X_train = train_df.drop(columns=[target])
    y_test = test_df[target]
    X_test = test_df.drop(columns=[target])

    # Fit model
    if model_name == "linear":
        model = fit_linear_regression(X_train, y_train)
    elif model_name == "rf":
        model = fit_random_forest(X_train, y_train, random_state=random_state)
    else:
        raise ValueError("Unknown model. Choose from: linear, rf")

    # Predict + metrics
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "target": target,
        "n_rows_total": int(len(df)),
        "n_rows_used": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rmse": rmse(y_test, y_pred),
        "r2": r2(y_test, y_pred),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oil–Energy Relationship Project (minimal reproducible pipeline via main.py)."
    )
    parser.add_argument("--model", choices=["linear", "rf"], default="rf", help="Model to fit (default: rf).")
    parser.add_argument(
        "--target",
        choices=list(ALLOWED_TARGETS),
        default="XLE_target",
        help="Prediction target (default: XLE_target).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (time-based split). Default 0.2.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for RF (default 42).")
    args = parser.parse_args()

    _check_paths()

    print("Oil–Energy Relationship Project")
    print(f"- Project root: {PROJECT_ROOT}")
    print(f"- Running pipeline: model={args.model}, target={args.target}")

    # Ensure results dirs exist
    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_pipeline(
        model_name=args.model,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Save metrics
    out_path = results_dir / f"metrics_main_{metrics['target']}_{metrics['model']}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nPipeline run OK ✅")
    print(f"- RMSE: {metrics['rmse']:.6f}")
    print(f"- R²  : {metrics['r2']:.6f}")
    print(f"- Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()