"""
XGBoost Rating Predictor.

Final stage of the pipeline. Takes the fused feature matrix and
predicts ratings using a gradient-boosted tree model.

Features:
  - Optuna hyperparameter optimization
  - Deterministic predictions (fixed random_state)
  - SHAP feature importance analysis
  - Model persistence
"""

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

logger = logging.getLogger(__name__)

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    save_dir: str = "checkpoints/xgboost",
) -> xgb.XGBRegressor:
    """
    Train XGBoost with Optuna hyperparameter tuning.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (ratings).
        X_val: Validation feature matrix.
        y_val: Validation labels.
        config: Predictor config dict from YAML.
        save_dir: Directory to save model and study results.

    Returns:
        Best XGBRegressor model.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    search = config["search_space"]

    def objective(trial):
        params = {
            "objective": config["objective"],
            "n_estimators": trial.suggest_int("n_estimators", *search["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *search["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *search["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *search["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *search["colsample_bytree"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *search["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *search["reg_lambda"], log=True),
            "random_state": config["random_state"],
            "tree_method": config["tree_method"],
            "verbosity": 0,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        return rmse

    logger.info("Starting Optuna hyperparameter search (%d trials)...", config["optuna_trials"])
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config["optuna_trials"])

    logger.info("Best trial — RMSE: %.4f", study.best_value)
    logger.info("Best params: %s", json.dumps(study.best_params, indent=2))

    # Save study results
    with open(save_path / "optuna_results.json", "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
        }, f, indent=2)

    # Train final model with best params
    best_params = {
        **study.best_params,
        "objective": config["objective"],
        "random_state": config["random_state"],
        "tree_method": config["tree_method"],
        "verbosity": 0,
    }
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Save model
    model_path = save_path / "best_model.json"
    best_model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    return best_model


def evaluate(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate XGBoost model on test set.

    Returns:
        dict with RMSE, MAE, and predictions.
    """
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    logger.info("Test Results — RMSE: %.4f, MAE: %.4f", rmse, mae)

    return {
        "rmse": rmse,
        "mae": mae,
        "predictions": predictions,
        "ground_truth": y_test,
    }


def analyze_features(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    feature_names: list = None,
    save_dir: str = "results",
    n_samples: int = 500,
):
    """
    Run SHAP analysis for feature importance.

    Args:
        model: Trained XGBoost model.
        X: Feature matrix (use test set).
        feature_names: Optional list of feature names.
        save_dir: Directory to save SHAP plots.
        n_samples: Number of samples for SHAP computation.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("SHAP or matplotlib not available. Skipping feature analysis.")
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values for %d samples...", n_samples)
    explainer = shap.TreeExplainer(model)

    # Subsample for speed
    if len(X) > n_samples:
        indices = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(save_path / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved to %s", save_path / "shap_summary.png")

    # Bar plot (mean absolute SHAP)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(save_path / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP importance plot saved to %s", save_path / "shap_importance.png")
