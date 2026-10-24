"""Train and compare churn models, then save the best one and the evaluation report.

Steps:
1. stratified 80/20 split, the test set is only used at the end
2. tune XGBoost with Optuna (5-fold CV)
3. compare logistic regression, random forest and XGBoost on the same folds
4. keep the model with the best CV PR-AUC
5. pick the threshold that maximises campaign profit on out-of-fold predictions
6. refit on the whole training set and evaluate once on the test set

Usage: python -m churn.train --trials 60
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import evaluate
from churn.business import best_threshold, campaign_profit, expected_value_targets
from churn.config import (
    CV_FOLDS,
    FIGURES_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from churn.data import load_dataset
from churn.explain import ChurnExplainer
from churn.features import INPUT_FEATURES, build_preprocessor

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost (tuned)",
}


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict:
    """Per-fold metrics and out-of-fold probabilities."""
    oof = np.zeros(len(y))
    fold_metrics = []
    for train_idx, val_idx in cv.split(X, y):
        fold_model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        fold_metrics.append(evaluate.ranking_metrics(y.iloc[val_idx], proba))
    folds = pd.DataFrame(fold_metrics)
    return {
        "mean": folds.mean().to_dict(),
        "std": folds.std().to_dict(),
        "oof": oof,
    }


def tune_xgboost(X: pd.DataFrame, y: pd.Series, cv, n_trials: int) -> dict:
    """Optuna (TPE) search that maximises the mean CV PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
        pipeline = build_pipeline(make_xgboost(params))
        return cross_validate(pipeline, X, y, cv)["mean"]["pr_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Best XGBoost CV PR-AUC %.4f with %s", study.best_value, study.best_params)
    return study.best_params


def make_xgboost(params: dict) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )


def candidate_models(xgb_params: dict) -> dict:
    return {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": make_xgboost(xgb_params),
    }


def compare_policies(y_true, proba, monthly_charges, threshold: float) -> pd.DataFrame:
    """Profit of each targeting strategy on the same customers."""
    n = len(y_true)
    policies = {
        "No campaign": np.zeros(n, dtype=bool),
        "Contact everyone": np.ones(n, dtype=bool),
        "Model, default 0.5 threshold": proba >= 0.5,
        f"Model, profit-optimal threshold ({threshold:.2f})": proba >= threshold,
        "Model, expected-value targeting": expected_value_targets(proba, monthly_charges),
    }
    y = np.asarray(y_true, dtype=bool)
    rows = {}
    for name, targeted in policies.items():
        rows[name] = {
            "contacted": int(targeted.sum()),
            "churners_reached": int((targeted & y).sum()),
            "churners_reached_pct": float((targeted & y).sum() / max(y.sum(), 1)),
            "profit": campaign_profit(y_true, targeted, monthly_charges),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def train(
    n_trials: int = 50,
    data_path: Path = RAW_DATA_PATH,
    model_path: Path = MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> dict:
    df = load_dataset(data_path)
    X, y = df[INPUT_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Train %d / test %d customers, churn rate %.1f%%", len(X_train), len(X_test), 100 * y.mean()
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    logger.info("Tuning XGBoost with %d Optuna trials", n_trials)
    xgb_params = tune_xgboost(X_train, y_train, cv, n_trials)
    candidates = candidate_models(xgb_params)

    cv_results = {}
    for name, model in candidates.items():
        cv_results[name] = cross_validate(build_pipeline(model), X_train, y_train, cv)
        logger.info(
            "%-20s CV ROC-AUC %.4f  PR-AUC %.4f",
            name,
            cv_results[name]["mean"]["roc_auc"],
            cv_results[name]["mean"]["pr_auc"],
        )

    best_name = max(cv_results, key=lambda name: cv_results[name]["mean"]["pr_auc"])
    threshold = best_threshold(
        y_train.to_numpy(), cv_results[best_name]["oof"], X_train["MonthlyCharges"].to_numpy()
    )
    logger.info("Selected %s, profit-optimal threshold %.2f", best_name, threshold)

    fitted = {
        name: build_pipeline(model).fit(X_train, y_train) for name, model in candidates.items()
    }
    test_probas = {name: m.predict_proba(X_test)[:, 1] for name, m in fitted.items()}
    best_pipeline, best_proba = fitted[best_name], test_probas[best_name]

    test_metrics = {name: evaluate.ranking_metrics(y_test, p) for name, p in test_probas.items()}
    final_metrics = evaluate.classification_metrics(y_test, best_proba, threshold)
    policies = compare_policies(y_test, best_proba, X_test["MonthlyCharges"].to_numpy(), threshold)

    explainer = ChurnExplainer(best_pipeline, X_train)
    shap_test = explainer.shap_values(X_test)

    figures_dir.mkdir(parents=True, exist_ok=True)
    labelled = {MODEL_LABELS[n]: p for n, p in test_probas.items()}
    evaluate.plot_roc_pr(y_test, labelled, figures_dir / "roc_pr_curves.png")
    evaluate.plot_calibration(y_test, labelled, figures_dir / "calibration.png")
    evaluate.plot_profit_curve(
        y_test,
        best_proba,
        X_test["MonthlyCharges"].to_numpy(),
        threshold,
        figures_dir / "profit_curve.png",
    )
    evaluate.plot_confusion_matrix(
        y_test, best_proba, threshold, figures_dir / "confusion_matrix.png"
    )
    evaluate.plot_shap_importance(shap_test, figures_dir / "shap_importance.png")
    evaluate.plot_policy_comparison(policies, len(y_test), figures_dir / "policy_comparison.png")

    report = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "customers": len(df),
            "churn_rate": float(y.mean()),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "selected_model": best_name,
        "decision_threshold": threshold,
        "xgboost_params": xgb_params,
        "cv_metrics": {n: {"mean": r["mean"], "std": r["std"]} for n, r in cv_results.items()},
        "test_metrics": test_metrics,
        "test_metrics_selected_model": final_metrics,
        "campaign_policies": policies.to_dict(orient="index"),
        "global_feature_importance": shap_test.abs().mean().sort_values(ascending=False).to_dict(),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=float))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "threshold": threshold,
            "background": X_train.sample(200, random_state=RANDOM_STATE),
            "trained_at": report["trained_at"],
            "metrics": final_metrics,
            "versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
        },
        model_path,
    )
    logger.info("Saved model bundle to %s", model_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials for XGBoost")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train(n_trials=args.trials)
    print(json.dumps(report["test_metrics_selected_model"], indent=2))
    print(pd.DataFrame(report["campaign_policies"]).T.to_string())


if __name__ == "__main__":
    main()
