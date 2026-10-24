"""Metrics and report plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.business import profit_curve  # noqa: E402

PALETTE = ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626"]
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)


def ranking_metrics(y_true, proba) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        **ranking_metrics(y_true, proba),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
    }


def plot_roc_pr(y_true, probas: dict[str, np.ndarray], path: Path) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for color, (name, proba) in zip(PALETTE, probas.items(), strict=False):
        fpr, tpr, _ = roc_curve(y_true, proba)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax_roc.plot(fpr, tpr, color=color, label=f"{name} (AUC {auc:.3f})")
        ax_pr.plot(recall, precision, color=color, label=f"{name} (AP {ap:.3f})")
    ax_roc.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax_pr.axhline(np.mean(y_true), ls="--", color="grey", lw=1, label="Random")
    ax_roc.set(
        title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate"
    )
    ax_pr.set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(y_true, probas: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for color, (name, proba) in zip(PALETTE, probas.items(), strict=False):
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=color, label=name, ms=4)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    ax.set(
        title="Calibration (test set)",
        xlabel="Predicted churn probability",
        ylabel="Observed churn rate",
    )
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_profit_curve(y_true, proba, monthly_charges, chosen_threshold: float, path: Path) -> None:
    thresholds, profits = profit_curve(y_true, proba, monthly_charges)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thresholds, profits / 1000, color=PALETTE[0], lw=2)
    ax.axvline(
        chosen_threshold,
        color=PALETTE[2],
        ls="--",
        label=f"Chosen threshold = {chosen_threshold:.2f}",
    )
    ax.axvline(0.5, color=PALETTE[4], ls=":", label="Default threshold = 0.50")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title="Retention campaign profit vs. decision threshold (test set)",
        xlabel="Contact customers with churn probability ≥ threshold",
        ylabel="Campaign profit ($ thousands)",
    )
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(y_true, proba, threshold: float, path: Path) -> None:
    cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = ["Stays", "Churns"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix @ {threshold:.2f}")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{value}",
            ha="center",
            va="center",
            color="white" if value > cm.max() / 2 else "black",
            fontsize=13,
        )
    fig.savefig(path)
    plt.close(fig)


def plot_shap_importance(shap_values: pd.DataFrame, path: Path, top: int = 15) -> None:
    importance = shap_values.abs().mean().sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(importance.index, importance.values, color=PALETTE[0])
    ax.set(
        title="What drives churn? Mean |SHAP value| per customer field",
        xlabel="Mean absolute impact on churn log-odds",
    )
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_policy_comparison(policies: pd.DataFrame, n_customers: int, path: Path) -> None:
    data = policies.sort_values("profit")
    colors = [PALETTE[2] if p > 0 else PALETTE[4] for p in data["profit"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(data.index, data["profit"] / 1000, color=colors)
    ax.bar_label(bars, labels=[f"${p / 1000:,.1f}k" for p in data["profit"]], padding=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        title=f"Campaign profit by targeting policy (test set, {n_customers:,} customers)",
        xlabel="Profit ($ thousands)",
    )
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.2)
    fig.savefig(path)
    plt.close(fig)
