import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from typing import Optional


def evaluate_by_confidence(
    df: pd.DataFrame,
    levels: list[int] = [1, 2, 3, 4],
    confidence_thresholds: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    average: str = "weighted",
) -> pd.DataFrame:
    """
    Calculate accuracy, F1, precision, recall for each level at each confidence threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: level{i}, predicted_level{i}, confidence_level{i}
    levels : list[int]
        Which levels to evaluate (default [1,2,3,4]).
    confidence_thresholds : list[float]
        Minimum confidence values to filter on.
    average : str
        Averaging method for F1/precision/recall ('weighted', 'macro', 'micro').

    Returns
    -------
    pd.DataFrame with columns:
        level, threshold, accuracy, f1, precision, recall, coverage, n_samples
    """
    results = []

    for level in levels:
        true_col = f"level{level}"
        pred_col = f"predicted_level{level}"
        conf_col = f"confidence_level{level}"

        if not all(c in df.columns for c in [true_col, pred_col, conf_col]):
            print(f"⚠️  Skipping level {level}: missing columns")
            continue

        for thresh in confidence_thresholds:
            mask = df[conf_col] >= thresh
            subset = df[mask].dropna(subset=[true_col, pred_col])

            n = len(subset)
            coverage = n / len(df) if len(df) > 0 else 0

            if n == 0:
                results.append({
                    "level": level,
                    "threshold": thresh,
                    "accuracy": np.nan,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "coverage": 0.0,
                    "n_samples": 0,
                })
                continue

            y_true = subset[true_col].astype(str)
            y_pred = subset[pred_col].astype(str)

            results.append({
                "level": level,
                "threshold": thresh,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
                "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
                "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
                "coverage": coverage,
                "n_samples": n,
            })

    return pd.DataFrame(results)


def classification_report_by_level(
    df: pd.DataFrame,
    level: int,
    min_confidence: float = 0.0,
) -> str:
    """
    Return sklearn classification_report for a given level filtered by confidence.
    """
    true_col = f"level{level}"
    pred_col = f"predicted_level{level}"
    conf_col = f"confidence_level{level}"

    mask = df[conf_col] >= min_confidence
    subset = df[mask].dropna(subset=[true_col, pred_col])

    y_true = subset[true_col].astype(str)
    y_pred = subset[pred_col].astype(str)

    return classification_report(y_true, y_pred, zero_division=0)


def plot_accuracy_vs_coverage(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot accuracy and coverage vs confidence threshold for each level.
    """
    import matplotlib.pyplot as plt

    levels = sorted(results_df["level"].unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 5), sharey=False)

    if len(levels) == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        data = results_df[results_df["level"] == level]

        ax.plot(data["threshold"], data["accuracy"], "o-", label="Accuracy", color="tab:blue")
        ax.plot(data["threshold"], data["f1"], "s--", label="F1", color="tab:orange")

        ax2 = ax.twinx()
        ax2.bar(data["threshold"], data["coverage"], alpha=0.2, width=0.05, label="Coverage", color="tab:green")
        ax2.set_ylabel("Coverage")
        ax2.set_ylim(0, 1.05)

        ax.set_title(f"Level {level}")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        ax2.legend(loc="lower right")

    plt.suptitle("Accuracy / F1 vs Confidence Threshold", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


# ──────────────────────────────────────────────
# Usage example
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # df = pd.read_csv("your_data.csv")

    # 1) Summary table
    # results = evaluate_by_confidence(df, levels=[1, 2, 3, 4])
    # print(results.to_string(index=False))

    # 2) Detailed report for level 3 with confidence >= 0.5
    # print(classification_report_by_level(df, level=3, min_confidence=0.5))

    # 3) Plot
    # plot_accuracy_vs_coverage(results)

    print("Import this module and use the functions with your DataFrame.")