from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
)

ORDINAL_LABELS = ["Low", "Medium", "High", "Critical"]


def classification_metrics(
    y_true: list[str], y_pred: list[str], probabilities: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = labels or sorted(set(y_true) | set(y_pred))
    result: dict[str, Any] = {
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "matthews_correlation_coefficient": (
            float(matthews_corrcoef(y_true, y_pred)) if y_true else None
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }
    if y_true and set(y_true + y_pred).issubset(ORDINAL_LABELS):
        ordinal_scores = {label: index for index, label in enumerate(ORDINAL_LABELS)}
        quadratic_weighted_kappa = cohen_kappa_score(
            y_true, y_pred, labels=ORDINAL_LABELS, weights="quadratic"
        )
        result["ordinal"] = {
            "label_order": ORDINAL_LABELS,
            "mean_absolute_error": float(
                np.mean(
                    [
                        abs(ordinal_scores[truth] - ordinal_scores[prediction])
                        for truth, prediction in zip(y_true, y_pred)
                    ]
                )
            ),
            "quadratic_weighted_kappa": (
                float(quadratic_weighted_kappa)
                if np.isfinite(quadratic_weighted_kappa)
                else None
            ),
        }
    if probabilities is not None and len(probabilities):
        result["log_loss"] = float(log_loss(y_true, probabilities, labels=labels))
    return result


def agreement_metrics(english: list[str], translated: list[str]) -> dict[str, float | int]:
    if len(english) != len(translated):
        raise ValueError("Paired prediction lists must have equal length")
    changes = [a != b for a, b in zip(english, translated)]
    return {
        "sample_count": len(english),
        "agreement_with_english": float(1 - np.mean(changes)) if changes else 0.0,
        "disagreement_count": int(sum(changes)),
    }
