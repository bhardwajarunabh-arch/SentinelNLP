from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


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
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
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
