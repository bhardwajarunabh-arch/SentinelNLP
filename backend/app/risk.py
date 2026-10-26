from __future__ import annotations

from typing import Any

from .severity import Severity

ORDER = {Severity.LOW.value: 0, Severity.MEDIUM.value: 1, Severity.HIGH.value: 2, Severity.CRITICAL.value: 3}


def critical_risk_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("Ground-truth and prediction lists must have equal length")
    critical = [prediction for truth, prediction in zip(y_true, y_pred) if truth == Severity.CRITICAL.value]
    underpredicted = [prediction for prediction in critical if ORDER.get(prediction, -1) < ORDER[Severity.CRITICAL.value]]
    return {
        "critical_ground_truth_count": len(critical),
        "critical_recall": (
            sum(prediction == Severity.CRITICAL.value for prediction in critical) / len(critical)
            if critical else None
        ),
        "critical_underprediction_count": len(underpredicted),
        "critical_underprediction_rate": len(underpredicted) / len(critical) if critical else None,
        "undefined_reason": "No eligible Critical ground-truth examples" if not critical else None,
    }
