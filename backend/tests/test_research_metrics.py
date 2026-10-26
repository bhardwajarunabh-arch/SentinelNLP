from scripts.research_metrics import (
    classification_report_verified,
    critical_metrics,
    wilson_interval,
)
from scripts.translate_expanded_test import _chunk_text


class _WordTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[str]]:
        return {"input_ids": text.split()}


def test_research_metrics_use_explicit_label_order() -> None:
    result = classification_report_verified(
        ["Critical", "Low", "Medium"],
        ["High", "Low", "Medium"],
        [[0.1, 0.8, 0.05, 0.05], [0.05, 0.05, 0.8, 0.1], [0.05, 0.1, 0.1, 0.75]],
    )

    assert result["labels"] == ["Critical", "High", "Low", "Medium"]
    assert result["confusion_matrix"] == [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_research_metrics_critical_underprediction_denominators() -> None:
    result = critical_metrics(
        ["Critical", "Critical", "Critical", "Low"],
        ["High", "Medium", "Critical", "Low"],
    )

    assert result["actual_critical_count"] == 3
    assert result["underprediction_count"] == 2
    assert result["underprediction_rate"] == 2 / 3
    assert result["critical_to_high"] == {"count": 1, "denominator": 3}
    assert result["critical_to_medium"] == {"count": 1, "denominator": 3}
    assert result["critical_to_low"] == {"count": 0, "denominator": 3}


def test_wilson_interval_has_no_interval_without_observations() -> None:
    assert wilson_interval(0, 0) is None


def test_long_translation_input_is_chunked_without_silent_truncation() -> None:
    chunks = _chunk_text("one two three four five six", _WordTokenizer(), 3)

    assert len(chunks) >= 2
    assert " ".join(chunks).split() == "one two three four five six".split()
