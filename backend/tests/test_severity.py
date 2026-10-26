import pytest

from backend.app.severity import Severity, severity_from_cvss, validate_cvss_label


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.1, Severity.LOW),
        (3.9, Severity.LOW),
        (4.0, Severity.MEDIUM),
        (6.9, Severity.MEDIUM),
        (7.0, Severity.HIGH),
        (8.9, Severity.HIGH),
        (9.0, Severity.CRITICAL),
        (10.0, Severity.CRITICAL),
    ],
)
def test_severity_boundaries(score: float, expected: Severity) -> None:
    assert severity_from_cvss(score) == expected


@pytest.mark.parametrize("score", [None, 0, -1, 10.1])
def test_invalid_scores_are_excluded(score: float | None) -> None:
    assert severity_from_cvss(score) is None


def test_cvss_label_validation() -> None:
    assert validate_cvss_label(9.8, "Critical")
    assert not validate_cvss_label(9.8, "High")
