from enum import StrEnum


class Severity(StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


def severity_from_cvss(score: float | int | None) -> Severity | None:
    """Map a valid CVSS v3.1 base score to the project's four classes."""
    if score is None:
        return None
    numeric_score = float(score)
    if not 0.1 <= numeric_score <= 10.0:
        return None
    if numeric_score <= 3.9:
        return Severity.LOW
    if numeric_score <= 6.9:
        return Severity.MEDIUM
    if numeric_score <= 8.9:
        return Severity.HIGH
    return Severity.CRITICAL


def validate_cvss_label(score: float | int | None, label: str | None) -> bool:
    expected = severity_from_cvss(score)
    return expected is not None and expected.value.lower() == (label or "").lower()
