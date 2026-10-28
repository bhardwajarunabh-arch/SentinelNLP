from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = {"en": "English", "hi": "Hindi", "zh": "Simplified Chinese", "es": "Spanish", "ar": "Arabic", "fr": "French"}
TRANSLATION_LANGUAGES = ("hi", "zh", "es", "ar", "fr")

_TECHNICAL_IDENTIFIER_RE = re.compile(
    r"<[A-Za-z][A-Za-z0-9_]*>"
    r"|CVE-\d{4}-\d+"
    r"|(?=[A-Za-z0-9_./:+#-]*\d)[A-Za-z0-9_./:+#-]+"
    r"|/?(?:[A-Za-z0-9_-]+/)+[A-Za-z0-9_.-]+"
    r"|[A-Za-z0-9_-]+\.[A-Za-z]{1,8}\b"
    r"|[A-Za-z0-9]+_[A-Za-z0-9_]+\b"
)
_IDENTIFIER_PLACEHOLDER_RE = re.compile(r"<SENTINEL_ID_\d+>")
_IDENTIFIER_PLACEHOLDER_LIKE_RE = re.compile(r"<?SENTINEL_ID_[A-Za-z0-9_]+>?")
_IDENTIFIER_TRAILING_PUNCTUATION = ".,;:!?)]}"
_IDENTIFIER_CATEGORIES = (
    ("cve_id", re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)),
    ("version", re.compile(r"^v?\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.]+)?$", re.IGNORECASE)),
    ("hash", re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)),
    ("path_or_filename", re.compile(r"(?:/|\\)|\.[A-Za-z]{1,8}$")),
    ("symbol_or_product", re.compile(r"[_-]|^[A-Za-z][A-Za-z0-9]*$")),
)
_SCRIPT_RANGES = {
    "hi": re.compile(r"[\u0900-\u097F]"),
    "zh": re.compile(r"[\u3400-\u9FFF]"),
    "ar": re.compile(r"[\u0600-\u06FF]"),
}


@dataclass(frozen=True)
class TranslationRecord:
    cve_id: str
    language: str
    original_text: str
    translated_text: str
    model: str
    provenance: str
    validation_status: str


def validate_translation(
    original: str, translated: str, language: str = "unknown"
) -> str:
    value = translated.strip()
    if not value:
        return "missing"
    lowered = value.casefold()
    if lowered.startswith(
        (
            "[translation",
            "[hindi translation",
            "[chinese (simplified) translation",
            "[spanish translation",
            "[arabic translation",
            "[french translation",
            "translation failed",
            "error:",
        )
    ):
        return "placeholder_or_error"
    if value == original.strip():
        return "source_identical"
    if language == "zh":
        if len(_SCRIPT_RANGES["zh"].findall(value)) < 2:
            return "suspiciously_short"
    elif len(value.split()) < 2:
        return "suspiciously_short"
    identifiers = technical_identifiers(original)
    missing = [identifier for identifier in identifiers if identifier not in value]
    if missing:
        return "technical_identifier_missing"
    return "valid" if _language_plausibility(value, "unknown") else "needs_review"


def _language_plausibility(text: str, language: str) -> bool:
    """Apply a conservative script check; technical English tokens are allowed."""
    if language in _SCRIPT_RANGES:
        return bool(_SCRIPT_RANGES[language].search(text))
    if language in {"es", "fr"}:
        letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
        return len(letters) >= 2
    return True


def technical_identifiers(text: str) -> list[str]:
    return sorted(
        {
            token.strip(_IDENTIFIER_TRAILING_PUNCTUATION)
            for token in _TECHNICAL_IDENTIFIER_RE.findall(text)
            if token.strip(_IDENTIFIER_TRAILING_PUNCTUATION)
            and token.strip(_IDENTIFIER_TRAILING_PUNCTUATION).casefold() != "n/a"
        }
    )


def protect_technical_identifiers(text: str) -> tuple[str, dict[str, str]]:
    identifiers = sorted(technical_identifiers(text), key=len, reverse=True)
    if not identifiers:
        return text, {}

    identifier_pattern = re.compile("|".join(re.escape(item) for item in identifiers))
    replacements: dict[str, str] = {}

    def replace_identifier(match: re.Match[str]) -> str:
        placeholder = f"<SENTINEL_ID_{len(replacements)}>"
        replacements[placeholder] = match.group(0)
        return placeholder

    return identifier_pattern.sub(replace_identifier, text), replacements


def split_technical_placeholders(text: str) -> list[tuple[str, str]]:
    """Return alternating literal-text and internal-placeholder parts."""
    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in _IDENTIFIER_PLACEHOLDER_RE.finditer(text):
        if match.start() > cursor:
            parts.append(("text", text[cursor : match.start()]))
        parts.append(("placeholder", match.group(0)))
        cursor = match.end()
    if cursor < len(text):
        parts.append(("text", text[cursor:]))
    if not parts:
        parts.append(("text", text))
    return parts


def validate_technical_placeholders(
    text_or_chunks: str | list[str], replacements: dict[str, str]
) -> list[str]:
    chunks = [text_or_chunks] if isinstance(text_or_chunks, str) else text_or_chunks
    counts = Counter(
        placeholder
        for chunk in chunks
        for placeholder in _IDENTIFIER_PLACEHOLDER_RE.findall(chunk)
    )
    issues = [
        f"Missing placeholder {placeholder} for {identifier}"
        for placeholder, identifier in replacements.items()
        if counts[placeholder] == 0
    ]
    issues.extend(
        f"Duplicated placeholder {placeholder} for {replacements[placeholder]} (found {count})"
        for placeholder, count in counts.items()
        if placeholder in replacements and count != 1
    )
    issues.extend(
        f"Unknown placeholder {placeholder}"
        for chunk in chunks
        for placeholder in _IDENTIFIER_PLACEHOLDER_LIKE_RE.findall(chunk)
        if placeholder not in replacements
    )
    return issues


def restore_technical_identifiers(
    text: str, replacements: dict[str, str]
) -> tuple[str, list[str]]:
    issues = validate_technical_placeholders(text, replacements)
    if issues:
        counts = Counter(_IDENTIFIER_PLACEHOLDER_RE.findall(text))
        invalid = [
            identifier
            for placeholder, identifier in replacements.items()
            if counts[placeholder] != 1
        ]
        invalid.extend(
            f"Unknown placeholder: {placeholder}"
            for placeholder in _IDENTIFIER_PLACEHOLDER_LIKE_RE.findall(text)
            if placeholder not in replacements
        )
        return text, invalid
    restored = _IDENTIFIER_PLACEHOLDER_RE.sub(
        lambda match: replacements[match.group(0)], text
    )
    return restored, []


def identifier_categories(identifiers: list[str]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {name: [] for name, _ in _IDENTIFIER_CATEGORIES}
    for identifier in identifiers:
        for name, pattern in _IDENTIFIER_CATEGORIES:
            if pattern.search(identifier):
                categories[name].append(identifier)
                break
    return categories


def translation_identifier_report(original: str, translated: str) -> dict[str, object]:
    required = technical_identifiers(original)
    preserved = [identifier for identifier in required if identifier in translated]
    report: dict[str, list[str]] = {
        "required": required,
        "preserved": preserved,
        "missing": [identifier for identifier in required if identifier not in translated],
    }
    report["required_categories"] = identifier_categories(required)
    report["missing_categories"] = identifier_categories(report["missing"])
    return report


def _count_identifier_occurrences(text: str, identifier: str) -> int:
    count = 0
    identifier_match = re.compile(re.escape(identifier))
    for match in identifier_match.finditer(text):
        start, end = match.span()
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if before.isascii() and (before.isalnum() or before == "_"):
            continue
        if (
            not identifier.startswith("<")
            and start >= 2
            and before in ".-/"
            and text[start - 2].isascii()
            and (text[start - 2].isalnum() or text[start - 2] == "_")
        ):
            continue
        if after.isascii() and (after.isalnum() or after == "_"):
            continue
        if not identifier.endswith(">") and after in ".-/" and end + 1 < len(text):
            following = text[end + 1]
            if following.isascii() and (following.isalnum() or following == "_"):
                continue
        count += 1
    return count


def technical_identifier_occurrence_report(
    original: str, translated: str
) -> dict[str, object]:
    _, replacements = protect_technical_identifiers(original)
    expected = Counter(replacements.values())
    actual = {
        identifier: _count_identifier_occurrences(translated, identifier)
        for identifier in expected
    }
    missing = {
        identifier: count - actual[identifier]
        for identifier, count in expected.items()
        if actual[identifier] < count
    }
    duplicated = {
        identifier: actual[identifier] - count
        for identifier, count in expected.items()
        if actual[identifier] > count
    }
    unexpected = sorted(
        set(technical_identifiers(translated))
        - set(technical_identifiers(original))
    )
    issues = [
        f"Missing {count} occurrence(s) of identifier {identifier}"
        for identifier, count in missing.items()
    ]
    issues.extend(
        f"Duplicated identifier {identifier} {count} extra time(s)"
        for identifier, count in duplicated.items()
    )
    issues.extend(f"Unexpected technical identifier {identifier}" for identifier in unexpected)
    return {
        "expected_occurrences": dict(expected),
        "actual_occurrences": actual,
        "missing": missing,
        "duplicated": duplicated,
        "unexpected": unexpected,
        "issues": issues,
    }


def validate_translation_for_language(original: str, translated: str, language: str) -> tuple[str, list[str]]:
    status = validate_translation(original, translated, language)
    if status != "valid":
        if status == "technical_identifier_missing":
            return "failed", translation_identifier_report(original, translated)["missing"]
        return status, []
    if not _language_plausibility(translated, language):
        return "needs_review", []
    report = translation_identifier_report(original, translated)
    return "valid", report["preserved"]


def is_reliable_translation(original: str, translated: str) -> bool:
    """Reject blank, source-identical, and generated placeholder text."""
    value = translated.strip()
    lowered = value.casefold()
    return bool(value and validate_translation(original, translated) == "valid")


def import_translations(path: Path, output_path: Path) -> int:
    required = {"cve_id", "language", "original_text", "translated_text"}
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Translation CSV must contain: {', '.join(sorted(required))}")
        for row in reader:
            language = row["language"].lower()
            if language not in LANGUAGES:
                raise ValueError(f"Unsupported language: {language}")
            translated = row["translated_text"].strip()
            rows.append(
                {
                    **row,
                    "language": language,
                    "model": row.get("model", "unknown"),
                    "provenance": row.get("provenance", "imported"),
                    "validation_status": validate_translation(
                        row["original_text"], translated, language
                    ),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows))) if rows else sorted(required))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
