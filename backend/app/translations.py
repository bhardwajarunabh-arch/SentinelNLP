from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = {"en": "English", "hi": "Hindi", "zh": "Simplified Chinese", "es": "Spanish", "ar": "Arabic", "fr": "French"}
TRANSLATION_LANGUAGES = ("hi", "zh", "es", "ar", "fr")

_TECHNICAL_IDENTIFIER_RE = re.compile(
    r"CVE-\d{4}-\d+"
    r"|(?=[A-Za-z0-9_./:+#-]*\d)[A-Za-z0-9_./:+#-]+"
    r"|/?(?:[A-Za-z0-9_-]+/)+[A-Za-z0-9_.-]+"
    r"|[A-Za-z0-9_-]+\.[A-Za-z]{1,8}\b"
    r"|[A-Za-z0-9]+_[A-Za-z0-9_]+\b"
)
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


def validate_translation(original: str, translated: str) -> str:
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
    if len(value.split()) < 2:
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
        }
    )


def protect_technical_identifiers(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    protected = text
    identifiers = sorted(technical_identifiers(text), key=len, reverse=True)
    for index, identifier in enumerate(identifiers):
        placeholder = f"<SENTINEL_ID_{index}>"
        protected = protected.replace(identifier, placeholder)
        replacements[placeholder] = identifier
    return protected, replacements


def restore_technical_identifiers(
    text: str, replacements: dict[str, str]
) -> tuple[str, list[str]]:
    restored = text
    missing: list[str] = []
    for placeholder, identifier in replacements.items():
        if placeholder not in restored:
            missing.append(identifier)
            continue
        restored = restored.replace(placeholder, identifier)
    return restored, missing


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


def validate_translation_for_language(original: str, translated: str, language: str) -> tuple[str, list[str]]:
    status = validate_translation(original, translated)
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
                    "validation_status": validate_translation(row["original_text"], translated),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows))) if rows else sorted(required))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
