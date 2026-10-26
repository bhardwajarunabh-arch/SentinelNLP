"""Validate generated/imported translations against the current test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data_pipeline import load_records
from backend.app.translations import (
    TRANSLATION_LANGUAGES,
    translation_identifier_report,
    validate_translation_for_language,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_file(translations: Path, records_path: Path, splits_path: Path) -> dict[str, object]:
    records = {record.cve_id: record for record in load_records(records_path)}
    manifest = json.loads(splits_path.read_text(encoding="utf-8"))
    test_ids = set(manifest["splits"]["test"])
    rows = list(csv.DictReader(translations.open(newline="", encoding="utf-8")))
    seen: set[tuple[str, str]] = set()
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    identifier_totals: Counter[str] = Counter()
    validation_false_positives: list[dict[str, str]] = []
    truncation_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        cve_id = row.get("cve_id", "")
        language = row.get("language", "").lower()
        pair = (cve_id, language)
        status = row.get("validation_status", "missing")
        if row.get("source_truncated") == "true":
            truncation_counts[language] += 1
        if row.get("output_truncated") == "true":
            truncation_counts[f"{language}:output"] += 1
        duplicate = pair in seen
        if duplicate:
            status = "duplicate_cve_language"
        seen.add(pair)
        if duplicate:
            pass
        elif language not in TRANSLATION_LANGUAGES:
            status = "unsupported_language"
        elif cve_id not in records:
            status = "unknown_cve"
        elif cve_id not in test_ids:
            status = "not_in_test_split"
        elif row.get("original_text", "") != records[cve_id].description:
            status = "source_mismatch"
        elif row.get("severity", records[cve_id].severity) != records[cve_id].severity:
            status = "severity_mismatch"
        else:
            computed, identifiers = validate_translation_for_language(
                records[cve_id].description, row.get("translated_text", ""), language
            )
            identifier_report = translation_identifier_report(
                records[cve_id].description, row.get("translated_text", "")
            )
            identifier_totals[language] += len(identifier_report["preserved"])
            # Recompute automated status from current source/text. A stale
            # producer status must not turn a valid translation into a
            # rejection; human review is reported separately below.
            status = computed
            if computed == "valid" and row.get("validation_status") not in {"", "valid"}:
                validation_false_positives.append(
                    {
                        "cve_id": cve_id,
                        "language": language,
                        "stored_status": row.get("validation_status", ""),
                        "reason": "stored_status_disagrees_with_recomputed_identifier_check",
                    }
                )
        by_language[language][status] += 1
        if status != "valid":
            rejection_reasons[status] += 1
        if len(samples[language]) < 3:
            samples[language].append(
                {
                    "cve_id": cve_id,
                    "status": status,
                    "translated_text": row.get("translated_text", ""),
                }
            )

    expected = len(test_ids) * len(TRANSLATION_LANGUAGES)
    return {
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "expected_row_count": expected,
        "complete_grid": len(rows) == expected and len(seen) == expected,
        "languages": {
            language: {
                "counts": dict(by_language[language]),
                "valid_count": by_language[language]["valid"],
                "identifier_preservation_observations": identifier_totals[language],
                "truncation_warnings": {
                    "source": truncation_counts[language],
                    "output": truncation_counts[f"{language}:output"],
                },
                "identifier_missing_examples": [
                {
                    "cve_id": row.get("cve_id", ""),
                    "missing": translation_identifier_report(
                        records[row["cve_id"]].description,
                        row.get("translated_text", ""),
                    )["missing"],
                }
                for row in rows
                if row.get("language", "").lower() == language
                and row.get("cve_id", "") in records
                and translation_identifier_report(
                    records[row["cve_id"]].description,
                    row.get("translated_text", ""),
                )["missing"]
                ][:3],
                "samples": samples[language],
            }
            for language in TRANSLATION_LANGUAGES
        },
        "rejection_reasons": dict(rejection_reasons),
        "validation_false_positives": validation_false_positives,
        "truncation_warnings": dict(truncation_counts),
        "automated_checks": [
            "test-split CVE membership",
            "exact source-description matching",
            "supported language codes",
            "duplicate CVE-language detection",
            "severity-label inheritance",
            "technical-identifier preservation",
            "placeholder/source-identical rejection",
        ],
        "human_review": {
            "performed": False,
            "note": "Automated checks do not establish human translation quality or semantic faithfulness.",
        },
        "provenance": {
            "translation_sha256": _sha256(translations),
            "dataset_sha256": _sha256(records_path),
            "split_sha256": _sha256(splits_path),
            "test_cve_count": len(test_ids),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate translations against the expanded test split.")
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_file(args.translations, args.records, args.splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
