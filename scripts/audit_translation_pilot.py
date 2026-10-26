"""Audit failed pilot rows and calculate strictly matched language coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
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


def classify(row: dict[str, str], report: dict[str, object], computed: str) -> tuple[str, str]:
    if computed == "suspiciously_short":
        return "suspiciously_short_output", "Automated length check rejected the output."
    if row.get("error"):
        return "generation_exception", row["error"]
    if row.get("output_truncated") == "true" or row.get("source_truncated") == "true":
        return "truncation_or_chunk_reassembly", "Explicit truncation flag is present."
    missing = report["missing"]
    if missing:
        return (
            "missing_or_altered_technical_identifiers",
            f"Missing identifiers: {', '.join(missing)}",
        )
    lowered = row.get("translated_text", "").casefold()
    if re.search(r"(.)\1{7,}", lowered):
        return "repetition_or_malformed_output", "Repeated character sequence detected."
    if computed == "needs_review":
        return "wrong_language_or_source_leakage", "Language plausibility check requested review."
    if computed != "valid":
        return "other", f"Recomputed validation status: {computed}"
    return "probable_validator_false_positive", "Stored status disagreed with recomputed valid result."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--selection", type=Path, default=Path("artifacts/translation_pilot_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/translation_pilot_failure_analysis.json"))
    parser.add_argument("--coverage-output", type=Path, default=Path("artifacts/translation_pilot_matched_coverage.json"))
    args = parser.parse_args()

    records = {record.cve_id: record for record in load_records(args.records)}
    split = json.loads(args.splits.read_text(encoding="utf-8"))
    selected = json.loads(args.selection.read_text(encoding="utf-8"))["cve_ids"]
    rows = list(csv.DictReader(args.translations.open(newline="", encoding="utf-8")))
    failures = []
    status_transitions = []
    by_language: dict[str, Counter[str]] = {language: Counter() for language in TRANSLATION_LANGUAGES}
    categories_by_language: dict[str, Counter[str]] = {
        language: Counter() for language in TRANSLATION_LANGUAGES
    }
    valid_ids: dict[str, set[str]] = {language: set() for language in TRANSLATION_LANGUAGES}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cve_id = row.get("cve_id", "")
        language = row.get("language", "").lower()
        pair = (cve_id, language)
        duplicate = pair in seen
        seen.add(pair)
        source = records.get(cve_id)
        report = translation_identifier_report(
            source.description if source else row.get("original_text", ""),
            row.get("translated_text", ""),
        )
        computed, _ = validate_translation_for_language(
            source.description if source else row.get("original_text", ""),
            row.get("translated_text", ""),
            language,
        )
        status = "duplicate_cve_language" if duplicate else computed
        by_language.setdefault(language, Counter())[status] += 1
        if status == "valid":
            valid_ids.setdefault(language, set()).add(cve_id)
        if status != "valid" or row.get("validation_status") != computed:
            primary, evidence = classify(row, report, computed)
            status_transitions.append({
                "cve_id": cve_id,
                "language": language,
                "stored_status": row.get("validation_status", ""),
                "recomputed_status": status,
                "reason": evidence,
            })
            if status != "valid":
                categories_by_language.setdefault(language, Counter())[primary] += 1
                failures.append({
                    "cve_id": cve_id,
                    "language": language,
                    "severity": row.get("severity", source.severity if source else ""),
                    "primary_failure_category": primary,
                    "failure_evidence": evidence,
                    "source_text": source.description if source else row.get("original_text", ""),
                    "translated_text": row.get("translated_text", ""),
                    "validation_reason": status,
                    "stored_validation_status": row.get("validation_status", ""),
                    "required_identifiers": report["required"],
                    "preserved_identifiers": report["preserved"],
                    "missing_identifiers": report["missing"],
                    "missing_identifier_categories": report["missing_categories"],
                    "source_token_count": row.get("source_token_count", ""),
                    "output_token_count": row.get("output_token_count", ""),
                    "source_truncated": row.get("source_truncated", ""),
                    "output_truncated": row.get("output_truncated", ""),
                    "chunk_count": row.get("chunk_count", ""),
                    "generation_attempts": row.get("generation_attempts", ""),
                    "generation_seconds": row.get("generation_seconds", ""),
                    "generated_at": row.get("generated_at", ""),
                    "error": row.get("error", ""),
                    "source_description_matches_record": bool(
                        source and row.get("original_text", "") == source.description
                    ),
                })

    common_ids = set.intersection(*(valid_ids[language] for language in TRANSLATION_LANGUAGES))
    records_by_id = {record.cve_id: record for record in records.values()}
    coverage = {
        "status": "exploratory_pilot",
        "selected_cve_count": len(selected),
        "expected_pairs": len(selected) * len(TRANSLATION_LANGUAGES),
        "row_count": len(rows),
        "languages": {
            language: {
                "eligible_cve_count": len(valid_ids[language]),
                "eligible_cve_ids": sorted(valid_ids[language]),
                "severity_distribution": dict(
                    Counter(records_by_id[cve_id].severity for cve_id in valid_ids[language])
                ),
                "duplicate_rows": by_language[language]["duplicate_cve_language"],
            }
            for language in TRANSLATION_LANGUAGES
        },
        "common_all_five_language_cve_count": len(common_ids),
        "common_all_five_language_cve_ids": sorted(common_ids),
        "common_severity_distribution": dict(
            Counter(records_by_id[cve_id].severity for cve_id in common_ids)
        ),
        "missing_prediction_joins": {},
        "duplicate_prediction_joins": {},
        "provenance": {
            "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
            "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
            "translation_sha256": hashlib.sha256(args.translations.read_bytes()).hexdigest(),
            "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
            "test_split_count": len(split["splits"]["test"]),
        },
    }
    failure_report = {
        "status": "complete_row_level_audit",
        "row_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "status_transitions": status_transitions,
        "counts_by_language": {language: dict(counts) for language, counts in by_language.items()},
        "failure_categories_by_language": {
            language: dict(counts) for language, counts in categories_by_language.items()
        },
        "human_review": {"performed": False, "unreviewed_failure_count": len(failures)},
        "note": "Automated checks do not establish semantic equivalence; full source and generated text are retained for human review.",
    }
    args.output.write_text(json.dumps(failure_report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.coverage_output.write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "failure_count": len(failures),
        "common_all_five_language_cve_count": len(common_ids),
        "counts_by_language": failure_report["counts_by_language"],
    }, indent=2))


if __name__ == "__main__":
    main()
