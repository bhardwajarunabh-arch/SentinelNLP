"""Validate the bounded GPU retry and compare its effect on pilot coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data_pipeline import load_records
from backend.app.translations import (
    TRANSLATION_LANGUAGES,
    technical_identifier_occurrence_report,
    translation_identifier_report,
    validate_technical_placeholders,
    validate_translation_for_language,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("artifacts/translation_pilot_failure_analysis.json"))
    parser.add_argument("--targets", type=Path, default=Path("artifacts/translation_pilot_retry_targets.csv"))
    parser.add_argument("--previous-validation", type=Path)
    parser.add_argument("--pilot", type=Path, default=Path("data/translations/expanded_translation_pilot_25.csv"))
    parser.add_argument("--baseline-retry", type=Path)
    parser.add_argument("--retry", type=Path, default=Path("data/translations/expanded_translation_pilot_25_retry_gpu.csv"))
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/translations/expanded_translation_pilot_25_retry_gpu.manifest.json"))
    parser.add_argument("--validation-output", type=Path, default=Path("artifacts/translation_pilot_retry_validation_gpu.json"))
    parser.add_argument("--comparison-output", type=Path, default=Path("artifacts/translation_pilot_retry_comparison_gpu.json"))
    args = parser.parse_args()

    failures = json.loads(args.audit.read_text(encoding="utf-8"))["failures"]
    target_rows = read_csv(args.targets)
    target_pairs = {(row["cve_id"], row["language"].lower()) for row in target_rows}
    if len(target_rows) != len(target_pairs):
        raise ValueError("Retry target manifest contains duplicate CVE-language pairs")
    previous_validation_report = None
    if args.previous_validation is not None:
        previous_validation_report = json.loads(
            args.previous_validation.read_text(encoding="utf-8")
        )
        expected_pairs = {
            (row["cve_id"], row["language"].lower())
            for row in previous_validation_report["rows"]
            if row["status"] != "valid"
        }
        if target_pairs != expected_pairs:
            raise ValueError(
                "Retry target manifest does not exactly match the previous unresolved pairs"
            )
    else:
        expected_pairs = {
            (row["cve_id"], row["language"].lower()) for row in failures
        }
        if target_pairs != expected_pairs:
            raise ValueError(
                "Retry target manifest does not exactly match the audited failure pairs"
            )

    records = {record.cve_id: record for record in load_records(args.records)}
    test_ids = set(json.loads(args.splits.read_text(encoding="utf-8"))["splits"]["test"])
    pilot_rows = read_csv(args.pilot)
    retry_rows = read_csv(args.retry)
    retry_by_pair = {(row["cve_id"], row["language"].lower()): row for row in retry_rows}
    if len(retry_rows) != len(target_pairs) or set(retry_by_pair) != target_pairs:
        raise ValueError("Retry output must contain exactly one row for every requested pair")
    baseline_by_pair = {}
    if args.baseline_retry is not None:
        baseline_rows = read_csv(args.baseline_retry)
        baseline_by_pair = {
            (row["cve_id"], row["language"].lower()): row for row in baseline_rows
        }

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation_rows: list[dict[str, Any]] = []
    retry_valid_by_language: Counter[str] = Counter()
    for row in retry_rows:
        pair = (row["cve_id"], row["language"].lower())
        source = records.get(row["cve_id"])
        issues: list[str] = []
        if source is None or row["cve_id"] not in test_ids:
            issues.append("Pair is not in the fixed test split")
        elif row.get("original_text") != source.description:
            issues.append("Original text does not match the fixed source record")
        if source and row.get("source_sha256") != hashlib.sha256(
            source.description.encode("utf-8")
        ).hexdigest():
            issues.append("Source SHA-256 does not match the fixed source record")

        status, _ = validate_translation_for_language(
            source.description if source else row.get("original_text", ""),
            row.get("translated_text", ""),
            pair[1],
        )
        identifier_report = translation_identifier_report(
            source.description if source else row.get("original_text", ""),
            row.get("translated_text", ""),
        )
        occurrence_report = technical_identifier_occurrence_report(
            source.description if source else row.get("original_text", ""),
            row.get("translated_text", ""),
        )
        if validate_technical_placeholders(row.get("translated_text", ""), {}):
            issues.append("Residual identifier placeholder")
        if row.get("missing_placeholders", "0") != "0" or row.get("placeholder_error"):
            issues.append("Identifier placeholder restoration failed")
        if identifier_report["missing"]:
            issues.append("Required technical identifier missing or altered")
        if occurrence_report["issues"]:
            issues.append(
                "Technical identifier integrity failed: "
                + " | ".join(occurrence_report["issues"])
            )
        if row.get("source_truncated", "").lower() == "true":
            issues.append("Source text was truncated")
        if row.get("output_truncated", "").lower() == "true":
            issues.append("Translation output was truncated")
        if row.get("error"):
            issues.append(f"Generation error: {row['error']}")
        if status != "valid":
            issues.append(f"Automated translation validation: {status}")
        if issues:
            status = "failed"
        else:
            retry_valid_by_language[pair[1]] += 1
        validation_rows.append({
            "cve_id": pair[0],
            "language": pair[1],
            "status": status,
            "issues": issues,
            "required_identifier_count": len(identifier_report["required"]),
            "preserved_identifier_count": len(identifier_report["preserved"]),
            "missing_identifiers": identifier_report["missing"],
            "identifier_occurrence_report": occurrence_report,
            "residual_placeholder": bool(
                validate_technical_placeholders(
                    row.get("translated_text", ""), {}
                )
            ),
            "generation_seconds": row.get("generation_seconds", ""),
            "error": row.get("error", ""),
        })

    pilot_by_pair = {(row["cve_id"], row["language"].lower()): row for row in pilot_rows}
    original_valid: dict[str, set[str]] = {language: set() for language in TRANSLATION_LANGUAGES}
    updated_valid: dict[str, set[str]] = {language: set() for language in TRANSLATION_LANGUAGES}
    for row in pilot_rows:
        cve_id, language = row["cve_id"], row["language"].lower()
        if language not in original_valid:
            continue
        source = records.get(cve_id)
        text = row.get("translated_text", "")
        source_text = source.description if source else row.get("original_text", "")
        status, _ = validate_translation_for_language(
            source_text, text, language
        )
        pilot_integrity = technical_identifier_occurrence_report(source_text, text)
        if status == "valid" and not pilot_integrity["issues"]:
            original_valid[language].add(cve_id)
        replacement = retry_by_pair.get((cve_id, language))
        if replacement:
            outcome = next(
                item for item in validation_rows
                if item["cve_id"] == cve_id and item["language"] == language
            )
            if outcome["status"] == "valid":
                updated_valid[language].add(cve_id)
        elif baseline_by_pair.get((cve_id, language)) is not None:
            baseline = baseline_by_pair[(cve_id, language)]
            baseline_status, _ = validate_translation_for_language(
                source_text,
                baseline.get("translated_text", ""),
                language,
            )
            baseline_integrity = technical_identifier_occurrence_report(
                source_text, baseline.get("translated_text", "")
            )
            if (
                baseline_status == "valid"
                and not baseline_integrity["issues"]
                and baseline.get("validation_status") == "valid"
                and baseline.get("missing_placeholders", "0") == "0"
                and not baseline.get("placeholder_error")
                and baseline.get("source_truncated", "").lower() != "true"
                and baseline.get("output_truncated", "").lower() != "true"
                and not baseline.get("error")
            ):
                updated_valid[language].add(cve_id)
        elif status == "valid" and not pilot_integrity["issues"]:
            updated_valid[language].add(cve_id)

    common_before = set.intersection(*(original_valid[language] for language in TRANSLATION_LANGUAGES))
    common_after = set.intersection(*(updated_valid[language] for language in TRANSLATION_LANGUAGES))
    per_language = {}
    for language in TRANSLATION_LANGUAGES:
        target_ids = {cve_id for cve_id, code in target_pairs if code == language}
        original_target_valid = sum(
            cve_id in original_valid[language] for cve_id in target_ids
        )
        per_language[language] = {
            "target_pair_count": len(target_ids),
            "original_target_valid_count": original_target_valid,
            "retry_target_valid_count": retry_valid_by_language[language],
            "retry_target_unresolved_count": len(target_ids) - retry_valid_by_language[language],
            "retry_issue_counts": dict(Counter(
                issue
                for row in validation_rows
                if row["language"] == language
                for issue in row["issues"]
            )),
            "original_pilot_valid_count": len(original_valid[language]),
            "updated_pilot_valid_count": len(updated_valid[language]),
        }

    valid_count = sum(row["status"] == "valid" for row in validation_rows)
    previous_status_counts = Counter(
        row["status"] for row in (previous_validation_report or {}).get("rows", [])
    )
    validation_report = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_pair_count": len(target_pairs),
        "valid_count": valid_count,
        "unresolved_count": len(validation_rows) - valid_count,
        "previous_unresolved_count": (
            sum(count for status, count in previous_status_counts.items() if status != "valid")
            if previous_validation_report is not None else None
        ),
        "previous_status_counts": dict(previous_status_counts),
        "status_counts": dict(Counter(row["status"] for row in validation_rows)),
        "per_language": per_language,
        "cuda_used": bool(manifest.get("generation", {}).get("cuda_used")),
        "model_revision": manifest.get("model_revision"),
        "rows": validation_rows,
        "provenance": {
            "audit_sha256": sha256(args.audit),
            "targets_sha256": sha256(args.targets),
            "retry_sha256": sha256(args.retry),
            "records_sha256": sha256(args.records),
            "splits_sha256": sha256(args.splits),
            "previous_validation_sha256": (
                sha256(args.previous_validation)
                if args.previous_validation is not None else None
            ),
        },
    }
    comparison_report = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cuda_used": bool(manifest.get("generation", {}).get("cuda_used")),
        "model_revision": manifest.get("model_revision"),
        "elapsed_seconds": manifest.get("profiling", {}).get("total_elapsed_seconds"),
        "aggregate_pairs_per_second": manifest.get("profiling", {}).get(
            "aggregate_pairs_per_second"
        ),
        "generation_seconds_by_language": manifest.get("profiling", {}).get(
            "generation_seconds_by_language", {}
        ),
        "pairs_per_second_by_language": manifest.get("profiling", {}).get(
            "pairs_per_second_by_language", {}
        ),
        "cuda_peak_allocated_bytes": manifest.get("profiling", {}).get(
            "cuda_peak_allocated_bytes"
        ),
        "cuda_peak_reserved_bytes": manifest.get("profiling", {}).get(
            "cuda_peak_reserved_bytes"
        ),
        "per_language": per_language,
        "retry_valid_count": valid_count,
        "retry_unresolved_count": len(validation_rows) - valid_count,
        "previous_retry_status_counts": dict(previous_status_counts),
        "common_five_language_cve_count_before": len(common_before),
        "common_five_language_cve_ids_before": sorted(common_before),
        "common_five_language_cve_count_after": len(common_after),
        "common_five_language_cve_ids_after": sorted(common_after),
        "baseline_retry_valid_row_count": sum(
            row.get("validation_status") == "valid"
            for row in baseline_by_pair.values()
        ),
        "unresolved_pairs": [
            {"cve_id": row["cve_id"], "language": row["language"], "issues": row["issues"]}
            for row in validation_rows if row["status"] != "valid"
        ],
        "human_reviewed": False,
        "human_review_required": True,
        "provenance": {
            "pilot_sha256": sha256(args.pilot),
            "retry_sha256": sha256(args.retry),
            "manifest_sha256": sha256(args.manifest),
            "baseline_retry_sha256": (
                sha256(args.baseline_retry) if args.baseline_retry is not None else None
            ),
        },
    }
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_output.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
    print(json.dumps({
        "target_pair_count": len(target_pairs),
        "valid_count": valid_count,
        "unresolved_count": len(validation_rows) - valid_count,
        "common_five_language_before": len(common_before),
        "common_five_language_after": len(common_after),
    }, indent=2))


if __name__ == "__main__":
    main()
