"""Merge the immutable pilot and exact retry outputs into a new v2 artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["cve_id"], row["language"].lower()): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=Path("data/translations/expanded_translation_pilot_25.csv"))
    parser.add_argument("--retry", type=Path, default=Path("data/translations/expanded_translation_pilot_25_retry.csv"))
    parser.add_argument("--targets", type=Path, default=Path("artifacts/translation_pilot_retry_targets.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/translations/expanded_translation_pilot_25_v2.csv"))
    parser.add_argument("--comparison", type=Path, default=Path("artifacts/translation_pilot_retry_comparison.json"))
    args = parser.parse_args()
    original = read_rows(args.pilot)
    retry = read_rows(args.retry)
    targets = read_rows(args.targets)
    assert len(original) == 125 and len(original) == len(set(original))
    assert set(targets) <= set(original)
    assert set(targets) <= set(retry)

    merged: list[dict[str, str]] = []
    comparisons = []
    for pair, old in sorted(original.items()):
        row = dict(old)
        if pair in targets:
            new = retry[pair]
            expected_source_sha256 = hashlib.sha256(
                old["original_text"].encode("utf-8")
            ).hexdigest()
            assert new["source_sha256"] == expected_source_sha256
            assert new["model_revision"]
            row["original_validation_status"] = old.get("validation_status", "")
            row["original_translated_text"] = old.get("translated_text", "")
            row["retry_validation_status"] = new.get("validation_status", "")
            row["retry_translated_text"] = new.get("translated_text", "")
            row["retry_error"] = new.get("error", "")
            row["retry_provenance"] = new.get("provenance", "")
            row.update(new)
            row["original_validation_status"] = old.get("validation_status", "")
            row["original_translated_text"] = old.get("translated_text", "")
            row["retry_validation_status"] = new.get("validation_status", "")
            row["retry_translated_text"] = new.get("translated_text", "")
            row["retry_error"] = new.get("error", "")
            row["retry_provenance"] = new.get("provenance", "")
        merged.append(row)
        if pair in targets:
            retry_status = retry[pair].get("validation_status", "missing")
            missing = retry[pair].get("missing_identifiers", "")
            comparisons.append({
                "cve_id": pair[0],
                "language": pair[1],
                "original_status": old.get("validation_status", ""),
                "retry_status": retry_status,
                "final_consolidated_status": row.get("validation_status", ""),
                "identifier_preservation_passed": not bool(missing),
                "output_length_and_truncation_passed": (
                    row.get("output_truncated", "").lower() != "true"
                    and retry_status != "suspiciously_short"
                ),
                "further_review_required": True,
                "unresolved_error": retry[pair].get("error", "") or missing,
            })
    assert len(merged) == 125
    fields = sorted(set().union(*(row.keys() for row in merged)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    report = {
        "status": "complete",
        "target_count": len(comparisons),
        "comparisons": comparisons,
        "original_sha256": hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
        "retry_sha256": hashlib.sha256(args.retry.read_bytes()).hexdigest(),
        "consolidated_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "human_reviewed_count": sum(row.get("review_status") == "reviewed" for row in merged),
        "note": "Original rows and outputs are preserved in audit columns; no row is marked human-reviewed.",
    }
    args.comparison.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"row_count": len(merged), "target_count": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
