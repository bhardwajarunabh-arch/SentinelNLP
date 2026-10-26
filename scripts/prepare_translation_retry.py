"""Create and verify the exact bounded retry manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("artifacts/translation_pilot_failure_analysis.json"))
    parser.add_argument("--pilot", type=Path, default=Path("data/translations/expanded_translation_pilot_25.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/translation_pilot_retry_targets.csv"))
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    pilot_rows = list(csv.DictReader(args.pilot.open(newline="", encoding="utf-8")))
    pilot_pairs = {(row["cve_id"], row["language"].lower()): row for row in pilot_rows}
    test_ids = set(json.loads(args.splits.read_text(encoding="utf-8"))["splits"]["test"])
    failures = audit["failures"]
    assert len(failures) == 40
    pairs = {(row["cve_id"], row["language"].lower()) for row in failures}
    assert len(pairs) == 40
    assert pairs <= set(pilot_pairs)
    assert pairs <= {(cve_id, language) for cve_id, language in pilot_pairs if cve_id in test_ids}
    assert not any(pilot_pairs[pair].get("validation_status") == "valid" for pair in pairs)

    fields = [
        "cve_id", "language", "original_validation_status",
        "primary_failure_category", "missing_identifiers", "source_sha256",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for failure in sorted(failures, key=lambda row: (row["cve_id"], row["language"])):
            pair = (failure["cve_id"], failure["language"].lower())
            source = hashlib.sha256(
                pilot_pairs[pair]["original_text"].encode("utf-8")
            ).hexdigest()
            writer.writerow({
                "cve_id": failure["cve_id"],
                "language": failure["language"].lower(),
                "original_validation_status": failure["stored_validation_status"],
                "primary_failure_category": failure["primary_failure_category"],
                "missing_identifiers": "|".join(failure["missing_identifiers"]),
                "source_sha256": source,
            })
    print(json.dumps({
        "row_count": len(pairs),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "languages": {language: sum(row["language"] == language for row in failures)
                      for language in sorted({row["language"] for row in failures})},
    }, indent=2))


if __name__ == "__main__":
    main()
