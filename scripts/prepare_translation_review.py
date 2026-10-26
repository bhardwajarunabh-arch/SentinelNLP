"""Create a non-destructive human-review worksheet from pilot translations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.translations.open(newline="", encoding="utf-8")))
    for row in rows:
        try:
            tokens = int(row.get("source_token_count", "0") or 0)
        except ValueError:
            tokens = 0
        row["_length_bucket"] = "short" if tokens < 80 else "long"
    rows.sort(key=lambda row: (row.get("language", ""), row.get("severity", ""), row["_length_bucket"], row.get("cve_id", "")))
    selected = []
    seen_strata: set[tuple[str, str, str]] = set()
    for row in rows:
        stratum = (row.get("language", ""), row.get("severity", ""), row["_length_bucket"])
        if stratum not in seen_strata:
            selected.append(row)
            seen_strata.add(stratum)
    for row in rows:
        if len(selected) >= min(25, len(rows)):
            break
        if row not in selected:
            selected.append(row)
    fields = [
        "cve_id", "language", "severity", "original_text", "translated_text",
        "validation_status", "missing_identifiers", "source_token_count",
        "output_token_count", "chunk_count", "review_status", "reviewer_notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                **{field: row.get(field, "") for field in fields},
                "review_status": "unreviewed",
                "reviewer_notes": "",
            })
    print(f"wrote {len(selected)} review rows to {args.output}")


if __name__ == "__main__":
    main()
