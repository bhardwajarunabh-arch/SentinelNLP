"""Create a no-network translation import file for the current test split.

This command never translates text. It reuses only reliable, source-matched
rows from an existing import and leaves all other target cells blank.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data_pipeline import load_records
from backend.app.translations import is_reliable_translation, validate_translation

LANGUAGES_TO_IMPORT = ("hi", "zh", "es", "ar", "fr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a manual translation import template.")
    parser.add_argument("--records", type=Path, default=Path("data/processed/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/cve_splits.json"))
    parser.add_argument("--existing", type=Path, default=Path("data/translations/translations.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/translations/current_test_translations.csv"))
    args = parser.parse_args()

    records = {record.cve_id: record for record in load_records(args.records)}
    test_ids = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]["test"]
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if args.existing.is_file():
        with args.existing.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing[(row["cve_id"], row["language"].lower())] = row

    rows: list[dict[str, str]] = []
    for cve_id in test_ids:
        record = records[cve_id]
        for language in LANGUAGES_TO_IMPORT:
            old = existing.get((cve_id, language))
            translated = old.get("translated_text", "").strip() if old else ""
            source_matches = old is not None and old.get("original_text", "") == record.description
            reliable = source_matches and is_reliable_translation(record.description, translated)
            validation = validate_translation(record.description, translated) if reliable else (
                "missing_translation" if not translated else "unreliable_existing_translation"
            )
            rows.append(
                {
                    "cve_id": cve_id,
                    "language": language,
                    "model": old.get("model", "manual-or-external") if reliable and old else "unprovided",
                    "original_text": record.description,
                    "provenance": "reused_source_matched" if reliable else "manual_or_external_import_required",
                    "translated_text": translated if reliable else "",
                    "validation_status": validation,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "translation_import_template",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        "test_cve_ids": test_ids,
        "languages": list(LANGUAGES_TO_IMPORT),
        "row_count": len(rows),
        "reused_reliable_rows": sum(row["validation_status"] == "valid" for row in rows),
        "missing_or_excluded_rows": sum(row["validation_status"] != "valid" for row in rows),
        "validation_rule": "technical identifiers must be preserved; placeholders and source-identical text are rejected",
        "note": "No translation service was called. Blank rows require manually or externally prepared translations.",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
