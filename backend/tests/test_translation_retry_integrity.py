import csv
import hashlib
from collections import Counter
from pathlib import Path

from scripts.translate_expanded_test import _write_rows


ROOT = Path(__file__).resolve().parents[2]


def test_retry_manifest_has_exact_unique_targets() -> None:
    path = ROOT / "artifacts" / "translation_pilot_retry_targets.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    pairs = {(row["cve_id"], row["language"]) for row in rows}
    assert len(rows) == 40
    assert len(pairs) == 40
    assert all(row["original_validation_status"] != "valid" for row in rows)
    assert Counter(row["primary_failure_category"] for row in rows) == {
        "missing_or_altered_technical_identifiers": 34,
        "suspiciously_short_output": 6,
    }
    assert all(
        row["language"] == "zh"
        for row in rows
        if row["primary_failure_category"] == "suspiciously_short_output"
    )


def test_retry_and_v2_preserve_expected_pair_counts() -> None:
    retry = list(csv.DictReader(
        (ROOT / "data" / "translations" / "expanded_translation_pilot_25_retry.csv").open(
            encoding="utf-8"
        )
    ))
    v2 = list(csv.DictReader(
        (ROOT / "data" / "translations" / "expanded_translation_pilot_25_v2.csv").open(
            encoding="utf-8"
        )
    ))
    assert len(retry) == len({(row["cve_id"], row["language"]) for row in retry}) == 40
    assert len(v2) == len({(row["cve_id"], row["language"]) for row in v2}) == 125
    assert all(row["review_status"] != "reviewed" for row in v2 if "review_status" in row)


def test_retry_source_hashes_match_original_text() -> None:
    path = ROOT / "data" / "translations" / "expanded_translation_pilot_25_retry.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert all(
        hashlib.sha256(row["original_text"].encode("utf-8")).hexdigest()
        == row["source_sha256"]
        for row in rows
    )


def test_retry_checkpoint_replaces_csv_atomically(tmp_path: Path) -> None:
    path = tmp_path / "retry.csv"
    _write_rows(path, [{"cve_id": "CVE-2026-0001", "language": "hi"}])

    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert len(rows) == 1
    assert rows[0]["cve_id"] == "CVE-2026-0001"
    assert not path.with_name(f"{path.name}.tmp").exists()
