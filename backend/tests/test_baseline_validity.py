import csv
import json
from pathlib import Path

from backend.app.data_pipeline import load_records, normalized_description
from backend.app.translations import (
    is_reliable_translation,
    translation_identifier_report,
    validate_translation,
)
from scripts.validate_translations import validate_file
from scripts.train_baseline import training_records
from scripts.translate_expanded_test import pending_cve_ids


def test_training_workflow_selects_only_train_cves(tmp_path: Path) -> None:
    records = load_records(Path("data/processed/nvd_records.json"))
    manifest = json.loads(Path("data/splits/cve_splits.json").read_text(encoding="utf-8"))
    selected = training_records(records, manifest)
    train_ids = set(manifest["splits"]["train"])
    assert {record.cve_id for record in selected} == train_ids
    assert not ({record.cve_id for record in selected} & set(manifest["splits"]["test"]))


def test_saved_test_predictions_are_held_out() -> None:
    manifest = json.loads(Path("data/splits/cve_splits.json").read_text(encoding="utf-8"))
    test_ids = set(manifest["splits"]["test"])
    records = {record.cve_id: record for record in load_records(Path("data/processed/nvd_records.json"))}
    with Path("artifacts/baseline_test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["cve_id"] for row in rows} == test_ids
    test_groups = {normalized_description(records[cve_id].description) for cve_id in test_ids}
    other_ids = set(records) - test_ids
    assert not test_groups & {
        normalized_description(records[cve_id].description) for cve_id in other_ids
    }


def test_translation_validation_rejects_placeholders_and_preserves_source() -> None:
    source = "CVE-2026-0001 affects Product 1.2."
    assert validate_translation(source, "") == "missing"
    assert not is_reliable_translation(source, "[Hindi Translation of CVE-2026-0001]: " + source)
    assert is_reliable_translation(source, "CVE-2026-0001 afecta Producto 1.2.")


def test_translation_validation_rejects_duplicate_and_out_of_split_rows(tmp_path: Path) -> None:
    records_path = tmp_path / "records.json"
    splits_path = tmp_path / "splits.json"
    translations_path = tmp_path / "translations.csv"
    records_path.write_text(
        json.dumps(
            [
                {
                    "cve_id": "CVE-2026-0001",
                    "description": "CVE-2026-0001 affects Product 1.2.",
                    "cvss_version": "3.1",
                    "cvss_base_score": 9.0,
                    "severity": "Critical",
                    "published": None,
                    "last_modified": None,
                    "cwes": [],
                },
                {
                    "cve_id": "CVE-2026-0002",
                    "description": "CVE-2026-0002 affects Product 2.0.",
                    "cvss_version": "3.1",
                    "cvss_base_score": 5.0,
                    "severity": "Medium",
                    "published": None,
                    "last_modified": None,
                    "cwes": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    splits_path.write_text(json.dumps({"splits": {"test": ["CVE-2026-0001"]}}), encoding="utf-8")
    translations_path.write_text(
        "cve_id,language,severity,original_text,translated_text,validation_status\n"
        "CVE-2026-0001,es,Critical,CVE-2026-0001 affects Product 1.2.,"
        "CVE-2026-0001 afecta Producto 1.2.,valid\n"
        "CVE-2026-0001,es,Critical,CVE-2026-0001 affects Product 1.2.,"
        "CVE-2026-0001 afecta Producto 1.2.,valid\n"
        "CVE-2026-0002,es,Medium,CVE-2026-0002 affects Product 2.0.,"
        "CVE-2026-0002 afecta Producto 2.0.,valid\n",
        encoding="utf-8",
    )
    result = validate_file(translations_path, records_path, splits_path)
    assert result["languages"]["es"]["valid_count"] == 1
    assert result["rejection_reasons"]["duplicate_cve_language"] == 1
    assert result["rejection_reasons"]["not_in_test_split"] == 1


def test_translation_generation_resume_skips_valid_rows() -> None:
    existing = {
        ("CVE-2026-0001", "es"): {"validation_status": "valid"},
        ("CVE-2026-0002", "es"): {"validation_status": "technical_identifier_missing"},
    }
    test_ids = ["CVE-2026-0001", "CVE-2026-0002", "CVE-2026-0003"]
    assert pending_cve_ids(existing, test_ids, "es") == ["CVE-2026-0003"]
    assert pending_cve_ids(existing, test_ids, "es", retry_failed=True) == [
        "CVE-2026-0002",
        "CVE-2026-0003",
    ]


def test_translation_identifier_report_does_not_allow_missing_cve_ids() -> None:
    source = "CVE-2026-0001 affects Product 1.2 in /srv/app.py."
    report = translation_identifier_report(
        source, "Afecta al Producto 1.2 en /srv/app.py."
    )
    assert "CVE-2026-0001" in report["missing"]
    assert "1.2" in report["preserved"]
    assert "/srv/app.py" in report["preserved"]
