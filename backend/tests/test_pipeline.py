import json
from pathlib import Path

import pytest

from backend.app.data_pipeline import (
    VulnerabilityRecord,
    create_splits,
    normalized_description,
    parse_nvd_vulnerability,
)
from backend.app.faithfulness import area_under_curve, perturbation_curve
from backend.app.metrics import agreement_metrics
from backend.app.risk import critical_risk_metrics


def test_parse_nvd_keeps_only_cvss_31() -> None:
    item = {
        "cve": {
            "id": "CVE-2026-0001",
            "descriptions": [{"lang": "en", "value": "A vulnerability in a parser."}],
            "metrics": {"cvssMetricV31": [{"cvssData": {"version": "3.1", "baseScore": 9.8}}]},
        }
    }
    record = parse_nvd_vulnerability(item)
    assert record is not None
    assert record.severity == "Critical"


def test_splits_do_not_overlap(tmp_path: Path) -> None:
    records = [
        VulnerabilityRecord(f"CVE-2026-{index:04d}", f"text {index}", "3.1", 9.8, "Critical", None, None, [])
        for index in range(20)
    ]
    splits = create_splits(records, tmp_path / "splits.json")
    assert not set(splits["train"]) & set(splits["validation"]) & set(splits["test"])
    assert set().union(*map(set, splits.values())) == {record.cve_id for record in records}


def test_grouped_splits_keep_normalized_duplicates_together(tmp_path: Path) -> None:
    records = [
        VulnerabilityRecord("CVE-2026-0001", "Same  description", "3.1", 9.8, "Critical", None, None, []),
        VulnerabilityRecord("CVE-2026-0002", "same description", "3.1", 9.8, "Critical", None, None, []),
    ] + [
        VulnerabilityRecord(f"CVE-2026-{index:04d}", f"unique {index}", "3.1", 5.0, "Medium", None, None, [])
        for index in range(3, 30)
    ]
    splits = create_splits(records, tmp_path / "splits.json")
    locations = {
        split: split_name
        for split_name, ids in splits.items()
        for split in ids
    }
    assert locations["CVE-2026-0001"] == locations["CVE-2026-0002"]
    assert normalized_description(" Same   description ") == normalized_description("same description")


def test_metrics_distinguish_agreement_and_critical_risk() -> None:
    assert agreement_metrics(["Critical", "Low"], ["High", "Low"])["agreement_with_english"] == 0.5
    result = critical_risk_metrics(["Critical", "Critical", "Low"], ["High", "Critical", "Low"])
    assert result["critical_underprediction_count"] == 1
    assert result["critical_underprediction_rate"] == 0.5


def test_faithfulness_curve_and_area() -> None:
    curve = perturbation_curve(["critical", "remote"], [0, 1], lambda text: len(text.split()) / 2)
    assert curve[0]["probability"] == 1.0
    assert curve[-1]["probability"] == 0.0
    assert area_under_curve(curve) == 0.5


def test_empty_critical_denominator_is_undefined() -> None:
    assert critical_risk_metrics(["Low"], ["Low"])["critical_underprediction_rate"] is None


def test_critical_risk_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        critical_risk_metrics(["Critical", "Low"], ["Critical"])
