from __future__ import annotations

import csv
from copy import deepcopy
import json
import sys

import pytest

from scripts.summarize_translation_human_review import (
    ReviewValidationError,
    main,
    summarize_reviews,
)


def _row(
    cve_id: str = "CVE-2024-0001",
    language: str = "es",
    severity: str = "High",
) -> dict[str, str]:
    row = {
        "cve_id": cve_id,
        "language": language,
        "severity": severity,
        "original_english": "Remote unauthenticated denial of service.",
        "machine_translation": "Traducción de ejemplo.",
        "exact_identifier_integrity_result": "pass",
        "reviewer_code": "",
        "reviewer_qualified_target_language": "",
        "reviewer_qualification_evidence": "",
        "semantic_accuracy": "",
        "completeness": "",
        "fluency_grammatical_quality": "",
        "technical_identifier_preservation": "",
        "vulnerability_meaning_retained": "",
        "severity_relevant_meaning_changed_or_omitted": "",
        "evidence_and_notes": "",
        "second_reviewer_code": "",
        "second_reviewer_qualified_target_language": "",
        "second_reviewer_qualification_evidence": "",
        "second_semantic_accuracy": "",
        "second_completeness": "",
        "second_fluency_grammatical_quality": "",
        "second_technical_identifier_preservation": "",
        "second_vulnerability_meaning_retained": "",
        "second_severity_relevant_meaning_changed_or_omitted": "",
        "second_evidence_and_notes": "",
        "adjudication_status": "",
        "adjudicator_code": "",
        "adjudicator_qualified_target_language": "",
        "adjudicated_semantic_accuracy": "",
        "adjudicated_completeness": "",
        "adjudicated_fluency_grammatical_quality": "",
        "adjudicated_technical_identifier_preservation": "",
        "adjudicated_vulnerability_meaning_retained": "",
        "adjudicated_severity_relevant_meaning_changed_or_omitted": "",
        "adjudication_evidence_and_notes": "",
        "review_status": "unreviewed",
    }
    return row


def _complete_review(row: dict[str, str]) -> dict[str, str]:
    row.update(
        {
            "reviewer_code": "R-ES-01",
            "reviewer_qualified_target_language": "yes",
            "reviewer_qualification_evidence": "Qualified Spanish-language reviewer",
            "semantic_accuracy": "5",
            "completeness": "5",
            "fluency_grammatical_quality": "4",
            "technical_identifier_preservation": "pass",
            "vulnerability_meaning_retained": "yes",
            "severity_relevant_meaning_changed_or_omitted": "no",
        }
    )
    return row


def test_unreviewed_and_partial_rows_are_never_counted_as_passing() -> None:
    unreviewed = _row()
    incomplete = _row("CVE-2024-0002", "fr", "Critical")
    incomplete.update(
        {
            "reviewer_code": "R-FR-01",
            "reviewer_qualified_target_language": "yes",
            "reviewer_qualification_evidence": "Qualified French-language reviewer",
            "semantic_accuracy": "5",
        }
    )

    summary = summarize_reviews([unreviewed, incomplete])

    assert summary["status_counts"] == {
        "unreviewed": 1,
        "incomplete": 1,
        "accepted": 0,
        "rejected": 0,
    }
    assert summary["reviewed_pair_count"] == 1
    assert summary["completed_review_pair_count"] == 0
    assert summary["all_rows_accepted"] is False


def test_invalid_rating_values_are_rejected() -> None:
    row = _row()
    row["semantic_accuracy"] = "6"

    with pytest.raises(ReviewValidationError, match="integer from 1 to 5"):
        summarize_reviews([row])


def test_missing_reviewer_identity_makes_entered_ratings_incomplete() -> None:
    row = _complete_review(_row())
    row["reviewer_code"] = ""

    summary = summarize_reviews([row])

    assert summary["status_counts"]["incomplete"] == 1
    assert summary["status_counts"]["accepted"] == 0
    assert "missing_reviewer_identity" in summary["rows"][0]["issues"]


def test_correct_aggregation_separates_accepted_rejected_and_unreviewed() -> None:
    accepted = _complete_review(_row("CVE-2024-0001", "es", "High"))
    rejected = _complete_review(_row("CVE-2024-0002", "fr", "Critical"))
    rejected.update(
        {
            "semantic_accuracy": "2",
            "vulnerability_meaning_retained": "no",
            "severity_relevant_meaning_changed_or_omitted": "yes",
            "evidence_and_notes": (
                "Source says remote; translation says local, changing attack conditions."
            ),
        }
    )
    unreviewed = _row("CVE-2024-0003", "zh", "Medium")

    summary = summarize_reviews([accepted, rejected, unreviewed])

    assert summary["status_counts"] == {
        "unreviewed": 1,
        "incomplete": 0,
        "accepted": 1,
        "rejected": 1,
    }
    assert summary["counts_by_language"]["es"]["accepted"] == 1
    assert summary["counts_by_severity"]["Critical"]["rejected"] == 1
    assert summary["rating_distributions"]["semantic_accuracy"] == {
        "2": 1,
        "5": 1,
    }
    assert summary["identifier_review_outcomes"]["pass"] == 2
    assert summary["severity_relevant_meaning_change_outcomes"]["yes"] == 1
    assert summary["scale_up_gate"] == "NO-GO"


def test_uncertain_review_requires_qualified_adjudication() -> None:
    row = _complete_review(_row())
    row.update(
        {
            "vulnerability_meaning_retained": "uncertain",
            "evidence_and_notes": "The attack precondition is ambiguous.",
        }
    )

    summary = summarize_reviews([row])

    assert summary["status_counts"]["incomplete"] == 1
    assert "uncertainty_or_disagreement_requires_adjudication" in (
        summary["rows"][0]["issues"]
    )

    adjudicated = deepcopy(row)
    adjudicated.update(
        {
            "adjudication_status": "resolved",
            "adjudicator_code": "A-ES-01",
            "adjudicator_qualified_target_language": "yes",
            "adjudicated_semantic_accuracy": "4",
            "adjudicated_completeness": "4",
            "adjudicated_fluency_grammatical_quality": "4",
            "adjudicated_technical_identifier_preservation": "pass",
            "adjudicated_vulnerability_meaning_retained": "yes",
            "adjudicated_severity_relevant_meaning_changed_or_omitted": "no",
            "adjudication_evidence_and_notes": "Compared source and target phrasing.",
        }
    )
    resolved = summarize_reviews([adjudicated])
    assert resolved["status_counts"]["accepted"] == 1


def test_cli_rejects_incomplete_reviews_unless_progress_is_requested(
    tmp_path, monkeypatch
) -> None:
    row = _row()
    row.update(
        {
            "reviewer_code": "R-ES-01",
            "reviewer_qualified_target_language": "yes",
            "reviewer_qualification_evidence": "Qualified Spanish-language reviewer",
            "semantic_accuracy": "5",
        }
    )
    worksheet = tmp_path / "worksheet.csv"
    targets = tmp_path / "targets.csv"
    output = tmp_path / "summary.json"
    with worksheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with targets.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cve_id", "language"])
        writer.writeheader()
        writer.writerow({"cve_id": row["cve_id"], "language": row["language"]})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_translation_human_review.py",
            "--worksheet",
            str(worksheet),
            "--targets",
            str(targets),
            "--expected-count",
            "1",
            "--output",
            str(output),
        ],
    )
    assert main() == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status_counts"][
        "incomplete"
    ] == 1

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--allow-incomplete"])
    assert main() == 0
