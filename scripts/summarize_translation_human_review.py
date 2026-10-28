"""Validate and summarize manually entered translation-review worksheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RATING_FIELDS = (
    "semantic_accuracy",
    "completeness",
    "fluency_grammatical_quality",
)
ENUM_FIELDS = {
    "technical_identifier_preservation": {"pass", "fail"},
    "vulnerability_meaning_retained": {"yes", "no", "uncertain"},
    "severity_relevant_meaning_changed_or_omitted": {
        "yes",
        "no",
        "uncertain",
    },
}
SECONDARY_FIELDS = {
    "second_reviewer_code",
    "second_reviewer_qualified_target_language",
    "second_reviewer_qualification_evidence",
    "second_evidence_and_notes",
    *(f"second_{field}" for field in RATING_FIELDS),
    *(f"second_{field}" for field in ENUM_FIELDS),
}
ADJUDICATED_FIELDS = {
    "adjudicated_evidence_and_notes",
    *(f"adjudicated_{field}" for field in RATING_FIELDS),
    *(f"adjudicated_{field}" for field in ENUM_FIELDS),
}
REVIEW_INPUT_FIELDS = {
    "reviewer_code",
    "reviewer_qualified_target_language",
    "reviewer_qualification_evidence",
    "evidence_and_notes",
    "second_reviewer_code",
    "second_reviewer_qualified_target_language",
    "second_reviewer_qualification_evidence",
    "second_evidence_and_notes",
    "adjudication_status",
    "adjudicator_code",
    "adjudicator_qualified_target_language",
    "adjudication_evidence_and_notes",
    *RATING_FIELDS,
    *ENUM_FIELDS,
    *(f"second_{field}" for field in RATING_FIELDS),
    *(f"second_{field}" for field in ENUM_FIELDS),
    *(f"adjudicated_{field}" for field in RATING_FIELDS),
    *(f"adjudicated_{field}" for field in ENUM_FIELDS),
}
ADJUDICATION_STATES = {"", "not_required", "pending", "resolved"}


class ReviewValidationError(ValueError):
    """Raised when a review value is malformed or an incomplete review is rejected."""


def _cell(row: dict[str, str], name: str) -> str:
    return row.get(name, "").strip()


def _rating(row: dict[str, str], field: str) -> int | None:
    value = _cell(row, field)
    if not value:
        return None
    if value not in {"1", "2", "3", "4", "5"}:
        raise ReviewValidationError(
            f"{row.get('cve_id', '<unknown>')} {row.get('language', '<unknown>')}: "
            f"{field} must be an integer from 1 to 5, got {value!r}"
        )
    return int(value)


def _enum(row: dict[str, str], field: str) -> str:
    value = _cell(row, field).casefold()
    base_field = field
    for prefix in ("second_", "adjudicated_"):
        if base_field.startswith(prefix):
            base_field = base_field.removeprefix(prefix)
            break
    if value and value not in ENUM_FIELDS[base_field]:
        allowed = ", ".join(sorted(ENUM_FIELDS[base_field]))
        raise ReviewValidationError(
            f"{row.get('cve_id', '<unknown>')} {row.get('language', '<unknown>')}: "
            f"{field} must be one of {allowed}, got {value!r}"
        )
    return value


def _read_rubric(
    row: dict[str, str], prefix: str = ""
) -> dict[str, int | str | None]:
    values: dict[str, int | str | None] = {}
    for field in RATING_FIELDS:
        values[field] = _rating(row, f"{prefix}{field}")
    for field in ENUM_FIELDS:
        values[field] = _enum(row, f"{prefix}{field}")
    return values


def _rubric_complete(values: dict[str, int | str | None]) -> bool:
    return all(value is not None and value != "" for value in values.values())


def _needs_evidence(values: dict[str, int | str | None]) -> bool:
    return (
        any(
            isinstance(values[field], int) and values[field] <= 3
            for field in RATING_FIELDS
        )
        or values["technical_identifier_preservation"] == "fail"
        or values["vulnerability_meaning_retained"] != "yes"
        or values["severity_relevant_meaning_changed_or_omitted"] != "no"
    )


def _qualified(row: dict[str, str], prefix: str) -> bool:
    return (
        _cell(row, f"{prefix}qualified_target_language").casefold() == "yes"
        and bool(_cell(row, f"{prefix}qualification_evidence"))
    )


def _rating_difference(
    first: dict[str, int | str | None],
    second: dict[str, int | str | None],
) -> list[str]:
    return [field for field in first if first[field] != second[field]]


def _classify_review(row: dict[str, str]) -> dict[str, Any]:
    first = _read_rubric(row)
    second = _read_rubric(row, "second_")
    adjudicated = _read_rubric(row, "adjudicated_")
    has_review_input = any(_cell(row, field) for field in REVIEW_INPUT_FIELDS)
    if not has_review_input:
        return {
            "status": "unreviewed",
            "issues": [],
            "ratings": None,
            "reviewer_codes": [],
            "identifier_failure": None,
            "severity_meaning_changed": None,
        }

    issues: list[str] = []
    first_code = _cell(row, "reviewer_code")
    if not first_code:
        issues.append("missing_reviewer_identity")
    if not _rubric_complete(first):
        issues.append("primary_rubric_incomplete")
    if not _qualified(row, "reviewer_"):
        issues.append("qualified_target_language_reviewer_not_documented")
    if _needs_evidence(first) and not _cell(row, "evidence_and_notes"):
        issues.append("evidence_required_for_flagged_primary_ratings")

    has_second_input = any(_cell(row, field) for field in SECONDARY_FIELDS)
    if has_second_input:
        if not _cell(row, "second_reviewer_code"):
            issues.append("missing_second_reviewer_identity")
        if not _rubric_complete(second):
            issues.append("second_rubric_incomplete")
        if not _qualified(row, "second_reviewer_"):
            issues.append("qualified_second_reviewer_not_documented")
        if _needs_evidence(second) and not _cell(row, "second_evidence_and_notes"):
            issues.append("evidence_required_for_flagged_second_ratings")

    adjudication_status = _cell(row, "adjudication_status").casefold()
    if adjudication_status not in ADJUDICATION_STATES:
        raise ReviewValidationError(
            f"{row.get('cve_id', '<unknown>')} {row.get('language', '<unknown>')}: "
            "adjudication_status must be blank, not_required, pending, or resolved"
        )
    disagreement = (
        has_second_input
        and _rubric_complete(first)
        and _rubric_complete(second)
        and bool(_rating_difference(first, second))
    )
    uncertain = any(
        values.get(field) == "uncertain"
        for values in (first, second)
        for field in (
            "vulnerability_meaning_retained",
            "severity_relevant_meaning_changed_or_omitted",
        )
    )
    needs_adjudication = disagreement or uncertain
    if needs_adjudication and adjudication_status != "resolved":
        issues.append("uncertainty_or_disagreement_requires_adjudication")
    if adjudication_status == "pending":
        issues.append("adjudication_pending")
    if adjudication_status == "resolved":
        if not _cell(row, "adjudicator_code"):
            issues.append("missing_adjudicator_identity")
        if _cell(row, "adjudicator_qualified_target_language").casefold() != "yes":
            issues.append("qualified_adjudicator_not_documented")
        if not _rubric_complete(adjudicated):
            issues.append("adjudicated_rubric_incomplete")
        if not _cell(row, "adjudication_evidence_and_notes"):
            issues.append("adjudication_evidence_required")
    elif any(_cell(row, field) for field in ADJUDICATED_FIELDS) or _cell(
        row, "adjudicator_code"
    ):
        issues.append("adjudication_values_present_without_resolved_status")

    if issues:
        return {
            "status": "incomplete",
            "issues": sorted(set(issues)),
            "ratings": None,
            "reviewer_codes": [
                code
                for code in (first_code, _cell(row, "second_reviewer_code"))
                if code
            ],
            "identifier_failure": None,
            "severity_meaning_changed": None,
        }

    final_values = adjudicated if adjudication_status == "resolved" else first
    if _needs_evidence(final_values):
        evidence_field = (
            "adjudication_evidence_and_notes"
            if adjudication_status == "resolved"
            else "evidence_and_notes"
        )
        if not _cell(row, evidence_field):
            return {
                "status": "incomplete",
                "issues": ["evidence_required_for_final_decision"],
                "ratings": None,
                "reviewer_codes": [first_code],
                "identifier_failure": None,
                "severity_meaning_changed": None,
            }

    scores = [int(final_values[field]) for field in RATING_FIELDS]
    identifier_failure = (
        final_values["technical_identifier_preservation"] == "fail"
    )
    meaning = final_values["vulnerability_meaning_retained"]
    severity_change = final_values[
        "severity_relevant_meaning_changed_or_omitted"
    ]
    if (
        min(scores) <= 2
        or identifier_failure
        or meaning == "no"
        or severity_change == "yes"
    ):
        status = "rejected"
    elif (
        min(scores) < 4
        or meaning != "yes"
        or severity_change != "no"
    ):
        status = "incomplete"
        issues = ["rating_or_uncertainty_does_not_meet_acceptance_threshold"]
    else:
        status = "accepted"
    return {
        "status": status,
        "issues": issues,
        "ratings": final_values,
        "reviewer_codes": [
            code
            for code in (
                first_code,
                _cell(row, "second_reviewer_code"),
                _cell(row, "adjudicator_code"),
            )
            if code
        ],
        "identifier_failure": identifier_failure,
        "severity_meaning_changed": severity_change,
    }


def summarize_reviews(
    rows: list[dict[str, str]],
    *,
    protocol_version: str = "1.0",
) -> dict[str, Any]:
    if not rows:
        raise ReviewValidationError("Review worksheet has no rows")
    required = {
        "cve_id",
        "language",
        "severity",
        "original_english",
        "machine_translation",
        "exact_identifier_integrity_result",
        *REVIEW_INPUT_FIELDS,
        "review_status",
    }
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        raise ReviewValidationError(
            "Worksheet is missing required columns: " + ", ".join(missing_columns)
        )
    pairs = [(row["cve_id"].strip(), row["language"].strip().lower()) for row in rows]
    if any(not cve_id or not language for cve_id, language in pairs):
        raise ReviewValidationError("Every row must have a CVE ID and language")
    if len(pairs) != len(set(pairs)):
        raise ReviewValidationError("Worksheet contains duplicate CVE-language pairs")

    assessments: list[dict[str, Any]] = []
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    by_severity: dict[str, Counter[str]] = defaultdict(Counter)
    rating_distributions: dict[str, Counter[int]] = {
        field: Counter() for field in RATING_FIELDS
    }
    identifier_outcomes: Counter[str] = Counter()
    severity_meaning_outcomes: Counter[str] = Counter()
    entered_rating_distributions: dict[str, dict[str, Counter[int]]] = {
        role: {field: Counter() for field in RATING_FIELDS}
        for role in ("primary", "second", "adjudicated")
    }
    entered_identifier_outcomes: dict[str, Counter[str]] = {
        role: Counter() for role in ("primary", "second", "adjudicated")
    }
    entered_severity_outcomes: dict[str, Counter[str]] = {
        role: Counter() for role in ("primary", "second", "adjudicated")
    }
    pairs_with_identifier_failure: set[tuple[str, str]] = set()
    pairs_with_severity_meaning_change: set[tuple[str, str]] = set()
    qualified_reviewers_by_language: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        assessment = _classify_review(row)
        status = assessment["status"]
        language = row["language"].strip().lower()
        severity = row["severity"].strip() or "unspecified"
        pair = (row["cve_id"].strip(), language)
        by_language[language][status] += 1
        by_severity[severity][status] += 1
        for identity_field, qualification_field in (
            ("reviewer_code", "reviewer_qualified_target_language"),
            (
                "second_reviewer_code",
                "second_reviewer_qualified_target_language",
            ),
            (
                "adjudicator_code",
                "adjudicator_qualified_target_language",
            ),
        ):
            if (
                _cell(row, identity_field)
                and _cell(row, qualification_field).casefold() == "yes"
            ):
                qualified_reviewers_by_language[language].add(
                    _cell(row, identity_field)
                )
        for role, prefix in (
            ("primary", ""),
            ("second", "second_"),
            ("adjudicated", "adjudicated_"),
        ):
            role_values = _read_rubric(row, prefix)
            for field in RATING_FIELDS:
                value = role_values[field]
                if isinstance(value, int):
                    entered_rating_distributions[role][field][value] += 1
            identifier_value = role_values["technical_identifier_preservation"]
            if identifier_value:
                entered_identifier_outcomes[role][str(identifier_value)] += 1
                if identifier_value == "fail":
                    pairs_with_identifier_failure.add(pair)
            severity_value = role_values[
                "severity_relevant_meaning_changed_or_omitted"
            ]
            if severity_value:
                entered_severity_outcomes[role][str(severity_value)] += 1
                if severity_value in {"yes", "uncertain"}:
                    pairs_with_severity_meaning_change.add(pair)
        for field, value in (assessment["ratings"] or {}).items():
            if field in rating_distributions and isinstance(value, int):
                rating_distributions[field][value] += 1
        if assessment["identifier_failure"] is not None:
            identifier_outcomes[
                "fail" if assessment["identifier_failure"] else "pass"
            ] += 1
        if assessment["severity_meaning_changed"] is not None:
            severity_meaning_outcomes[
                str(assessment["severity_meaning_changed"])
            ] += 1
        assessments.append(
            {
                "cve_id": row["cve_id"],
                "language": language,
                "severity": severity,
                "review_status": status,
                "reviewer_codes": assessment["reviewer_codes"],
                "issues": assessment["issues"],
                "final_ratings": assessment["ratings"],
            }
        )

    status_counts = Counter(row["review_status"] for row in assessments)
    return {
        "schema_version": "1.0",
        "protocol_version": protocol_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "Entered worksheet review data only; no automated translation checks or back-translations are counted as human review.",
        "reviewed_pair_count": len(rows) - status_counts.get("unreviewed", 0),
        "completed_review_pair_count": sum(
            count
            for status, count in status_counts.items()
            if status in {"accepted", "rejected"}
        ),
        "pair_count": len(rows),
        "status_counts": {
            status: status_counts.get(status, 0)
            for status in ("unreviewed", "incomplete", "accepted", "rejected")
        },
        "counts_by_language": {
            language: {
                status: counts.get(status, 0)
                for status in ("unreviewed", "incomplete", "accepted", "rejected")
            }
            for language, counts in sorted(by_language.items())
        },
        "review_coverage_by_language": {
            language: {
                "pair_count": sum(counts.values()),
                "accepted": counts.get("accepted", 0),
                "rejected": counts.get("rejected", 0),
                "incomplete": counts.get("incomplete", 0),
                "unreviewed": counts.get("unreviewed", 0),
                "qualified_reviewer_count": len(
                    qualified_reviewers_by_language.get(language, set())
                ),
            }
            for language, counts in sorted(by_language.items())
        },
        "awaiting_qualified_review_languages": sorted(
            language
            for language in by_language
            if not qualified_reviewers_by_language.get(language)
        ),
        "counts_by_severity": {
            severity: {
                status: counts.get(status, 0)
                for status in ("unreviewed", "incomplete", "accepted", "rejected")
            }
            for severity, counts in sorted(by_severity.items())
        },
        "rating_distributions": {
            field: {
                str(score): count
                for score, count in sorted(distribution.items())
            }
            for field, distribution in rating_distributions.items()
        },
        "entered_rating_distributions_by_reviewer": {
            role: {
                field: {
                    str(score): count
                    for score, count in sorted(distribution.items())
                }
                for field, distribution in fields.items()
            }
            for role, fields in entered_rating_distributions.items()
        },
        "identifier_review_outcomes": {
            outcome: identifier_outcomes.get(outcome, 0)
            for outcome in ("pass", "fail")
        },
        "entered_identifier_review_outcomes_by_reviewer": {
            role: {
                outcome: counts.get(outcome, 0)
                for outcome in ("pass", "fail")
            }
            for role, counts in entered_identifier_outcomes.items()
        },
        "pairs_with_any_entered_identifier_failure": len(
            pairs_with_identifier_failure
        ),
        "severity_relevant_meaning_change_outcomes": {
            outcome: severity_meaning_outcomes.get(outcome, 0)
            for outcome in ("no", "yes", "uncertain")
        },
        "entered_severity_relevant_change_outcomes_by_reviewer": {
            role: {
                outcome: counts.get(outcome, 0)
                for outcome in ("no", "yes", "uncertain")
            }
            for role, counts in entered_severity_outcomes.items()
        },
        "pairs_with_any_entered_severity_change_or_uncertainty": len(
            pairs_with_severity_meaning_change
        ),
        "all_rows_accepted": status_counts.get("accepted", 0) == len(rows),
        "scale_up_gate": "NO-GO",
        "scale_up_gate_reason": (
            "This targeted failure-recovery set is not a representative sample of "
            "all planned translations. Scale-up also requires qualified review "
            "across all five languages, no unresolved critical errors, and "
            "adjudication of disagreements and uncertain cases."
        ),
        "rows": assessments,
    }


def read_worksheet(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_expected_pairs(path: Path) -> set[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"cve_id", "language"}.issubset(reader.fieldnames or set()):
            raise ReviewValidationError(
                "Target manifest must contain cve_id and language columns"
            )
        pairs = {
            (row["cve_id"].strip(), row["language"].strip().lower())
            for row in reader
        }
    if len(pairs) == 0:
        raise ReviewValidationError("Target manifest has no pairs")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worksheet",
        type=Path,
        default=Path("artifacts/translation_human_review_worksheet_gpu_v5.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/translation_human_review_summary_v1.json"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("artifacts/translation_pilot_retry_targets_gpu_v2.csv"),
        help="Expected target-pair manifest for the review worksheet.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a progress summary with incomplete rows; they remain non-passing.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=22,
        help="Expected number of targeted rows; use 0 to disable the count check.",
    )
    args = parser.parse_args()
    try:
        rows = read_worksheet(args.worksheet)
        if args.expected_count and len(rows) != args.expected_count:
            raise ReviewValidationError(
                f"Expected {args.expected_count} review rows, found {len(rows)}"
            )
        worksheet_pairs = {
            (row["cve_id"].strip(), row["language"].strip().lower())
            for row in rows
        }
        expected_pairs = read_expected_pairs(args.targets)
        if worksheet_pairs != expected_pairs:
            raise ReviewValidationError(
                "Worksheet pairs do not exactly match the expected target manifest"
            )
        summary = summarize_reviews(rows)
    except (OSError, csv.Error, ReviewValidationError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                **summary,
                "worksheet": str(args.worksheet),
                "worksheet_sha256": hashlib.sha256(
                    args.worksheet.read_bytes()
                ).hexdigest(),
                "target_manifest": str(args.targets),
                "target_manifest_sha256": hashlib.sha256(
                    args.targets.read_bytes()
                ).hexdigest(),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    if summary["status_counts"]["incomplete"] and not args.allow_incomplete:
        print(
            "Summary written, but incomplete review rows are rejected by default. "
            "Use --allow-incomplete only for a clearly labeled progress report.",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pair_count": summary["pair_count"],
                "status_counts": summary["status_counts"],
                "scale_up_gate": summary["scale_up_gate"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
