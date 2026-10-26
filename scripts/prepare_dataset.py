"""Acquire and filter a reproducible NVD CVE dataset.

This script performs network access only when invoked explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

# Permit the documented ``python scripts\prepare_dataset.py`` invocation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data_pipeline import (
    NVD_URL,
    VulnerabilityRecord,
    parse_nvd_vulnerability,
)

# Keep the default response bounded: NVD descriptions and metric histories can
# make a 2,000-record page consume substantial memory before filtering.
DEFAULT_PAGE_SIZE = 200
MAX_PAGE_SIZE = 2000
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RETRIES = 4
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class NVDRequestError(RuntimeError):
    """Raised when an NVD request cannot be completed after retries."""


def _normalise_date(value: str | None, option: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{option} must be an ISO-8601 date or datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _retry_delay(response: requests.Response | None, attempt: int, backoff: float) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return backoff * (2**attempt)


def _get_page(
    session: requests.Session,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    retries: int,
    backoff: float,
    sleeper: Callable[[float], None],
) -> dict[str, Any]:
    for attempt in range(retries + 1):
        response: requests.Response | None = None
        try:
            response = session.get(NVD_URL, params=params, headers=headers, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < retries:
                sleeper(_retry_delay(response, attempt, backoff))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise NVDRequestError("NVD returned a non-object JSON response")
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt >= retries:
                raise NVDRequestError(
                    f"NVD request failed after {retries + 1} attempts: {exc}"
                ) from exc
            sleeper(_retry_delay(response, attempt, backoff))
    raise NVDRequestError("NVD request failed without a response")


def _classify_item(item: dict[str, Any]) -> tuple[VulnerabilityRecord | None, str | None]:
    cve = item.get("cve") or {}
    if str(cve.get("vulnStatus", "")).strip().lower() == "rejected":
        return None, "rejected"
    if not cve.get("id"):
        return None, "missing_cve_id"
    if not cve.get("descriptions"):
        return None, "missing_description"
    metrics = cve.get("metrics") or {}
    v31 = (metrics.get("cvssMetricV31") or [None])[0]
    if not v31:
        return None, "missing_cvss_v3_1"
    cvss_data = v31.get("cvssData") or {}
    if cvss_data.get("version") != "3.1":
        return None, "non_cvss_v3_1"
    if cvss_data.get("baseScore") is None:
        return None, "missing_base_score"
    record = parse_nvd_vulnerability(item)
    if record is None:
        return None, "invalid_score_or_description"
    reported_label = str(cvss_data.get("baseSeverity", "")).strip().lower()
    if reported_label and reported_label != record.severity.lower():
        return None, "score_label_mismatch"
    return record, None


def _write_csv(path: Path, records: list[VulnerabilityRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cve_id",
        "english_description",
        "cvss_version",
        "cvss_base_score",
        "severity",
        "published",
        "last_modified",
        "cwe_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "cve_id": record.cve_id,
                    "english_description": record.description,
                    "cvss_version": record.cvss_version,
                    "cvss_base_score": record.cvss_base_score,
                    "severity": record.severity,
                    "published": record.published or "",
                    "last_modified": record.last_modified or "",
                    "cwe_ids": json.dumps(record.cwes, ensure_ascii=False),
                }
            )


def prepare_dataset(
    *,
    limit: int,
    output_path: Path,
    raw_snapshot_path: Path,
    metadata_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    backoff: float = 1.0,
    request_delay: float = 0.7,
    api_key: str | None = None,
    session: requests.Session | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
    if retries < 0:
        raise ValueError("retries cannot be negative")
    normalised_start = _normalise_date(start_date, "--start-date")
    normalised_end = _normalise_date(end_date, "--end-date")
    if normalised_start and normalised_end and normalised_start > normalised_end:
        raise ValueError("--start-date must not be later than --end-date")

    client = session or requests.Session()
    headers = {"User-Agent": "SentinelNLP-research/0.1"}
    if api_key:
        headers["apiKey"] = api_key
    records: dict[str, VulnerabilityRecord] = {}
    excluded: Counter[str] = Counter()
    raw_items: list[dict[str, Any]] = []
    requested_items = 0
    duplicate_count = 0
    start_index = 0
    total_results: int | None = None

    while len(records) < limit:
        params: dict[str, Any] = {
            "startIndex": start_index,
            "resultsPerPage": page_size,
        }
        if normalised_start:
            params["pubStartDate"] = normalised_start
        if normalised_end:
            params["pubEndDate"] = normalised_end
        payload = _get_page(client, params, headers, timeout, retries, backoff, sleeper)
        vulnerabilities = payload.get("vulnerabilities") or []
        if not isinstance(vulnerabilities, list):
            raise NVDRequestError("NVD response has an invalid vulnerabilities field")
        raw_items.extend(vulnerabilities)
        requested_items += len(vulnerabilities)
        total_results = int(payload.get("totalResults", requested_items))
        for item in vulnerabilities:
            record, reason = _classify_item(item)
            if reason:
                excluded[reason] += 1
                continue
            assert record is not None
            if record.cve_id in records:
                duplicate_count += 1
                excluded["duplicate_cve_id"] += 1
                continue
            records[record.cve_id] = record
        start_index += len(vulnerabilities)
        if not vulnerabilities or start_index >= total_results:
            break
        if request_delay:
            sleeper(request_delay)

    raw_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    raw_snapshot_path.write_text(
        json.dumps(
            {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "source": NVD_URL,
                "query": {
                    "start_date": normalised_start,
                    "end_date": normalised_end,
                    "page_size": page_size,
                },
                "total_results_reported": total_results,
                "vulnerabilities": raw_items,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    retained = list(records.values())
    _write_csv(output_path, retained)
    class_distribution = dict(sorted(Counter(record.severity for record in retained).items()))
    metadata = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": NVD_URL,
        "filters": {
            "cvss_version": "3.1",
            "valid_score_range": "0.1-10.0",
            "english_description_required": True,
            "rejected_records_excluded": True,
            "start_date": normalised_start,
            "end_date": normalised_end,
            "requested_limit": limit,
        },
        "counts": {
            "nvd_items_retrieved": requested_items,
            "records_retained": len(retained),
            "records_excluded": sum(excluded.values()),
            "duplicate_cve_ids": duplicate_count,
        },
        "excluded_records": dict(sorted(excluded.items())),
        "class_distribution": class_distribution,
        "raw_snapshot_path": str(raw_snapshot_path),
        "processed_dataset_path": str(output_path),
        "api_key_used": bool(api_key),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve and prepare CVSS v3.1 NVD records.")
    parser.add_argument("--limit", type=int, default=100, help="Number of valid records to retain")
    parser.add_argument("--start-date", help="NVD publication start date, ISO-8601")
    parser.add_argument("--end-date", help="NVD publication end date, ISO-8601")
    parser.add_argument("--output", type=Path, default=Path("data/processed/cves.csv"))
    parser.add_argument("--raw-output", type=Path, default=Path("data/raw/nvd_cves_raw.json"))
    parser.add_argument("--metadata-output", type=Path, default=Path("data/processed/dataset_metadata.json"))
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--request-delay", type=float, default=0.7)
    args = parser.parse_args()
    metadata = prepare_dataset(
        limit=args.limit,
        output_path=args.output,
        raw_snapshot_path=args.raw_output,
        metadata_path=args.metadata_output,
        start_date=args.start_date,
        end_date=args.end_date,
        page_size=args.page_size,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        request_delay=args.request_delay,
        api_key=os.getenv("NVD_API_KEY") or os.getenv("NVD_API_KEY_VALUE"),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
