"""Reproducible NVD retrieval, filtering, and leakage-safe splitting utilities."""

from __future__ import annotations

import json
import os
import hashlib
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .severity import Severity, severity_from_cvss

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


@dataclass(frozen=True)
class VulnerabilityRecord:
    cve_id: str
    description: str
    cvss_version: str
    cvss_base_score: float
    severity: str
    published: str | None
    last_modified: str | None
    cwes: list[str]


def _english_description(descriptions: list[dict[str, Any]]) -> str | None:
    for description in descriptions:
        if description.get("lang") == "en" and description.get("value"):
            return str(description["value"]).strip()
    return None


def parse_nvd_vulnerability(item: dict[str, Any]) -> VulnerabilityRecord | None:
    cve = item.get("cve", {})
    cve_id = cve.get("id")
    description = _english_description(cve.get("descriptions", []))
    metrics = cve.get("metrics", {})
    v31 = (metrics.get("cvssMetricV31") or [None])[0]
    cvss = (v31 or {}).get("cvssData", {})
    score = cvss.get("baseScore")
    version = cvss.get("version")
    severity = severity_from_cvss(score)
    if not cve_id or not description or version != "3.1" or severity is None:
        return None
    weaknesses = cve.get("weaknesses", [])
    cwes = sorted(
        {
            value["value"]
            for weakness in weaknesses
            for value in weakness.get("description", [])
            if value.get("lang") == "en" and value.get("value", "").startswith("CWE-")
        }
    )
    return VulnerabilityRecord(
        cve_id=str(cve_id),
        description=description,
        cvss_version="3.1",
        cvss_base_score=float(score),
        severity=severity.value,
        published=cve.get("published"),
        last_modified=cve.get("lastModified"),
        cwes=cwes,
    )


def fetch_nvd(
    output_path: Path,
    target_count: int = 5000,
    page_size: int = 2000,
    request_delay_seconds: float = 0.7,
    session: requests.Session | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch eligible CVEs and save records plus retrieval provenance."""
    client = session or requests.Session()
    headers: dict[str, str] = {"User-Agent": "SentinelNLP-research/0.1"}
    key = api_key or os.getenv("NVD_API_KEY") or os.getenv("NVD_API_KEY_VALUE")
    if key:
        headers["apiKey"] = key
    records: dict[str, VulnerabilityRecord] = {}
    start_index = 0
    while len(records) < target_count:
        response = client.get(
            NVD_URL,
            params={"startIndex": start_index, "resultsPerPage": min(page_size, 2000)},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("vulnerabilities", []):
            record = parse_nvd_vulnerability(item)
            if record:
                records.setdefault(record.cve_id, record)
                if len(records) >= target_count:
                    break
        total = int(payload.get("totalResults", 0))
        start_index += len(payload.get("vulnerabilities", []))
        if start_index >= total or not payload.get("vulnerabilities"):
            break
        time.sleep(request_delay_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records.values()]
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    provenance = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": NVD_URL,
        "cvss_version": "3.1",
        "target_count": target_count,
        "retained_count": len(rows),
        "filter": "English description, CVSS v3.1 score 0.1-10.0, non-rejected records",
    }
    output_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance


def load_records(path: Path) -> list[VulnerabilityRecord]:
    return [VulnerabilityRecord(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def normalized_description(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _target_counts(total: int) -> dict[str, int]:
    train = round(total * 0.70)
    validation = round(total * 0.15)
    return {"train": train, "validation": validation, "test": total - train - validation}


def create_splits(
    records: list[VulnerabilityRecord], output_path: Path, seed: int = 42
) -> dict[str, list[str]]:
    """Create deterministic grouped splits, keeping identical descriptions together."""
    groups: dict[str, list[VulnerabilityRecord]] = defaultdict(list)
    for record in records:
        groups[normalized_description(record.description)].append(record)
    targets = _target_counts(len(records))
    label_totals = Counter(record.severity for record in records)
    label_targets = {
        split: {label: round(count * fraction) for label, count in label_totals.items()}
        for split, fraction in (("train", 0.70), ("validation", 0.15), ("test", 0.15))
    }
    label_targets["test"] = {
        label: label_totals[label] - label_targets["train"][label] - label_targets["validation"][label]
        for label in label_totals
    }
    assigned: dict[str, list[VulnerabilityRecord]] = {split: [] for split in targets}
    current_labels = {split: Counter() for split in targets}
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: (
            -len(group),
            hashlib.sha256(f"{seed}:{normalized_description(group[0].description)}".encode()).hexdigest(),
        ),
    )
    for group in ordered_groups:
        label_counts = Counter(record.severity for record in group)
        candidates = [split for split in targets if len(assigned[split]) + len(group) <= targets[split]]
        if not candidates:
            candidates = list(targets)
        split = min(
            candidates,
            key=lambda candidate: (
                sum(
                    abs(
                        current_labels[candidate][label]
                        + label_counts[label]
                        - label_targets[candidate][label]
                    )
                    for label in label_totals
                ),
                abs(targets[candidate] - len(assigned[candidate]) - len(group)),
                candidate,
            ),
        )
        assigned[split].extend(group)
        current_labels[split].update(label_counts)
    splits = {split: sorted(record.cve_id for record in group) for split, group in assigned.items()}
    manifest = {
        "version": 2,
        "strategy": "deterministic_normalized_description_grouped",
        "seed": seed,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "target_counts": targets,
        "actual_counts": {split: len(ids) for split, ids in splits.items()},
        "class_distribution": {
            split: dict(sorted(Counter(record.severity for record in assigned[split]).items()))
            for split in assigned
        },
        "duplicate_groups": sum(len(group) > 1 for group in groups.values()),
        "duplicate_records": sum(len(group) for group in groups.values() if len(group) > 1),
        "splits": splits,
    }
    dataset_payload = [
        asdict(record) for record in sorted(records, key=lambda record: record.cve_id)
    ]
    manifest["dataset_sha256"] = hashlib.sha256(
        json.dumps(dataset_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest["split_content_sha256"] = hashlib.sha256(
        json.dumps(splits, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return splits
