import json
from pathlib import Path
from typing import Any

from scripts.prepare_dataset import prepare_dataset


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def nvd_item(cve_id: str, score: float, severity: str = "CRITICAL") -> dict[str, Any]:
    return {
        "cve": {
            "id": cve_id,
            "vulnStatus": "Analyzed",
            "descriptions": [{"lang": "en", "value": f"Description for {cve_id}"}],
            "published": "2026-01-01T00:00:00.000",
            "lastModified": "2026-01-02T00:00:00.000",
            "metrics": {
                "cvssMetricV31": [
                    {"cvssData": {"version": "3.1", "baseScore": score, "baseSeverity": severity}}
                ]
            },
            "weaknesses": [{"description": [{"lang": "en", "value": "CWE-79"}]}],
        }
    }


def test_prepare_dataset_writes_raw_csv_metadata_and_filters_duplicates(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "totalResults": 3,
                    "vulnerabilities": [
                        nvd_item("CVE-2026-0001", 9.8),
                        nvd_item("CVE-2026-0001", 9.8),
                        nvd_item("CVE-2026-0002", 5.0, "MEDIUM"),
                    ],
                }
            )
        ]
    )
    metadata = prepare_dataset(
        limit=2,
        output_path=tmp_path / "processed" / "cves.csv",
        raw_snapshot_path=tmp_path / "raw" / "snapshot.json",
        metadata_path=tmp_path / "processed" / "metadata.json",
        session=session,
        request_delay=0,
        sleeper=lambda _: None,
    )
    assert metadata["counts"]["records_retained"] == 2
    assert metadata["excluded_records"]["duplicate_cve_id"] == 1
    assert (tmp_path / "raw" / "snapshot.json").is_file()
    assert (tmp_path / "processed" / "cves.csv").read_text(encoding="utf-8").count("\n") == 3
    saved_metadata = json.loads((tmp_path / "processed" / "metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["class_distribution"] == {"Critical": 1, "Medium": 1}
    assert session.calls[0]["params"]["resultsPerPage"] == 200


def test_prepare_dataset_retries_rate_limit_and_sends_api_key(tmp_path: Path) -> None:
    rate_limited = FakeResponse({}, status_code=429)
    rate_limited.headers["Retry-After"] = "0"
    session = FakeSession(
        [
            rate_limited,
            FakeResponse({"totalResults": 1, "vulnerabilities": [nvd_item("CVE-2026-0003", 7.5, "HIGH")]}),
        ]
    )
    prepare_dataset(
        limit=1,
        output_path=tmp_path / "cves.csv",
        raw_snapshot_path=tmp_path / "raw.json",
        metadata_path=tmp_path / "metadata.json",
        session=session,
        api_key="test-key",
        request_delay=0,
        sleeper=lambda _: None,
    )
    assert len(session.calls) == 2
    assert session.calls[0]["headers"]["apiKey"] == "test-key"
