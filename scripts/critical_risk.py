import argparse
import json
from pathlib import Path

from backend.app.risk import critical_risk_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Critical underprediction metrics.")
    parser.add_argument("--input", type=Path, required=True, help="JSON with y_true and y_pred arrays")
    parser.add_argument("--output", type=Path, default=Path("artifacts/critical-risk.json"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = critical_risk_metrics(payload["y_true"], payload["y_pred"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
