#!/usr/bin/env python
"""Summarize local HR retrieval results into report-ready tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "local_hr"
METRICS = ("ndcg@5", "recall@5", "mrr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze local HR benchmark JSON outputs.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def load_json_results(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    loaded = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "metrics" in payload:
            loaded.append((path, payload))
    return loaded


def result_key(payload: dict[str, Any]) -> str | None:
    method = payload.get("method")
    mode = payload.get("mode")

    if method == "singleview" and mode == "clean":
        return "singleview_clean"
    if method == "singleview" and mode == "degraded":
        return "singleview_degraded"
    if method == "multiview" and mode == "degraded":
        return "multiview_degraded"
    if method == "invariant_calibration":
        return "invariant_calibration_degraded"
    return None


def pick_latest_by_key(results: list[tuple[Path, dict[str, Any]]]) -> dict[str, tuple[Path, dict[str, Any]]]:
    picked: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, payload in results:
        key = result_key(payload)
        if key is None:
            continue
        if key not in picked or path.stat().st_mtime > picked[key][0].stat().st_mtime:
            picked[key] = (path, payload)
    return picked


def safe_recovery(method_value: float, clean_value: float, degraded_value: float) -> float | None:
    degradation_loss = clean_value - degraded_value
    if abs(degradation_loss) < 1e-12:
        return None
    return (method_value - degraded_value) / degradation_loss


def build_summary(picked: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    clean_metrics = picked.get("singleview_clean", ({}, {"metrics": {}}))[1]["metrics"]
    degraded_metrics = picked.get("singleview_degraded", ({}, {"metrics": {}}))[1]["metrics"]

    for key, (path, payload) in sorted(picked.items()):
        metrics = payload["metrics"]
        row: dict[str, Any] = {
            "key": key,
            "source": str(path),
            "method": payload.get("method"),
            "mode": payload.get("mode", "degraded"),
            "variant": payload.get("variant") or payload.get("eval_variant"),
            "use_multiview": payload.get("use_multiview", False),
        }
        for metric in METRICS:
            value = metrics.get(metric)
            row[metric] = value
            if metric in clean_metrics and metric in degraded_metrics and value is not None:
                row[f"{metric}_degradation_loss"] = clean_metrics[metric] - degraded_metrics[metric]
                row[f"{metric}_robust_gain"] = value - degraded_metrics[metric]
                row[f"{metric}_recovery"] = safe_recovery(value, clean_metrics[metric], degraded_metrics[metric])
        rows.append(row)

    return {
        "baseline_keys": {
            "clean": "singleview_clean" if "singleview_clean" in picked else None,
            "degraded": "singleview_degraded" if "singleview_degraded" in picked else None,
        },
        "rows": rows,
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    headers = [
        "key",
        "method",
        "mode",
        "variant",
        "nDCG@5",
        "Recall@5",
        "MRR",
        "nDCG recovery",
        "Recall recovery",
        "MRR recovery",
    ]
    lines = ["# Local HR Analysis Summary", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary["rows"]:
        values = [
            row.get("key"),
            row.get("method"),
            row.get("mode"),
            row.get("variant"),
            row.get("ndcg@5"),
            row.get("recall@5"),
            row.get("mrr"),
            row.get("ndcg@5_recovery"),
            row.get("recall@5_recovery"),
            row.get("mrr_recovery"),
        ]
        lines.append("| " + " | ".join(fmt(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(pick_latest_by_key(load_json_results(args.results_dir)))

    json_path = args.output_dir / "analysis_summary.json"
    md_path = args.output_dir / "analysis_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md_path.write_text(render_markdown(summary))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
