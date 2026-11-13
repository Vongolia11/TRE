#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tre_dllm.scoring import score_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TRE scores from saved entropy trajectories.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--empty-value", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = score_manifest(args.manifest, empty_value=args.empty_value)

    scores_path = output_dir / "tre_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "question_id", "tre_score", "label", "entropy_file"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "tre_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved scores: {scores_path}")
    if summary:
        print(f"Saved AUROC summary: {summary_path}")
        print(json.dumps(summary, indent=2))
    else:
        print("No labels found in manifest; AUROC evaluation was skipped.")


if __name__ == "__main__":
    main()
