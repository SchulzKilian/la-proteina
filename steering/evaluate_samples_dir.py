"""Evaluate biophysical properties for a directory of generated .pt samples.

Thin wrapper around steering.property_evaluate.evaluate_directory that:
  - Reads all .pt files from a samples dir
  - Computes all 13 developability properties (incl. TANGO + FreeSASA-based)
  - Writes a single CSV to the parent dir

Usage:
    python -m steering.evaluate_samples_dir \\
        --samples_dir results/generated_baseline_300_800/samples \\
        --output_csv results/generated_baseline_300_800/properties_generated.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steering.property_evaluate import evaluate_directory  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--skip-tango", action="store_true",
                        help="Skip TANGO (aggregation) computation — faster but drops 2 columns.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    samples_dir = Path(args.samples_dir)
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"No such dir: {samples_dir}")

    df = evaluate_directory(samples_dir, skip_tango=args.skip_tango)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols to {out_csv}")


if __name__ == "__main__":
    main()
