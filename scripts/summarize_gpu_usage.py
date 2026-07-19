#!/usr/bin/env python3
"""Summarize per-GPU nvidia-smi samples and enforce a utilization gate."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--warmup-samples", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument(
        "--start-after-active",
        action="store_true",
        help="Discard cold-start samples before the first utilization value of at least 10%.",
    )
    parser.add_argument(
        "--end-after-active",
        action="store_true",
        help="Discard shutdown samples after the last utilization value of at least 10%.",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> int:
    args = parse_args()
    samples: dict[int, list[float]] = defaultdict(list)

    with args.csv_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            try:
                gpu_index = int(row[1].strip())
                utilization = float(row[3].strip())
            except ValueError:
                continue
            samples[gpu_index].append(utilization)

    failed = False
    if not samples:
        print("No valid GPU samples found")
        return 2

    for gpu_index, raw_values in sorted(samples.items()):
        start_index = 0
        end_index = len(raw_values)
        if args.start_after_active:
            start_index = next((i for i, value in enumerate(raw_values) if value >= 10), len(raw_values))
        if args.end_after_active:
            end_index = next(
                (len(raw_values) - i for i, value in enumerate(reversed(raw_values)) if value >= 10),
                0,
            )
        raw_values = raw_values[start_index:end_index]
        values = raw_values[args.warmup_samples :]
        if len(values) < args.min_samples:
            print(f"GPU {gpu_index}: only {len(values)} post-warmup samples")
            failed = True
            continue
        mean = statistics.fmean(values)
        median = statistics.median(values)
        p10 = percentile(values, 0.10)
        print(
            f"GPU {gpu_index}: samples={len(values)} mean={mean:.1f}% "
            f"median={median:.1f}% p10={p10:.1f}%"
        )
        failed |= mean < args.threshold

    if len(samples) != args.expected_gpus:
        print(f"Expected {args.expected_gpus} GPUs, found {len(samples)}")
        failed = True

    if failed:
        print(f"GPU utilization gate failed (required per-GPU mean >= {args.threshold:.1f}%)")
        return 1
    print(f"GPU utilization gate passed (all per-GPU means >= {args.threshold:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
