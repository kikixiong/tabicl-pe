#!/usr/bin/env python3
"""Read-only 30-minute monitor for the targeted legacy-pilot continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTION_INFO = {
    "rope-stage1-479k-to-500k": ("rope/seed-42/stage1", 500000, "rope-s1", True),
    "rope-stage2-after-500k": ("rope/seed-42/stage2", 40000, "rope-s2", False),
    "none-stage2-6200-to-40k": ("none/seed-42/stage2", 40000, "none-s2", False),
}
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
STEP_FILE = re.compile(r"^step-([1-9][0-9]*)\.ckpt$")
ANOMALY = re.compile(
    rb"(?:\bOOM\b|out of memory|OutOfMemoryError|\bnan\b|non-finite|"
    rb"Traceback|ENOSPC|No space left|NCCL[^\r\n]*error|CUDA[^\r\n]*error)",
    re.IGNORECASE,
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("kind") != "tabicl-legacy-pilot-continuation-receipt"
    ):
        raise ValueError("receipt kind is invalid")
    jobs = value.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != set(ACTION_INFO):
        raise ValueError("receipt job matrix is invalid")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != hashlib.sha256(canonical(body)).hexdigest():
        raise ValueError("receipt self-hash is invalid")
    return value


def latest_checkpoint(directory: Path) -> tuple[int, Path] | None:
    values: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return None
    for child in directory.iterdir():
        match = STEP_FILE.fullmatch(child.name)
        if match is not None and child.is_file() and not child.is_symlink():
            values.append((int(match.group(1)), child))
    return max(values, default=None)


def checkpoint_crc(path: Path) -> tuple[bool, str]:
    try:
        metadata_before = path.stat()
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
        metadata_after = path.stat()
    except (OSError, zipfile.BadZipFile) as error:
        return False, str(error)
    stable = (
        metadata_before.st_size == metadata_after.st_size
        and metadata_before.st_mtime_ns == metadata_after.st_mtime_ns
        and metadata_before.st_ino == metadata_after.st_ino
    )
    if corrupt is not None:
        return False, f"CRC failure in {corrupt}"
    if not stable:
        return False, "file changed during CRC validation"
    return True, "ok"


def tail_has_anomaly(path: Path, max_bytes: int = 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        raw = handle.read(max_bytes)
    match = ANOMALY.search(raw)
    return match.group(0).decode("utf-8", errors="replace") if match else None


def gpu_summary(path: Path) -> dict[int, float] | None:
    if not path.is_file():
        return None
    samples: dict[int, list[float]] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            try:
                index = int(row[1].strip())
                utilization = float(row[3].strip())
            except ValueError:
                continue
            samples.setdefault(index, []).append(utilization)
    result: dict[int, float] = {}
    for index, values in samples.items():
        first = next((i for i, value in enumerate(values) if value >= 10), len(values))
        last = next(
            (
                len(values) - i
                for i, value in enumerate(reversed(values))
                if value >= 10
            ),
            0,
        )
        active = values[first:last][12:]
        if active:
            result[index] = statistics.fmean(active)
    return result or None


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}: {result.stderr.strip()}"
        )
    return result.stdout


def query_states(job_ids: list[str]) -> dict[str, str]:
    states: dict[str, str] = {}
    joined = ",".join(job_ids)
    for line in run(["squeue", "-h", "-j", joined, "-o", "%i|%T"]).splitlines():
        fields = line.strip().split("|", 1)
        if len(fields) == 2:
            states[fields[0]] = fields[1].split("+", 1)[0].upper()
    missing = [job_id for job_id in job_ids if job_id not in states]
    if missing:
        output = run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(missing),
                "--format=JobIDRaw,State",
            ]
        )
        for line in output.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 2 and fields[0] in missing:
                states[fields[0]] = fields[1].split("+", 1)[0].upper()
    return states


def snapshot_manifest_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "not-created"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        body = {key: item for key, item in value.items() if key != "manifest_sha256"}
        if value.get("manifest_sha256") != hashlib.sha256(canonical(body)).hexdigest():
            return False, "self-hash-invalid"
        checkpoint = path.parent / value["snapshot_filename"]
        if checkpoint.stat().st_size != value["checkpoint_size_bytes"]:
            return False, "checkpoint-size-mismatch"
        if file_sha256(checkpoint) != value["checkpoint_sha256"]:
            return False, "checkpoint-sha256-mismatch"
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
    ) as error:
        return False, f"invalid:{error}"
    return True, "ok"


def monitor_once(
    receipt: dict[str, Any], previous_steps: dict[str, int]
) -> tuple[bool, dict[str, int]]:
    root = Path(receipt["repository_root"])
    continuation_id = receipt["continuation_id"]
    jobs: dict[str, str] = receipt["jobs"]
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] pilot continuation {continuation_id}")

    try:
        states = query_states(list(jobs.values()))
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ALERT scheduler query failed: {error}")
        states = {}

    disk = shutil.disk_usage(root)
    free_gib = disk.free / (1024**3)
    level = "CRITICAL" if free_gib < 20 else "WARNING" if free_gib < 22 else "OK"
    print(f"disk={free_gib:.1f}GiB level={level}")

    next_steps = dict(previous_steps)
    all_terminal = bool(states) and all(
        states.get(job_id) in TERMINAL_STATES for job_id in jobs.values()
    )
    for action, job_id in jobs.items():
        relative_dir, terminal_step, log_suffix, stage1_hard_gate = ACTION_INFO[action]
        state = states.get(job_id, "UNKNOWN")
        checkpoint_dir = root / "artifacts/tabiclv2-clf-identity" / relative_dir
        latest = latest_checkpoint(checkpoint_dir)
        checkpoint_text = "none"
        if latest is not None:
            step, checkpoint = latest
            next_steps[action] = step
            crc_ok, crc_detail = checkpoint_crc(checkpoint)
            age_minutes = (time.time() - checkpoint.stat().st_mtime) / 60
            checkpoint_text = (
                f"step={step}/{terminal_step} age={age_minutes:.0f}m "
                f"crc={'ok' if crc_ok else 'FAIL:' + crc_detail}"
            )
            if not crc_ok:
                print(f"ALERT {action} checkpoint integrity failed: {crc_detail}")
            previous = previous_steps.get(action)
            if state == "RUNNING" and previous == step and age_minutes >= 90:
                print(f"ALERT {action} checkpoint has not advanced for >=90 minutes")

        out_path = (
            root
            / "artifacts/logs"
            / f"pilot-{continuation_id}-{log_suffix}-{job_id}.out"
        )
        err_path = (
            root
            / "artifacts/logs"
            / f"pilot-{continuation_id}-{log_suffix}-{job_id}.err"
        )
        anomalies = [
            item
            for item in (tail_has_anomaly(out_path), tail_has_anomaly(err_path))
            if item
        ]
        if anomalies:
            print(f"ALERT {action} log signature: {', '.join(sorted(set(anomalies)))}")

        gpu_path = (
            root
            / "artifacts/gpu-monitor"
            / f"pilot-{continuation_id}-{action}-{job_id}.csv"
        )
        means = gpu_summary(gpu_path)
        gpu_text = "not-yet-available"
        if means:
            gpu_text = ",".join(
                f"gpu{index}={mean:.1f}%" for index, mean in sorted(means.items())
            )
            if any(mean < 80 for mean in means.values()):
                classification = "hard-gate" if stage1_hard_gate else "diagnostic-only"
                print(
                    f"WARNING {action} GPU mean below 80% ({classification}): {gpu_text}"
                )
        print(
            f"job={job_id} action={action} state={state} checkpoint={checkpoint_text} gpu={gpu_text}"
        )

    snapshot_root = root / "artifacts/pilot-continuation-snapshots" / continuation_id
    for mode in ("none", "rope"):
        manifest = snapshot_root / mode / "stage1-step500000" / "snapshot-manifest.json"
        exists, detail = snapshot_manifest_status(manifest)
        if manifest.exists() and not exists:
            print(f"ALERT {mode} 500k snapshot manifest: {detail}")
        elif exists:
            print(f"snapshot={mode}-stage1-500k status=verified")
    return all_terminal, next_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.once and args.interval_seconds < 60:
        print("monitor interval must be at least 60 seconds", file=sys.stderr)
        return 2
    try:
        receipt = load_receipt(args.receipt)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        print(f"pilot monitor receipt validation failed: {error}", file=sys.stderr)
        return 1

    previous_steps: dict[str, int] = {}
    while True:
        terminal, previous_steps = monitor_once(receipt, previous_steps)
        sys.stdout.flush()
        if args.once or terminal:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
