#!/usr/bin/env python3
"""Sample per-stage resources for one sequential final-chain OOF fold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--failure-file", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--hard-free-disk-gib", type=float, default=6.0)
    parser.add_argument("--transient-budget-gib", type=float, default=5.0)
    return parser.parse_args()


def terminate_tree(root: int) -> None:
    """Terminate the fold launcher tree without killing this monitor."""

    current = os.getpid()
    targets = descendants(root) - {current}
    for pid in sorted(targets, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def descendants(root: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    result = {int(root)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def gpu_memory(pids: set[int]) -> tuple[int, list[dict[str, int]]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0, []
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid_text, memory_text = [value.strip() for value in line.split(",", 1)]
        pid = int(pid_text)
        if pid in pids:
            rows.append({"pid": pid, "used_memory_mib": int(memory_text)})
    return sum(row["used_memory_mib"] for row in rows), rows


def current_stage(run_dir: Path) -> str:
    manifest = run_dir / "pipeline_manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            running = [
                name
                for name, state in dict(payload.get("stages") or {}).items()
                if isinstance(state, dict) and state.get("status") == "running"
            ]
            if running:
                return running[0]
            if payload.get("sealed") is True:
                return "postseal"
        except (OSError, ValueError):
            pass
    return "preflight"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    samples_path = Path(args.samples).resolve()
    summary_path = Path(args.summary).resolve()
    stop_file = Path(args.stop_file).resolve()
    failure_file = Path(args.failure_file).resolve()
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    baseline_run = tree_bytes(run_dir)
    baseline_output = tree_bytes(output_dir)
    stages: dict[str, dict] = {}
    hard_stop_reason = None
    max_transient = 0
    max_gpu = 0
    while not stop_file.exists():
        now = time.time()
        stage = current_stage(run_dir)
        pids = descendants(args.root_pid)
        gpu_mib, gpu_processes = gpu_memory(pids)
        run_bytes = tree_bytes(run_dir)
        output_bytes = tree_bytes(output_dir)
        transient = max(0, run_bytes - baseline_run) + max(0, output_bytes - baseline_output)
        disk_free = shutil.disk_usage(run_dir.parent if run_dir.parent.exists() else Path.cwd()).free
        max_transient = max(max_transient, transient)
        max_gpu = max(max_gpu, gpu_mib)
        state = stages.setdefault(
            stage,
            {
                "first_timestamp": now,
                "last_timestamp": now,
                "peak_gpu_mib": 0,
                "start_disk_free_bytes": disk_free,
                "end_disk_free_bytes": disk_free,
                "start_fold_bytes": run_bytes + output_bytes,
                "end_fold_bytes": run_bytes + output_bytes,
            },
        )
        state["last_timestamp"] = now
        state["peak_gpu_mib"] = max(int(state["peak_gpu_mib"]), gpu_mib)
        state["end_disk_free_bytes"] = disk_free
        state["end_fold_bytes"] = run_bytes + output_bytes
        sample = {
            "timestamp": now,
            "fold_id": args.fold_id,
            "stage": stage,
            "gpu_mib": gpu_mib,
            "gpu_processes": gpu_processes,
            "disk_free_bytes": disk_free,
            "run_bytes": run_bytes,
            "output_bytes": output_bytes,
            "transient_bytes": transient,
        }
        with samples_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
        if disk_free < int(args.hard_free_disk_gib * 1024**3):
            hard_stop_reason = "hard_free_disk_floor"
        elif transient > int(args.transient_budget_gib * 1024**3):
            hard_stop_reason = "transient_fold_budget"
        if hard_stop_reason:
            stop_file.write_text(hard_stop_reason + "\n", encoding="utf-8")
            terminate_tree(args.root_pid)
            break
        time.sleep(float(args.interval))
    ended = time.time()
    for state in stages.values():
        state["wall_time_seconds"] = float(state["last_timestamp"] - state["first_timestamp"])
    failure_reason = None
    if failure_file.exists():
        failure_reason = failure_file.read_text(encoding="utf-8").strip() or "unknown"
    status = "HARD_STOP" if hard_stop_reason else ("FAILED" if failure_reason else "COMPLETE")
    summary = {
        "kind": "final_chain_oof_fold_resource_summary",
        "format_version": 1,
        "status": status,
        "fold_id": args.fold_id,
        "started_timestamp": started,
        "ended_timestamp": ended,
        "wall_time_seconds": ended - started,
        "peak_gpu_mib": max_gpu,
        "maximum_transient_bytes": max_transient,
        "final_run_bytes": tree_bytes(run_dir),
        "final_output_bytes": tree_bytes(output_dir),
        "hard_stop_reason": hard_stop_reason,
        "failure_reason": failure_reason,
        "stages": stages,
        "test_accessed": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
