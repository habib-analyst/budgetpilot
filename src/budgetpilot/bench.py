"""Benchmark CLI: run the task suite under the strategies, write results.

Resilient by design: every finished task is appended (and fsynced) to a
per-task JSONL immediately, and `--resume` skips tasks already present
there, so a crashed or restarted machine loses at most one task.

Usage:
    python -m budgetpilot.bench --small-path PATH --large-path PATH \
        --out results.json [--seed 42] [--strategies a,b,c] [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .analytics import PRICE_PER_TOKEN, summarize
from .models import LocalModel
from .runner import run_task
from .tasks import build_suite

ALL_STRATEGIES = ["always_small", "always_large", "budgetpilot"]


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _assemble(rows: list[dict], config: dict, out: str) -> dict:
    results = {"config": config, "strategies": {}, "per_task": rows}
    for strat in ALL_STRATEGIES:
        srows = [r for r in rows if r["strategy"] == strat]
        if srows:
            results["strategies"][strat] = summarize(srows)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-path", required=True)
    ap.add_argument("--large-path", required=True)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--per-task-out", default="per_task.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--strategies", default=",".join(ALL_STRATEGIES))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    tasks = build_suite(args.seed)
    if args.limit:
        tasks = tasks[: args.limit]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    print(f"[bench] {len(tasks)} tasks, seed {args.seed}, strategies {strategies}", flush=True)

    done_keys: set[tuple[str, str]] = set()
    rows = _load_jsonl(args.per_task_out) if args.resume else []
    if args.resume:
        done_keys = {(r["strategy"], r["task_id"]) for r in rows}
        print(f"[bench] resume: {len(done_keys)} task runs already recorded", flush=True)

    need_small = any(s in ("always_small", "budgetpilot") for s in strategies)
    need_large = any(s in ("always_large", "budgetpilot") for s in strategies)
    models: dict = {}
    if need_small:
        print("[bench] loading small model...", flush=True)
        models["small"] = LocalModel(args.small_path, "small")
    if need_large:
        print("[bench] loading large model...", flush=True)
        models["large"] = LocalModel(args.large_path, "large")

    config = {
        "seed": args.seed,
        "tasks": len(tasks),
        "small_model": args.small_path,
        "large_model": args.large_path,
        "price_per_token_usd": PRICE_PER_TOKEN,
        "decoding": "greedy",
    }

    fout = open(args.per_task_out, "a")
    for strat in strategies:
        t0 = time.time()
        for i, task in enumerate(tasks):
            if (strat, task.task_id) in done_keys:
                continue
            row = run_task(task, strat, models)
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            print(
                f"[bench] {strat} {i + 1}/{len(tasks)} {task.task_id} "
                f"correct={row['correct']} cost=${row['cost_usd']:.6f}",
                flush=True,
            )
        srows = [r for r in rows if r["strategy"] == strat]
        print(f"[bench] {strat} cumulative: {summarize(srows)} ({time.time() - t0:.0f}s this run)", flush=True)
        _assemble(rows, config, args.out)
    fout.close()
    print(f"[bench] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
