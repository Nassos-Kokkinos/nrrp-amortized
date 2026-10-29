#!/usr/bin/env python3
"""Solve the construction problem (15) with CP-SAT and record the results.

  python scripts/solve_cp.py data/raw/Instance1.txt data/raw/Instance2.txt --run mac-60s
  python scripts/solve_cp.py data/raw/Instance{1..7}.txt --time-limit 60 --run mac-60s

Each instance is solved under the evaluation profile of the family
(BURKE_CURTOIS) from scratch. Results go to results/cp_baseline/<run>/:

  results.csv      one row per instance, written as soon as it is solved
  run.json         machine, versions and parameters of the run
  solutions/       the best roster found, InstanceN.cp.roster (usable as x°)

A run name starting with "provisional-" is ignored by git: use it for runs
whose timings are not meant for the paper (e.g. inside a VM). The reported J is
recomputed by objective.J, not taken from the solver. Plot with
scripts/plot_results.py cp results/cp_baseline/<run>.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from nrrp.bc_format import PUBLISHED, read_bc, write_roster
from nrrp.cpsat import solve
from nrrp.instance import Context
from nrrp.profile import BURKE_CURTOIS

FIELDS = ["instance", "n_emp", "n_days", "n_shifts", "status", "J", "bound", "gap_to_bound",
          "best_known", "gap_to_best_known", "feasible", "solve_seconds", "build_seconds",
          "deterministic_time", "time_limit", "workers", "seed"]


def machine_info() -> dict:
    import ortools
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__,
            "ortools": ortools.__version__}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="instance files in the Burke & Curtois text format")
    ap.add_argument("--run", default=time.strftime("run-%Y%m%d-%H%M"), help="run name (folder under --root)")
    ap.add_argument("--root", type=Path, default=Path("results/cp_baseline"))
    ap.add_argument("--time-limit", type=float, default=60.0, help="wall-clock seconds per instance")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.root / args.run
    (out / "solutions").mkdir(parents=True, exist_ok=True)
    info = {"run": args.run, "started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "machine": machine_info(),
            "profile": BURKE_CURTOIS.name, "time_limit": args.time_limit, "workers": args.workers,
            "seed": args.seed, "files": args.files}
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"{info['machine']['platform']} | Python {info['machine']['python']} | ortools "
          f"{info['machine']['ortools']} | {args.workers} workers, {args.time_limit:g} s per instance")
    print(f"writing {out}/results.csv\n")

    csv_path = out / "results.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for f in args.files:
            inst = read_bc(f)
            res = solve(Context(inst), BURKE_CURTOIS, time_limit=args.time_limit,
                        workers=args.workers, seed=args.seed)
            best = PUBLISHED.get(inst.name, (None,) * 4)[3]
            gap_best = (res.J - best) / best if (res.J is not None and best) else None
            row = {"instance": inst.name, "n_emp": inst.n_emp, "n_days": inst.n_days,
                   "n_shifts": inst.n_shifts, "status": res.status, "J": res.J, "bound": res.bound,
                   "gap_to_bound": res.gap, "best_known": best, "gap_to_best_known": gap_best,
                   "feasible": res.feasible, "solve_seconds": round(res.wall_time, 3),
                   "build_seconds": round(res.build_time, 3),
                   "deterministic_time": round(res.info.get("deterministic_time", 0.0), 3),
                   "time_limit": args.time_limit, "workers": args.workers, "seed": args.seed}
            writer.writerow(row)
            fh.flush()
            if res.has_solution:
                write_roster(res.roster, inst, out / "solutions" / f"{inst.name}.cp.roster", penalty=res.J)
            fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
            print(f"{inst.name:12s} {inst.size:32s} {res.status:9s} J={fmt(res.J, 'g'):>8s} "
                  f"bound={fmt(res.bound, '.0f'):>7s} best known={fmt(best, 'g'):>6s} "
                  f"gap to best={fmt(gap_best, '+.1%'):>7s}  {res.wall_time:6.1f} s", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
