#!/usr/bin/env python3
"""How often a disruption leaves no feasible roster, with and without the absence credit (Remark 3).

For each of ``--n`` disruptions xi ~ P (seeds from ``--seed0``) on a Burke & Curtois instance,
CP-SAT decides whether X(phi) is empty under two readings of the hard constraints:

  credited    the reading of the paper and of the code (decision of 10/11/2026): the shifts x° had
              on the absent days count towards H^min (5), and a work stretch next to an absent day
              is exempt from the minimum stretch (9), as at an end of the horizon;
  uncredited  the benchmark's constraints read literally: neither of the two.

Only the CP-SAT model is affected by the reading; absences are forbidden cells in both.

  python scripts/feasibility_rate.py --instance data/raw/Instance5.txt --n-absent 1 2 --absence-days 1 7
  python scripts/feasibility_rate.py --instance data/raw/Instance5.txt --n-absent 1 2 --absence-days 1 1

Output: results/feasibility/<name>/results.csv (one row per disruption and reading, written row
by row) and run.json (parameters, machine, and per reading the counts of feasible, infeasible and
undecided disruptions). A disruption is undecided when CP-SAT neither finds a roster nor proves
there is none within --time-limit; it is reported, not counted as infeasible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nrrp.cpsat as cpsat  # noqa: E402
from nrrp.bc_format import read_bc, read_roster, solution_name  # noqa: E402
from nrrp.disruption import DisruptionModel  # noqa: E402
from nrrp.profile import BURKE_CURTOIS  # noqa: E402

READINGS = ("credited", "uncredited")


def best_published(instance_file: Path) -> Path:
    """The lowest-valued published solution next to the instance (as in scripts/train.py)."""
    found = [(solution_name(f)[1], f) for f in sorted((instance_file.parent / "solutions").glob("*.roster"))
             if solution_name(f) and solution_name(f)[0] == instance_file.stem]
    if not found:
        raise FileNotFoundError(f"no published solution for {instance_file.stem} in {instance_file.parent / 'solutions'}")
    return min(found)[1]


def machine() -> dict:
    out = {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version()}
    try:
        out["ortools"] = __import__("ortools").__version__
    except ImportError:
        out["ortools"] = None
    return out


@contextmanager
def uncredited(ctx):
    """The literal reading of a context: no credit towards H^min, and no exemption from (9)
    next to an absent day. Restores the context and the CP-SAT encoder on exit."""
    credit, encoder = ctx.credit, cpsat._interior_run_patterns

    def no_boundary(*args, boundary=None, **kwargs):            # absences no longer end a stretch
        return encoder(*args, **kwargs)

    ctx.credit = np.zeros_like(credit)
    cpsat._interior_run_patterns = no_boundary
    try:
        yield ctx
    finally:
        ctx.credit = credit
        cpsat._interior_run_patterns = encoder


def decide(ctx, reading, time_limit, workers):
    """(status, seconds) of CP-SAT on X(phi) under ``reading``: feasible, infeasible or undecided."""
    def run():
        return cpsat.solve(ctx, BURKE_CURTOIS, time_limit=time_limit, workers=workers, seed=0)
    if reading == "credited":
        res = run()
    else:
        with uncredited(ctx):
            res = run()
    if res.status in ("UNKNOWN", "FEASIBLE") and not res.has_solution and res.wall_time < 0.9 * time_limit:
        raise KeyboardInterrupt                                  # Ctrl+C, caught by CP-SAT
    if res.has_solution and res.feasible:
        status = "feasible"
    elif res.status == "INFEASIBLE":
        status = "infeasible"
    else:
        status = "undecided"
    return status, res.wall_time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instance", type=Path, required=True, help="Burke & Curtois instance file")
    ap.add_argument("--published", type=Path, help="x° (default: the lowest-valued published solution)")
    ap.add_argument("--n-absent", type=int, nargs=2, default=[1, 2], metavar=("LO", "HI"))
    ap.add_argument("--absence-days", type=int, nargs=2, default=[1, 7], metavar=("LO", "HI"))
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=60, help="number of disruptions")
    ap.add_argument("--seed0", type=int, default=300000, help="first seed (disjoint from evaluation and validation)")
    ap.add_argument("--time-limit", type=float, default=10.0, help="CP-SAT seconds per disruption and reading")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", help="folder under --results (default from the instance and P)")
    ap.add_argument("--results", type=Path, default=Path("results/feasibility"))
    args = ap.parse_args(argv)

    spec = {"instance": str(args.instance), "published": str(args.published or best_published(args.instance)),
            "n_absent": args.n_absent, "days": args.absence_days, "mu": args.mu, "first_day": 1}
    inst = read_bc(args.instance)
    P = DisruptionModel(inst, read_roster(spec["published"], inst), n_absent=tuple(args.n_absent),
                        days=tuple(args.absence_days), mu=args.mu, first_day=1)
    name = args.name or (f"{args.instance.stem}-a{args.n_absent[0]}-{args.n_absent[1]}"
                         f"-l{args.absence_days[0]}-{args.absence_days[1]}")
    out = args.results / name
    out.mkdir(parents=True, exist_ok=True)
    counts = {r: {"feasible": 0, "infeasible": 0, "undecided": 0} for r in READINGS}
    t0 = time.time()
    fields = ["seed", "d_xi", "absent_employees", "absent_days", "reading", "status", "seconds"]
    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        try:
            for k, seed in enumerate(range(args.seed0, args.seed0 + args.n)):
                ctx = P.context(seed)
                row = {"seed": seed, "d_xi": ctx.d_xi, "absent_employees": int(ctx.absent.any(axis=1).sum()),
                       "absent_days": int(ctx.absent.sum())}
                for reading in READINGS:
                    status, seconds = decide(ctx, reading, args.time_limit, args.workers)
                    counts[reading][status] += 1
                    w.writerow({**row, "reading": reading, "status": status, "seconds": round(seconds, 3)})
                fh.flush()
                print(f"{k + 1:>3}/{args.n}  seed {seed}  d_xi {ctx.d_xi:>2}  "
                      + "  ".join(f"{r}: {counts[r]['infeasible']} infeasible" for r in READINGS), flush=True)
        except KeyboardInterrupt:
            print("interrupted: the rows written so far are kept")
    done = sum(counts["credited"].values())
    summary = {r: {**c, "infeasible_share": round(c["infeasible"] / done, 4) if done else None}
               for r, c in counts.items()}
    (out / "run.json").write_text(json.dumps({
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reroster": spec, "machine": machine(), "disruptions": done,
        "seconds": round(time.time() - t0, 1), "summary": summary}, indent=2))
    for r in READINGS:
        s = summary[r]
        print(f"{r:>10}: {s['infeasible']}/{done} infeasible ({100 * (s['infeasible_share'] or 0):.0f}%), "
              f"{s['undecided']} undecided")
    print(f"written {out}/results.csv, run.json")


if __name__ == "__main__":
    main()
