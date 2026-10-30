#!/usr/bin/env python3
"""Policy alone against policy with repair (Remark 8), over a trained run's held-out pool.

  python scripts/eval_repair.py --name mac-5x7-1M
  python scripts/eval_repair.py --name mac-5x7-1M --budgets 0 0.05 0.2 1 5 --w0 1
  python scripts/eval_repair.py --name mac-5x7-1M --first-feasible   # the first-feasible form, for comparison

For every held-out instance of runs/<name>/ (pools.json, latest.zip), the policy
builds a roster x^(T) (deterministic rollout by default, or the best of k
stochastic rollouts with --source best), and the repair operator of Algorithm 1
runs on it once per budget tau_cp in --budgets. tau_cp = 0 is the policy alone
(the endpoint of the quality-latency dial, Section 4.4). Every roster is
evaluated under the published profile (BURKE_CURTOIS), against the CP-SAT
reference stored in pools.json.

results/repair/<name>[-best]-anytime/   or   <name>[-best]-first/  with --first-feasible (committed unless the name starts with provisional-)
  results.csv   one row per (instance, tau_cp): J, feasibility, violations, window reached,
                |V_w| per subproblem, employees in V^ter, T_policy, T_repair, J and time of CP-SAT
  run.json      machine, versions, parameters, and the summary per tau_cp

Timings are wall-clock on this machine: run it natively, not in a VM.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

FIELDS = ["instance", "tau_cp", "w0", "anytime", "source", "J_cp", "cp_status", "cp_seconds",
          "J_policy", "feasible_policy", "policy_ms", "J", "feasible", "gap_to_cp", "violations",
          "window", "windows", "statuses", "n_free", "n_vars", "n_ter_employees", "n_empty_pairs",
          "changed_cells", "repair_seconds", "infer_seconds"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="training run: runs/<name>/")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.0, 0.1, 0.5, 2.0],
                    help="tau_cp values in seconds; 0 = policy alone")
    ap.add_argument("--w0", type=int, default=1, help="initial window of Algorithm 1")
    ap.add_argument("--first-feasible", action="store_true",
                    help="stop at the first feasible window (the 25/10 PDF's form) instead of the anytime "
                         "form (default; see nrrp/repair.py)")
    ap.add_argument("--source", choices=["det", "best"], default="det",
                    help="repair the deterministic rollout, or the best of --k-samples stochastic ones")
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8, help="CP-SAT workers in the repair")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results/repair"))
    ap.add_argument("--out-name", help="folder under --results (default: <name>, plus -best with --source best)")
    args = ap.parse_args()

    import torch
    from sb3_contrib import MaskablePPO

    from nrrp.gym_env import GymRosterEnv
    from nrrp.instance import Context
    from nrrp.profile import BURKE_CURTOIS, burke_curtois_training
    from nrrp.repair import repair
    from train import rebuild_pool, versions  # scripts/train.py, same folder

    run = args.runs / args.name
    pools = json.loads((run / "pools.json").read_text())
    config = json.loads((run / "config.json").read_text())
    targs = config["args"]
    n, m, k = pools["size"]
    kappa = int(targs.get("kappa", 1))
    insts = rebuild_pool(n, m, k, pools["eval"])
    train_profile = burke_curtois_training(w_min_minutes=targs["w_min_minutes"],
                                           w_min_consecutive=targs["w_min_consecutive"])
    env = GymRosterEnv(insts, train_profile, eval_profile=BURKE_CURTOIS, kappa=kappa,
                       reward_scale=targs.get("reward_scale", 0.01))
    torch.set_num_threads(1)               # inference latency of one rollout, one thread
    model = MaskablePPO.load(run / "latest.zip", env=env, device="cpu")
    rng = np.random.default_rng(args.seed)

    def rollout(idx, deterministic):
        obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)), options={"instance": idx})
        done, info = False, {}
        while not done:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=deterministic)
            obs, _, done, _, info = env.step(int(a))
        return env.roster, info["evaluation"]

    out = args.results / (args.out_name or (args.name + ("-best" if args.source == "best" else "")
                                            + ("-first" if args.first_feasible else "-anytime")))
    out.mkdir(parents=True, exist_ok=True)
    budgets = sorted(set(args.budgets))
    print(f"{len(insts)} held-out {n}x{m}x{k} instances, model at {model.num_timesteps:,d} steps; "
          f"source {args.source}, w0 {args.w0}, tau_cp {budgets} s", flush=True)
    rows = []
    for idx, (inst, rec) in enumerate(zip(insts, pools["eval"])):
        ctx = Context(inst)
        t0 = time.perf_counter()
        if args.source == "det":
            x, ev = rollout(idx, True)
        else:
            cands = [rollout(idx, False) for _ in range(args.k_samples)]
            x, ev = min(cands, key=lambda c: (not c[1]["feasible"], c[1]["J"]))
        t_pol = time.perf_counter() - t0
        for tau in budgets:
            if tau > 0:
                r = repair(x, ctx, BURKE_CURTOIS, w0=args.w0, budget=tau, kappa=kappa,
                           workers=args.workers, seed=args.seed, anytime=not args.first_feasible)
                J, feas, viol = r.J, r.feasible, r.violations
                extra = {"window": r.window, "windows": json.dumps(r.windows), "statuses": json.dumps(r.statuses),
                         "n_free": json.dumps(r.n_free), "n_ter_employees": r.n_ter_employees,
                         "n_empty_pairs": r.n_empty_pairs, "changed_cells": r.changed_cells,
                         "repair_seconds": round(r.seconds, 4)}
            else:
                J, feas, viol = ev["J"], ev["feasible"], ev["violations"]
                extra = {"window": "", "windows": "[]", "statuses": "[]", "n_free": "[]", "n_ter_employees": "",
                         "n_empty_pairs": "", "changed_cells": 0, "repair_seconds": 0.0}
            rows.append({"instance": inst.name, "tau_cp": tau, "w0": args.w0, "anytime": not args.first_feasible,
                         "source": args.source,
                         "J_cp": rec["J_cp"], "cp_status": rec["cp_status"], "cp_seconds": rec["cp_seconds"],
                         "J_policy": ev["J"], "feasible_policy": ev["feasible"],
                         "policy_ms": round(1000 * t_pol, 2), "J": J, "feasible": feas,
                         "gap_to_cp": J - rec["J_cp"], "violations": json.dumps(viol),
                         "n_vars": n * m * k, "infer_seconds": round(t_pol + extra["repair_seconds"], 4), **extra})
        print(f"  {inst.name:22s} CP-SAT {rec['J_cp']:6.0f}   policy {ev['J']:6.0f} ({'feasible' if ev['feasible'] else 'infeasible'})"
              + "".join(f"   tau={r['tau_cp']:g}: {r['J']:6.0f}{'' if r['feasible'] else '*'}"
                        for r in rows if r["instance"] == inst.name and r["tau_cp"] > 0), flush=True)

    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    summary = {}
    print(f"\n{'tau_cp':>8s} {'mean J':>8s} {'feasible':>9s} {'mean J - J_cp':>14s} {'mean T_repair':>14s} "
          f"{'max T_repair':>13s}   (CP-SAT mean J {np.mean([r['J_cp'] for r in pools['eval']]):.1f})")
    for tau in budgets:
        sel = [r for r in rows if r["tau_cp"] == tau]
        s = {"mean_J": float(np.mean([r["J"] for r in sel])),
             "feasible": float(np.mean([bool(r["feasible"]) for r in sel])),
             "mean_gap_to_cp": float(np.mean([r["gap_to_cp"] for r in sel])),
             "mean_repair_seconds": float(np.mean([r["repair_seconds"] for r in sel])),
             "max_repair_seconds": float(np.max([r["repair_seconds"] for r in sel]))}
        summary[str(tau)] = s
        print(f"{tau:8g} {s['mean_J']:8.1f} {s['feasible']:9.0%} {s['mean_gap_to_cp']:14.1f} "
              f"{s['mean_repair_seconds']:13.3f}s {s['max_repair_seconds']:12.3f}s")
    (out / "run.json").write_text(json.dumps({
        "training_run": args.name, "steps": model.num_timesteps, "args": {k_: str(v) if isinstance(v, Path) else v
                                                                        for k_, v in vars(args).items()},
        "machine": versions(), "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"), "summary": summary},
        indent=2) + "\n")
    print(f"\nwritten {out}/results.csv (* = infeasible)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
