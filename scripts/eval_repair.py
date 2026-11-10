#!/usr/bin/env python3
"""Policy alone against policy with repair (Remark 8), over a trained run's held-out pool.

  python scripts/eval_repair.py --name mac-5x7-1M
  python scripts/eval_repair.py --name mac-5x7-1M --budgets 0 0.05 0.2 1 5 --w0 1
  python scripts/eval_repair.py --name mac-5x7-1M --first-feasible   # the first-feasible form, for comparison
  python scripts/eval_repair.py --name 25x7-F-ext-sep --empty-kappa 0 # V^∅ = pairs below rho only

For every held-out instance of runs/<name>/ (pools.json, latest.zip), the policy
builds a roster x^(T) (deterministic rollout by default, or the best of k
stochastic rollouts with --source best), and the repair operator of Algorithm 1
runs on it once per budget tau_cp in --budgets. tau_cp = 0 is the policy alone
(the endpoint of the quality-latency dial, Section 4.4). Every roster is
evaluated under the published profile (BURKE_CURTOIS), against the CP-SAT
reference stored in pools.json.

--source random replaces the policy by a uniformly random masked construction
(same environment, same instances): the ablation that measures what the trained
policy contributes beyond the repair operator. A good policy leaves few H^ter
violations, hence a small V^ter and a small neighbourhood |V_w|; the summary
reports both.

--empty-kappa 0 restricts V^∅ to the pairs below rho (genuine under-coverage):
with kappa = 1, Eq. (40) as written also frees every pair whose optional
(rho + 1)-th slot was left empty, about half of all pairs on 25x7, whatever the
policy (measured 02/11/2026), so the neighbourhood is ~85% of the problem.

Re-rostering runs (train.py --reroster): the held-out pool is the run's
contexts phi = (Inst, x°, xi). --source published starts the repair from x°
with the absent employees' assignments removed and no learned policy, the
classical re-rostering baseline (does the policy add anything beyond repairing
the published roster?). A second baseline, CP-SAT on (20) hinted with x° (warm
start), with the same total time, is added unless --no-cp-hint. Every row
reports Delta(x, x°), the number of changed assignments.

Fair baseline (Eqs. (28), (42)): for every tau_cp > 0, CP-SAT from scratch on
the full problem with the same total response time as ours on that instance,
T_infer = T_policy + T_repair (building its model included), and the same
workers; per instance, win / tie / loss of policy + repair against it (a CP-SAT
run with no feasible roster is a loss for CP-SAT if ours is feasible).
--no-cp-scratch skips it.

results/repair/<name>[-best|-random]-anytime[-ek0]/   or   ...-first/  with --first-feasible (committed unless the name starts with provisional-)
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

FIELDS = ["instance", "tau_cp", "w0", "anytime", "source", "empty_kappa", "J_cp", "cp_status", "cp_seconds",
          "J_policy", "feasible_policy", "policy_ms", "J", "feasible", "gap_to_cp", "violations",
          "window", "windows", "statuses", "n_free", "n_vars", "n_ter_employees", "n_empty_pairs",
          "changed_cells", "repair_seconds", "infer_seconds",
          "J_cp_scratch", "feasible_cp_scratch", "cp_scratch_status", "cp_scratch_budget", "cp_scratch_seconds",
          "vs_cp_scratch", "d_xi", "changes", "changes_cp_scratch",
          "J_cp_hint", "feasible_cp_hint", "cp_hint_status", "cp_hint_seconds", "vs_cp_hint", "changes_cp_hint"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="training run: runs/<name>/")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.0, 0.1, 0.5, 2.0],
                    help="tau_cp values in seconds; 0 = policy alone")
    ap.add_argument("--w0", type=int, default=1, help="initial window of Algorithm 1")
    ap.add_argument("--first-feasible", action="store_true",
                    help="stop at the first feasible window (the 25/10 PDF's form) instead of the anytime "
                         "form (default; see nrrp/repair.py)")
    ap.add_argument("--source", choices=["det", "best", "random", "published"], default="det",
                    help="repair the deterministic rollout, the best of --k-samples stochastic ones, or a "
                         "uniformly random masked construction (ablation)")
    ap.add_argument("--empty-kappa", type=int, default=None,
                    help="kappa used for V^∅ (slots left unfilled): pairs below min(rho + this, n). Default: the "
                         "training kappa, as Eq. (40) is written; 0 = only pairs below rho (genuine under-coverage)")
    ap.add_argument("--no-cp-hint", action="store_true",
                    help="re-rostering: skip the baseline CP-SAT hinted with x° (same total time)")
    ap.add_argument("--no-cp-scratch", action="store_true",
                    help="skip the fair baseline: CP-SAT from scratch with the same total time T_infer and workers")
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
    from nrrp.cpsat import RosterModel
    from nrrp.disruption import changes, without_absences
    from nrrp.objective import evaluate
    from nrrp.profile import BURKE_CURTOIS, burke_curtois_training
    from nrrp.repair import repair
    from train import pool_contexts, versions  # scripts/train.py, same folder

    run = args.runs / args.name
    pools = json.loads((run / "pools.json").read_text())
    config = json.loads((run / "config.json").read_text())
    targs = config["args"]
    n, m, k = pools["size"]
    kappa = int(targs.get("kappa", 1))
    empty_kappa = kappa if args.empty_kappa is None else int(args.empty_kappa)
    reroster = pools.get("regime") == "reroster"
    if args.source == "published" and not reroster:
        sys.exit("--source published needs a re-rostering run (there is no x° in construction)")
    ctxs = pool_contexts(pools)
    insts = [c.instance for c in ctxs]
    train_profile = burke_curtois_training(w_min_minutes=targs["w_min_minutes"],
                                           w_min_consecutive=targs["w_min_consecutive"])
    env = GymRosterEnv(insts, train_profile, contexts=ctxs, eval_profile=BURKE_CURTOIS, kappa=kappa,
                       reward_scale=targs.get("reward_scale", 0.01), features=targs.get("features", "table2"))
    torch.set_num_threads(1)               # inference latency of one rollout, one thread
    model = MaskablePPO.load(run / "latest.zip", env=env, device="cpu")
    rng = np.random.default_rng(args.seed)

    def rollout(idx, deterministic, random_policy=False):
        obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)), options={"instance": idx})
        done, info = False, {}
        while not done:
            mask = env.action_masks()
            if random_policy:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            obs, _, done, _, info = env.step(int(a))
        return env.roster, info["evaluation"]

    out = args.results / (args.out_name or (args.name + {"det": "", "best": "-best", "random": "-random",
                                                          "published": "-published"}[args.source]
                                            + ("-first" if args.first_feasible else "-anytime")
                                            + ("" if empty_kappa == kappa else f"-ek{empty_kappa}")))
    out.mkdir(parents=True, exist_ok=True)
    budgets = sorted(set(args.budgets))
    print(f"{len(insts)} held-out {n}x{m}x{k} {'re-rostering contexts' if reroster else 'instances'}, "
          f"model at {model.num_timesteps:,d} steps; "
          f"source {args.source}, w0 {args.w0}, V^∅ below rho + {empty_kappa}, tau_cp {budgets} s", flush=True)
    def cp_baseline(ctx, target, hint=None):
        """CP-SAT on the full problem (20) with total time ``target``, building its model included."""
        t1 = time.perf_counter()
        cm = RosterModel(ctx, BURKE_CURTOIS)
        left = max(target - (time.perf_counter() - t1), 1e-3)
        cs = cm.solve(time_limit=left, workers=args.workers, seed=args.seed, hint=hint)
        ok = bool(cs.has_solution and cs.feasible)
        return cs, ok, round(time.perf_counter() - t1, 4)

    def verdict(feas, J, ok, J_other):
        if feas and (not ok or J < J_other - 1e-9):
            return "win"
        if ok and (not feas or J > J_other + 1e-9):
            return "loss"
        return "tie"

    rows = []
    for idx, (inst, ctx, rec) in enumerate(zip(insts, ctxs, pools["eval"])):
        name = ctx.name if reroster else inst.name
        t0 = time.perf_counter()
        if args.source == "det":
            x, ev = rollout(idx, True)
        elif args.source == "random":
            x, ev = rollout(idx, False, random_policy=True)
        elif args.source == "published":
            x = without_absences(ctx)
            ev = evaluate(x, ctx, BURKE_CURTOIS)
        else:
            cands = [rollout(idx, False) for _ in range(args.k_samples)]
            x, ev = min(cands, key=lambda c: (not c[1]["feasible"], c[1]["J"]))
        t_pol = time.perf_counter() - t0
        for tau in budgets:
            if tau > 0:
                r = repair(x, ctx, BURKE_CURTOIS, w0=args.w0, budget=tau, kappa=empty_kappa,
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
            # baselines with the same total response time as ours, T_infer = T_policy + T_repair
            # (Eq. (42)), building their model included, so that the comparison is the one of (28)
            target = t_pol + extra["repair_seconds"]
            scratch = {"J_cp_scratch": "", "feasible_cp_scratch": "", "cp_scratch_status": "",
                       "cp_scratch_budget": "", "cp_scratch_seconds": "", "vs_cp_scratch": "",
                       "changes_cp_scratch": "", "J_cp_hint": "", "feasible_cp_hint": "", "cp_hint_status": "",
                       "cp_hint_seconds": "", "vs_cp_hint": "", "changes_cp_hint": ""}
            if tau > 0 and not args.no_cp_scratch:
                cs, ok, secs = cp_baseline(ctx, target)
                scratch.update({"J_cp_scratch": cs.J if ok else "", "feasible_cp_scratch": ok,
                                "cp_scratch_status": cs.status, "cp_scratch_budget": round(target, 4),
                                "cp_scratch_seconds": secs, "vs_cp_scratch": verdict(feas, J, ok, cs.J),
                                "changes_cp_scratch": changes(cs.roster, ctx) if ok else ""})
            if tau > 0 and reroster and not args.no_cp_hint:      # CP-SAT warm-started from x°
                cs, ok, secs = cp_baseline(ctx, target, hint=ctx.published)
                scratch.update({"J_cp_hint": cs.J if ok else "", "feasible_cp_hint": ok,
                                "cp_hint_status": cs.status, "cp_hint_seconds": secs,
                                "vs_cp_hint": verdict(feas, J, ok, cs.J),
                                "changes_cp_hint": changes(cs.roster, ctx) if ok else ""})
            roster_out = r.roster if tau > 0 else x
            rows.append({**scratch, "instance": name, "tau_cp": tau, "d_xi": ctx.d_xi,
                         "changes": changes(roster_out, ctx), "w0": args.w0, "anytime": not args.first_feasible,
                         "source": args.source, "empty_kappa": empty_kappa,
                         "J_cp": rec["J_cp"], "cp_status": rec["cp_status"], "cp_seconds": rec["cp_seconds"],
                         "J_policy": ev["J"], "feasible_policy": ev["feasible"],
                         "policy_ms": round(1000 * t_pol, 2), "J": J, "feasible": feas,
                         "gap_to_cp": J - rec["J_cp"], "violations": json.dumps(viol),
                         "n_vars": n * m * k, "infer_seconds": round(t_pol + extra["repair_seconds"], 4), **extra})
        def cell(r):
            out = f"   tau={r['tau_cp']:g}: {r['J']:6.0f}{'' if r['feasible'] else '*'}"
            if r["cp_scratch_status"]:
                out += " | cp " + (f"{r['J_cp_scratch']:.0f}" if r["feasible_cp_scratch"] else "-")
            if r["cp_hint_status"]:
                out += " | hint " + (f"{r['J_cp_hint']:.0f}" if r["feasible_cp_hint"] else "-")
            return out
        print(f"  {name:22s} CP-SAT {rec['J_cp']:6.0f}   policy {ev['J']:6.0f} ({'feasible' if ev['feasible'] else 'infeasible'})"
              + "".join(cell(r) for r in rows if r["instance"] == name and r["tau_cp"] > 0), flush=True)

    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    summary = {}
    print(f"\n{'tau_cp':>8s} {'mean J':>8s} {'feasible':>9s} {'mean J - J_cp':>14s} {'mean T_repair':>14s} "
          f"{'max T_repair':>13s} {'V^ter emp.':>11s} {'|V_w0|/vars':>12s}   "
          f"(CP-SAT mean J {np.mean([r['J_cp'] for r in pools['eval']]):.1f})")
    for tau in budgets:
        sel = [r for r in rows if r["tau_cp"] == tau]
        s = {"mean_J": float(np.mean([r["J"] for r in sel])),
             "feasible": float(np.mean([bool(r["feasible"]) for r in sel])),
             "mean_gap_to_cp": float(np.mean([r["gap_to_cp"] for r in sel])),
             "mean_repair_seconds": float(np.mean([r["repair_seconds"] for r in sel])),
             "max_repair_seconds": float(np.max([r["repair_seconds"] for r in sel])),
             "mean_infer_seconds": float(np.mean([r["infer_seconds"] for r in sel]))}
        reps = [r for r in sel if r["n_free"] not in ("", "[]")]
        s["mean_ter_employees"] = float(np.mean([r["n_ter_employees"] for r in reps])) if reps else None
        s["mean_first_free_share"] = (float(np.mean([json.loads(r["n_free"])[0] / r["n_vars"] for r in reps]))
                                      if reps else None)
        cps = [r for r in sel if r["cp_scratch_status"]]
        if cps:
            solved = [r for r in cps if r["feasible_cp_scratch"]]
            s["cp_scratch_feasible"] = len(solved) / len(cps)
            s["cp_scratch_mean_J_solved"] = float(np.mean([r["J_cp_scratch"] for r in solved])) if solved else None
            s["ours_mean_J_same_instances"] = float(np.mean([r["J"] for r in solved])) if solved else None
            s["wins"], s["ties"], s["losses"] = (sum(r["vs_cp_scratch"] == x for r in cps) for x in ("win", "tie", "loss"))
        hs = [r for r in sel if r["cp_hint_status"]]
        if hs:
            solved = [r for r in hs if r["feasible_cp_hint"]]
            s["cp_hint_feasible"] = len(solved) / len(hs)
            s["cp_hint_mean_J_solved"] = float(np.mean([r["J_cp_hint"] for r in solved])) if solved else None
            s["ours_mean_J_hint_instances"] = float(np.mean([r["J"] for r in solved])) if solved else None
            s["hint_wins"], s["hint_ties"], s["hint_losses"] = (sum(r["vs_cp_hint"] == x for r in hs)
                                                               for x in ("win", "tie", "loss"))
        s["mean_changes"] = float(np.mean([r["changes"] for r in sel]))
        summary[str(tau)] = s
        ter = "-" if s["mean_ter_employees"] is None else f"{s['mean_ter_employees']:.1f}"
        share = "-" if s["mean_first_free_share"] is None else f"{s['mean_first_free_share']:.0%}"
        print(f"{tau:8g} {s['mean_J']:8.1f} {s['feasible']:9.0%} {s['mean_gap_to_cp']:14.1f} "
              f"{s['mean_repair_seconds']:13.3f}s {s['max_repair_seconds']:12.3f}s {ter:>11s} {share:>12s}")
    if any(r["cp_scratch_status"] for r in rows):
        print(f"\nfair baseline, CP-SAT from scratch with the same total time T_infer (policy + repair) and {args.workers} workers:")
        print(f"{'tau_cp':>8s} {'T_infer':>8s} {'CP feasible':>12s} {'CP mean J':>10s} {'ours, same inst.':>17s} {'win/tie/loss':>13s}   (CP mean J over the instances it solved)")
        for tau in budgets:
            s = summary[str(tau)]
            if "wins" not in s:
                continue
            cj = "-" if s["cp_scratch_mean_J_solved"] is None else f"{s['cp_scratch_mean_J_solved']:.1f}"
            oj = "-" if s["ours_mean_J_same_instances"] is None else f"{s['ours_mean_J_same_instances']:.1f}"
            print(f"{tau:8g} {s['mean_infer_seconds']:7.3f}s {s['cp_scratch_feasible']:12.0%} {cj:>10s} {oj:>17s} "
                  f"{s['wins']:>5d}/{s['ties']}/{s['losses']}")
    if any(r["cp_hint_status"] for r in rows):
        print(f"\nre-rostering baseline, CP-SAT hinted with x° (warm start), same total time and {args.workers} workers:")
        print(f"{'tau_cp':>8s} {'T_infer':>8s} {'CP feasible':>12s} {'CP mean J':>10s} {'ours, same inst.':>17s} "
              f"{'win/tie/loss':>13s} {'our changes':>12s}")
        for tau in budgets:
            s = summary[str(tau)]
            if "hint_wins" not in s:
                continue
            cj = "-" if s["cp_hint_mean_J_solved"] is None else f"{s['cp_hint_mean_J_solved']:.1f}"
            oj = "-" if s["ours_mean_J_hint_instances"] is None else f"{s['ours_mean_J_hint_instances']:.1f}"
            print(f"{tau:8g} {s['mean_infer_seconds']:7.3f}s {s['cp_hint_feasible']:12.0%} {cj:>10s} {oj:>17s} "
                  f"{s['hint_wins']:>5d}/{s['hint_ties']}/{s['hint_losses']} {s['mean_changes']:12.1f}")
    (out / "run.json").write_text(json.dumps({
        "training_run": args.name, "steps": model.num_timesteps, "args": {k_: str(v) if isinstance(v, Path) else v
                                                                        for k_, v in vars(args).items()},
        "machine": versions(), "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"), "summary": summary},
        indent=2) + "\n")
    print(f"\nwritten {out}/results.csv (* = infeasible)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
