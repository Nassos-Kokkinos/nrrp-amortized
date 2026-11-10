#!/usr/bin/env python3
"""Learned LNS against its baselines, with the same total response time on every context.

  python scripts/eval_lns.py --name bc8-lns --steps-list 2 4 8 16

For every held-out context of runs/<name>/pools.json and every K in --steps-list:

* learned LNS: K steps of the policy of runs/<name>/best.zip (or latest.zip with
  --checkpoint latest), each releasing the q employees with the highest log-odds;
  its wall-clock time T, the policy's decisions included, is the budget of the rest;
* random LNS: K steps of --step-time, each releasing as many employees as the
  learned policy did on average on that context, chosen uniformly at random
  (does the policy add anything beyond LNS itself?);
* repair of x°: Algorithm 1 on x° without the absent employees' assignments, budget T;
* CP-SAT on (20) from scratch, and hinted with x°, total time T (model building included).

All CP-SAT calls use --workers workers. Per context and K: win / tie / loss of the
learned LNS against each baseline (lower J wins; a feasible roster beats an
infeasible one).

results/lns/<name>[-latest]/  results.csv (one row per context, K and method), run.json (summary)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import pool_contexts, versions  # noqa: E402

METHODS = ["learned_lns", "random_lns", "repair_x0", "cp_scratch", "cp_hint"]
FIELDS = ["context", "d_xi", "J_cp_ref", "K", "method", "J", "feasible", "changes", "seconds", "budget", "released"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="learned-LNS run: runs/<name>/")
    ap.add_argument("--steps-list", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--step-time", type=float, default=None, help="CP-SAT seconds per LNS step (default: as trained)")
    ap.add_argument("--workers", type=int, default=8, help="CP-SAT workers, for every method")
    ap.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results/lns"))
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO

    from nrrp.cpsat import RosterModel
    from nrrp.disruption import changes, without_absences
    from nrrp.lns_env import LNSEnv, random_q, run_episode, top_q
    from nrrp.profile import BURKE_CURTOIS
    from nrrp.repair import repair

    torch.set_num_threads(1)
    run = args.runs / args.name
    pools = json.loads((run / "pools.json").read_text())
    config = json.loads((run / "config.json").read_text())
    step_time = args.step_time if args.step_time is not None else float(config["env"]["step_time"])
    ctxs = pool_contexts(pools)
    ckpt = run / f"{args.checkpoint}.zip"
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found")
    model = PPO.load(ckpt, device="cpu")
    rng = np.random.default_rng(args.seed)

    def learned(obs, env):
        with torch.no_grad():
            logits = model.policy.mlp_extractor.forward_actor(torch.as_tensor(obs[None])).numpy()[0]
        return top_q(logits)

    def cp(ctx, budget, hint=None):
        t1 = time.perf_counter()
        m = RosterModel(ctx, BURKE_CURTOIS)
        left = max(budget - (time.perf_counter() - t1), 1e-3)
        r = m.solve(time_limit=left, workers=args.workers, seed=args.seed, hint=hint)
        ok = bool(r.has_solution and r.feasible)
        return (r.J if ok else float("nan")), ok, (changes(r.roster, ctx) if ok else ""), time.perf_counter() - t1

    out = args.results / (args.name + ("" if args.checkpoint == "best" else "-latest"))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{len(ctxs)} held-out contexts of {ctxs[0].instance.name}; K {args.steps_list}, {step_time:g} s per step, "
          f"{args.workers} CP-SAT workers; checkpoint {ckpt.name}", flush=True)
    for K in args.steps_list:
        env = LNSEnv(ctxs, steps=K, step_time=step_time, workers=args.workers, seed=args.seed)
        for j, (ctx, rec) in enumerate(zip(ctxs, pools["eval"])):
            base = {"context": ctx.name, "d_xi": ctx.d_xi, "J_cp_ref": rec["J_cp"], "K": K}
            x, info = run_episode(env, j, learned)
            T = info["seconds"]
            q = max(1, int(round(np.mean([f for f in info["freed_per_step"]]))))
            rows.append({**base, "method": "learned_lns", "J": info["J"], "feasible": info["feasible"],
                         "changes": info["changes"], "seconds": round(T, 4), "budget": round(T, 4),
                         "released": float(np.mean(info["freed_per_step"]))})
            x, info = run_episode(env, j, lambda obs, e: random_q(e.n_emp, q, rng))
            rows.append({**base, "method": "random_lns", "J": info["J"], "feasible": info["feasible"],
                         "changes": info["changes"], "seconds": round(info["seconds"], 4), "budget": round(T, 4),
                         "released": float(np.mean(info["freed_per_step"]))})
            t1 = time.perf_counter()
            r = repair(without_absences(ctx), ctx, BURKE_CURTOIS, budget=T, workers=args.workers, seed=args.seed)
            rows.append({**base, "method": "repair_x0", "J": r.J, "feasible": r.feasible,
                         "changes": changes(r.roster, ctx), "seconds": round(time.perf_counter() - t1, 4),
                         "budget": round(T, 4), "released": ""})
            for method, hint in (("cp_scratch", None), ("cp_hint", ctx.published)):
                J, ok, chg, secs = cp(ctx, T, hint)
                rows.append({**base, "method": method, "J": J, "feasible": ok, "changes": chg,
                             "seconds": round(secs, 4), "budget": round(T, 4), "released": ""})
            mine = {r_["method"]: r_ for r_ in rows[-5:]}
            print(f"  K={K:2d} {ctx.name:20s} ref {rec['J_cp']:7.0f}  T {T:5.2f}s  "
                  + "  ".join(f"{m_} {mine[m_]['J']:7.0f}{'' if mine[m_]['feasible'] else '*'}" for m_ in METHODS),
                  flush=True)

    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    def compare(a, b):                 # 0 win, 1 tie, 2 loss of a against b: feasible first, then lower J
        if a["feasible"] != b["feasible"]:
            return 0 if a["feasible"] else 2
        if not a["feasible"] or abs(a["J"] - b["J"]) <= 1e-9:
            return 1
        return 0 if a["J"] < b["J"] else 2

    summary = {}
    print(f"\n{'K':>3s} {'T_infer':>8s} " + " ".join(f"{m_:>12s}" for m_ in METHODS)
          + "   learned LNS win/tie/loss vs " + ", ".join(METHODS[1:]))
    for K in args.steps_list:
        sel = [r_ for r_ in rows if r_["K"] == K]
        by = {m_: [r_ for r_ in sel if r_["method"] == m_] for m_ in METHODS}
        s = {"T_infer": float(np.mean([r_["seconds"] for r_ in by["learned_lns"]]))}
        for m_ in METHODS:
            feas = [r_ for r_ in by[m_] if r_["feasible"]]
            s[m_] = {"mean_J_feasible": float(np.mean([r_["J"] for r_ in feas])) if feas else None,
                     "feasible": len(feas) / len(by[m_])}
        for m_ in METHODS[1:]:
            wtl = [0, 0, 0]
            for a, b in zip(by["learned_lns"], by[m_]):
                wtl[compare(a, b)] += 1
            s[f"vs_{m_}"] = wtl
        summary[str(K)] = s
        cells = " ".join(f"{(s[m_]['mean_J_feasible'] or float('nan')):9.1f}{'' if s[m_]['feasible'] == 1 else '*'}  "
                         for m_ in METHODS)
        print(f"{K:3d} {s['T_infer']:7.2f}s {cells}  " + ", ".join("/".join(map(str, s[f"vs_{m_}"])) for m_ in METHODS[1:]))
    (out / "run.json").write_text(json.dumps({"training_run": args.name, "checkpoint": ckpt.name,
                                              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                              "machine": versions(), "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                              "summary": summary}, indent=2) + "\n")
    print(f"\n(* = not every context feasible; mean J over the feasible ones)\nwritten {out}/results.csv")


if __name__ == "__main__":
    main()
