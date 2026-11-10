#!/usr/bin/env python3
"""Train the learned LNS of nrrp/lns_env.py with PPO, on one Burke & Curtois instance.

  python scripts/train_lns.py --name bc8-lns --pools-from bc8-big-p5 --timesteps 50000

The context distribution is the P of a re-rostering run: --pools-from RUN takes
its instance, x°, P and held-out contexts (with their CP-SAT reference), so that
results compare with that run; without it, --reroster INSTANCE and the P options
build a new held-out pool as train.py --reroster does. Training draws a fresh
disruption every episode, in --n-envs parallel environments (one CP-SAT worker
each by default). An episode is --steps CP-SAT calls of --step-time seconds.

Every --eval-every steps the deterministic policy (the q employees with the
highest log-odds, q = the expected number released) is run on the held-out
contexts and on --val-pool validation contexts; the checkpoint best on
validation (fewest infeasible, then lowest mean J) is kept as best.zip.

runs/<name>/          config.json, pools.json, latest.zip, best.zip, best.json, progress.csv, tb/
results/train_lns/<name>/  results.csv (best checkpoint, per held-out context), progress.csv, run.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import best_published, disruption_model, feasible_contexts, pool_contexts, versions  # noqa: E402

PROGRESS_FIELDS = ["steps", "wall_seconds", "J_start", "J_det", "feasible_det", "J_cp", "J_val", "feasible_val",
                   "best", "released_per_step", "days_per_step"]
RESULT_FIELDS = ["context", "d_xi", "J_cp", "cp_status", "J_start", "feasible_start", "J", "feasible", "changes",
                 "seconds", "released_per_step", "days_per_step", "statuses"]


def make_env(spec, env_kwargs, seed):
    """A factory for SubprocVecEnv: builds P inside the worker process."""
    def _thunk():
        from nrrp.lns_env import LNSEnv
        env = LNSEnv(disruption_model(spec), **env_kwargs)
        env.reset(seed=seed)
        return env
    return _thunk


def policy_chooser(model):
    import torch

    from nrrp.lns_env import decide

    def choose(obs, env):
        with torch.no_grad():
            logits = model.policy.mlp_extractor.forward_actor(torch.as_tensor(obs[None])).numpy()[0]
        return decide(logits, env)
    return choose


def evaluate(model, env, n):
    """The deterministic policy on the n fixed contexts of ``env``."""
    from nrrp.lns_env import run_episode
    choose = policy_chooser(model)
    rows = []
    for j in range(n):
        x, info = run_episode(env, j, choose)
        rows.append({"J_start": info["J0"], "feasible_start": info["feasible0"], "J": info["J"],
                     "feasible": info["feasible"], "changes": info["changes"], "seconds": round(info["seconds"], 4),
                     "released_per_step": float(np.mean(info["freed_per_step"])),
                     "days_per_step": float(np.mean(info["days_per_step"])),
                     "statuses": json.dumps(info["statuses"])})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pools-from", metavar="RUN", help="re-rostering run whose instance, x°, P and held-out pool to use")
    ap.add_argument("--reroster", type=Path, metavar="INSTANCE", help="without --pools-from: the instance file")
    ap.add_argument("--published", type=Path)
    ap.add_argument("--n-absent", type=int, nargs=2, default=[3, 5])
    ap.add_argument("--absence-days", type=int, nargs=2, default=[3, 7])
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--eval-pool", type=int, default=16)
    ap.add_argument("--eval-seed0", type=int, default=100000)
    ap.add_argument("--cp-time-limit", type=float, default=60.0, help="CP-SAT reference per held-out context")
    ap.add_argument("--val-pool", type=int, default=16)
    ap.add_argument("--val-seed0", type=int, default=200000)
    ap.add_argument("--steps", type=int, default=8, help="LNS steps (CP-SAT calls) per episode, K")
    ap.add_argument("--step-time", type=float, default=0.05, help="CP-SAT seconds per step, model building included")
    ap.add_argument("--cp-workers", type=int, default=1, help="CP-SAT workers per environment")
    ap.add_argument("--release", choices=["employees", "both"], default="employees",
                    help="what the policy releases for CP-SAT: employees (all their days from d_xi), or both "
                         "employees and days (all employees on a day)")
    ap.add_argument("--n-envs", type=int, default=8, help="parallel training environments")
    ap.add_argument("--timesteps", type=int, default=50_000, help="total LNS steps (cumulative on --resume)")
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--n-steps", type=int, default=64, help="rollout length per environment per update")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--init-bias", type=float, default=-2.0, help="initial log-odds of releasing an employee")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results/train_lns"))
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from nrrp.instance import Context
    from nrrp.lns_env import LNSEnv
    from nrrp.objective import evaluate as evaluate_roster
    from nrrp.policy import LNSPolicy
    from nrrp.profile import BURKE_CURTOIS

    torch.set_num_threads(1)
    run = args.runs / args.name
    run.mkdir(parents=True, exist_ok=True)
    pools_file = run / "pools.json"
    if args.pools_from and not pools_file.exists():
        src = args.runs / args.pools_from / "pools.json"
        if not src.exists():
            sys.exit(f"{src} not found")
        pools_file.write_text(src.read_text())
        print(f"pools copied from {src}")
    if pools_file.exists():
        pools = json.loads(pools_file.read_text())
        if pools.get("regime") != "reroster":
            sys.exit(f"{pools_file} is not a re-rostering pool")
    else:
        if args.resume or not args.reroster:
            sys.exit("give --pools-from RUN, or --reroster INSTANCE to build a new held-out pool")
        spec = {"instance": str(args.reroster), "published": str(args.published or best_published(args.reroster)),
                "n_absent": list(args.n_absent), "days": list(args.absence_days), "mu": args.mu, "first_day": 1}
        P = disruption_model(spec)
        ev0 = evaluate_roster(P.published, Context(P.instance), BURKE_CURTOIS)
        if not ev0["feasible"]:
            sys.exit(f"the published roster violates hard constraints: {ev0['violations']}")
        print("building the held-out pool of contexts with CP-SAT ...", flush=True)
        try:
            _, records, rejected = feasible_contexts(P, args.eval_seed0, args.eval_pool, args.cp_time_limit)
        except KeyboardInterrupt:
            sys.exit("interrupted while building pools: nothing written (the same command starts over)")
        pools = {"regime": "reroster", "reroster": spec, "size": [P.instance.n_emp, P.instance.n_days,
                 P.instance.n_shifts], "train": [], "eval": records, "rejected": rejected}
        pools_file.write_text(json.dumps(pools, indent=1) + "\n")
    spec = pools["reroster"]
    P = disruption_model(spec)
    inst, n = P.instance, P.instance.n_emp
    eval_ctxs = pool_contexts(pools)
    val_ctxs = [P.context(args.val_seed0 + j) for j in range(args.val_pool)]
    env_kwargs = dict(steps=args.steps, step_time=args.step_time, workers=args.cp_workers, release=args.release)
    print(f"learned LNS on {inst.name} ({n}x{inst.n_days}x{inst.n_shifts}); P {P.describe()}; "
          f"K = {args.steps} steps of {args.step_time:g} s; {args.n_envs} environments", flush=True)

    config = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "machine": versions(), "disruption": P.describe(), "env": env_kwargs,
              "started": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    (run / ("config_resume.json" if args.resume else "config.json")).write_text(json.dumps(config, indent=2) + "\n")

    venv = SubprocVecEnv([make_env(spec, env_kwargs, args.seed * 1000 + j) for j in range(args.n_envs)])
    eval_env = LNSEnv(eval_ctxs, **env_kwargs)
    val_env = LNSEnv(val_ctxs, **env_kwargs) if val_ctxs else None
    latest, best_zip, best_json = run / "latest.zip", run / "best.zip", run / "best.json"
    if args.resume:
        model = PPO.load(latest, env=venv, device="cpu")
        print(f"loaded {latest} at {model.num_timesteps} steps")
    else:
        model = PPO(LNSPolicy, venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
                    policy_kwargs=dict(n_emp=n, n_days=inst.n_days if args.release == "both" else 0,
                                       hidden=tuple(args.hidden), init_bias=args.init_bias),
                    tensorboard_log=str(run / "tb"), seed=args.seed, verbose=0, device="cpu")
    progress = run / "progress.csv"
    J_cp = float(np.mean([r["J_cp"] for r in pools["eval"]]))

    def log_eval(steps, wall):
        rows = evaluate(model, eval_env, len(eval_ctxs))
        s = {"J_start": float(np.mean([r["J_start"] for r in rows])), "J_det": float(np.mean([r["J"] for r in rows])),
             "feasible_det": float(np.mean([r["feasible"] for r in rows])),
             "released": float(np.mean([r["released_per_step"] for r in rows])),
             "days": float(np.mean([r["days_per_step"] for r in rows]))}
        better, J_val, f_val = False, "", ""
        if val_env is not None:
            vals = evaluate(model, val_env, len(val_ctxs))
            score = [float(np.mean([not r["feasible"] for r in vals])), float(np.mean([r["J"] for r in vals]))]
            best = json.loads(best_json.read_text()) if best_json.exists() else None
            better = best is None or tuple(score) < tuple(best["score"])
            if better:
                model.save(best_zip)
                best_json.write_text(json.dumps({"steps": steps, "score": score}) + "\n")
            J_val, f_val = score[1], 1 - score[0]
        new = not progress.exists()
        with progress.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=PROGRESS_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"steps": steps, "wall_seconds": round(wall, 1), "J_start": s["J_start"], "J_det": s["J_det"],
                        "feasible_det": s["feasible_det"], "J_cp": J_cp, "J_val": J_val, "feasible_val": f_val,
                        "best": better, "released_per_step": s["released"], "days_per_step": s["days"]})
        print(f"  {steps:>8,d} steps {wall:7.0f} s   held-out J: start {s['J_start']:8.1f}  learned LNS {s['J_det']:8.1f} "
              f"(feasible {s['feasible_det']:.0%}, {s['released']:.1f} employees"
              + (f" + {s['days']:.1f} days" if args.release == "both" else "") + f" released/step)  CP-SAT {J_cp:7.1f}"
              + (f"   validation {J_val:8.1f}{'  <- best, saved' if better else ''}" if val_env is not None else ""),
              flush=True)

    class Periodic(BaseCallback):
        def __init__(self, t_start):
            super().__init__()
            self.t_start, self.next_eval = t_start, 0

        def _on_training_start(self):
            self.next_eval = (self.num_timesteps // args.eval_every + 1) * args.eval_every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self.model.save(latest)
                log_eval(self.num_timesteps, time.time() - self.t_start)
                self.next_eval += args.eval_every
            return True

    remaining = args.timesteps - model.num_timesteps
    t0 = time.time()
    if model.num_timesteps == 0:
        log_eval(0, 0.0)
    interrupted = False
    if remaining > 0:
        print(f"training {remaining:,d} LNS steps (Ctrl+C saves {latest} and stops)", flush=True)
        try:
            model.learn(total_timesteps=remaining, callback=Periodic(t0), reset_num_timesteps=False,
                        tb_log_name="ppo", progress_bar=False)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            model.save(latest)
            print(f"saved {latest} at {model.num_timesteps:,d} steps ({time.time() - t0:.0f} s)", flush=True)
    venv.close()
    if interrupted:
        print("interrupted: continue with --resume")
        return

    # final: the best checkpoint on validation, on the held-out pool
    final = PPO.load(best_zip, device="cpu") if best_zip.exists() else model
    rows = evaluate(final, eval_env, len(eval_ctxs))
    out = args.results / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r, rec, ctx in zip(rows, pools["eval"], eval_ctxs):
            w.writerow({**r, "context": ctx.name, "d_xi": ctx.d_xi, "J_cp": rec["J_cp"], "cp_status": rec["cp_status"]})
    if progress.exists():
        (out / "progress.csv").write_text(progress.read_text())
    best = json.loads(best_json.read_text()) if best_json.exists() else {}
    summary = {"J_start": float(np.mean([r["J_start"] for r in rows])), "J": float(np.mean([r["J"] for r in rows])),
               "feasible": float(np.mean([r["feasible"] for r in rows])), "J_cp": J_cp,
               "seconds": float(np.mean([r["seconds"] for r in rows]))}
    (out / "run.json").write_text(json.dumps({**config, "steps": model.num_timesteps, "best_steps": best.get("steps"),
                                              "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                              "summary": summary}, indent=2) + "\n")
    print(f"\nbest checkpoint ({best.get('steps')} steps) on {len(rows)} held-out contexts: start J {summary['J_start']:.1f}, "
          f"learned LNS {summary['J']:.1f} (feasible {summary['feasible']:.0%}, {summary['seconds']:.2f} s), "
          f"CP-SAT {J_cp:.1f}\n  written {out}/results.csv; compare with: python scripts/eval_lns.py --name {args.name}")


if __name__ == "__main__":
    main()
