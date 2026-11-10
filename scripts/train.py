#!/usr/bin/env python3
"""Train the policy of Eq. (39) with MaskablePPO on a pool of instances, and
evaluate it against CP-SAT on held-out instances.

  python scripts/train.py --name syn5x7 --size 5 7 3 --timesteps 200000
  python scripts/train.py --name syn5x7 --resume --timesteps 400000      # continue
  python scripts/train.py --name syn5x7 --evaluate-only                   # final evaluation only

Construction regime (d_xi = 0, no disruption): the first rung of the ladder
5x7 -> 25x7 -> Burke & Curtois. Pools are synthetic instances of one size;
only instances for which CP-SAT finds a feasible roster are kept (about one
in four to one in three synthetic instances has none). The held-out pool uses
other seeds and is never trained on; for each of its instances the CP-SAT
optimum (or best found within --cp-time-limit) is the reference.

Re-rostering regime (--reroster INSTANCE, Remark 5): one Burke & Curtois
instance, x° fixed (by default its best published solution, --published to
choose another), and a disruption xi ~ P drawn afresh at every training
episode (nrrp/disruption.py: --n-absent, --absence-days, --mu), so that the
policy never sees the same disruption twice. The held-out pool is
--eval-pool contexts from seed --eval-seed0 on, kept if CP-SAT finds a feasible
roster of (20) within --cp-time-limit (the CP-SAT result is the reference);
the rejected ones are recorded in pools.json. --size is taken from the instance.

  python scripts/train.py --name bc5-rr --reroster data/raw/Instance5.txt --features extended \
      --separate-critic --ent-coef 0.001 --timesteps 2000000 --eval-every 100000 --save-every 100000 \
      --cp-time-limit 60

runs/<name>/ (git-ignored)
  config.json          every parameter, machine and versions
  pools.json           seeds of the training and held-out pools (re-rostering: of the held-out contexts, and
                       P), CP-SAT reference per held-out instance or context
  latest.zip           the model, saved every --save-every steps and on Ctrl+C / kill -INT
  best.zip, best.json  with --val-pool: the checkpoint best on the validation contexts, and its score
  validation.csv       with --val-pool: the validation score at every evaluation
  progress.csv         one row per evaluation: steps, wall time, J (deterministic and stochastic), feasibility, gap to CP-SAT
  progress_instances.csv  the same per held-out instance (for the per-instance grid of learning curves)
  tb/                  TensorBoard logs (tensorboard --logdir runs)
results/train/<name>/ (committed unless the name starts with provisional-)
  results.csv          final evaluation, one row per held-out instance
  run.json             machine, versions, parameters
  figures/             learning curve and final gaps (scripts/plot_results.py train)

Several runs of one configuration with different --seed values, and runs of
other configurations with another --label, give the grid of per-instance
learning curves (scripts/plot_results.py grid), one panel per held-out
instance and one thin line per seed, as in Fig. 6 of Schulman et al. (2017).
The held-out pool depends only on --size and --eval-seed0, so it is the same
across seeds and labels.

Training reward: the training profile (H^ter penalised, weights --w-min-minutes
and --w-min-consecutive, hyperparameters reported in Section 5), scaled by
--reward-scale, c^con not charged (Remark 7), gamma = 1 so that the return is
the objective itself (Prop. 2). Evaluation: J and feasibility under the
published profile, with c^con.

Options for scaling up (30/10/2026, after the 25x7 run barely moved from the
uniform policy: tiny KL per update, entropy staying high): --ent-coef, --gamma
and --gae-lambda (gamma < 1 shortens credit assignment in training only; the
evaluation is still J), --lr-schedule linear, --target-kl, --vf-coef,
--separate-critic (critic with its own encoder), and --init-from RUN, which
starts from a policy trained on another size (curriculum 5x7 -> 25x7: the
network has no parameter that depends on the number of employees).
--pools-from RUN reuses another run's pools so that runs compare on the same
instances. --features extended adds to Table 2 the quantities the H^ter
constraints depend on (RosterEnv.EXTENDED): the deficit of a work run below
C^min_i, the urgency of the minutes shortfall, the immediate cost of each
assignment, the shift length and the global tightness.

  python scripts/train.py --name 25x7-ent001-init5x7 --size 25 7 3 --pools-from mac-25x7-2M \
      --init-from mac-5x7-1M --ent-coef 0.001 --timesteps 2000000 --eval-every 100000 --save-every 100000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import signal
import sys
import time
from pathlib import Path

import numpy as np

from nrrp.bc_format import read_bc, read_roster, solution_name
from nrrp.disruption import DisruptionModel
from nrrp.generator import synthetic_instance
from nrrp.gym_env import GymRosterEnv
from nrrp.instance import Context
from nrrp.objective import evaluate as evaluate_roster
from nrrp.profile import BURKE_CURTOIS, burke_curtois_training

INSTANCE_FIELDS = ["steps", "instance", "J_cp", "J_det", "J_best_of_k", "feasible_det", "feasible_best_of_k"]
EVAL_FIELDS = ["steps", "wall_seconds", "J_det", "J_stoch", "J_cp", "gap_det", "gap_stoch",
               "feasible_det", "feasible_stoch", "episodes_per_second"]
RESULT_FIELDS = ["instance", "J_cp", "cp_status", "J_det", "feasible_det", "gap_det", "J_best_of_k",
                 "feasible_best_of_k", "gap_best_of_k", "k", "violations_det", "policy_ms_per_roster"]


def versions() -> dict:
    out = {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
           "python": platform.python_version(), "numpy": np.__version__}
    for mod in ("torch", "gymnasium", "stable_baselines3", "sb3_contrib", "ortools"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def feasible_pool(n, m, k, first_seed, size, cp_time_limit, prefix):
    """``size`` synthetic instances, from seed ``first_seed`` on, that CP-SAT can solve.
    Returns (instances, records) where each record holds the seed and the CP-SAT result."""
    from nrrp.cpsat import solve
    insts, records, seed = [], [], first_seed
    while len(insts) < size:
        inst = synthetic_instance(n, m, k, seed=seed, name=f"{prefix}{n}x{m}x{k}s{seed}")
        res = solve(Context(inst), BURKE_CURTOIS, time_limit=cp_time_limit, workers=4, seed=0)
        # CP-SAT catches Ctrl+C itself: it ends the current solve early and Python never sees the
        # signal. A solve that ends before its time limit without a proof (optimal or infeasible)
        # was interrupted, so pool building stops here.
        if res.status in ("UNKNOWN", "FEASIBLE") and res.wall_time < 0.9 * cp_time_limit:
            raise KeyboardInterrupt
        if res.has_solution and res.feasible:
            insts.append(inst)
            records.append({"seed": seed, "name": inst.name, "J_cp": res.J, "cp_status": res.status,
                            "cp_bound": res.bound, "cp_seconds": round(res.wall_time, 3)})
        seed += 1
        if seed - first_seed > 20 * size:
            raise RuntimeError(f"fewer than {size} feasible instances in {20 * size} seeds")
    return insts, records


def rebuild_pool(n, m, k, records):
    return [synthetic_instance(n, m, k, seed=r["seed"], name=r["name"]) for r in records]


def best_published(instance_file) -> Path:
    """The published solution with the lowest value, ``solutions/<Instance>.Solution.VALUE.roster``
    next to the instance file (Burke & Curtois)."""
    instance_file = Path(instance_file)
    found = [(solution_name(f)[1], f) for f in sorted((instance_file.parent / "solutions").glob("*.roster"))
             if solution_name(f) and solution_name(f)[0] == instance_file.stem]
    if not found:
        raise FileNotFoundError(f"no published solution for {instance_file.stem} in {instance_file.parent / 'solutions'}")
    return min(found)[1]


def disruption_model(spec) -> DisruptionModel:
    """The P of a re-rostering run, from the ``reroster`` entry of its pools.json."""
    inst = read_bc(spec["instance"])
    x0 = read_roster(spec["published"], inst)
    return DisruptionModel(inst, x0, n_absent=tuple(spec["n_absent"]), days=tuple(spec["days"]),
                           mu=float(spec["mu"]), first_day=int(spec.get("first_day", 1)))


def pool_contexts(pools, which="eval") -> list:
    """The contexts of a pool: construction (Context of each synthetic instance) or
    re-rostering (the context of each seed under the run's P)."""
    if pools.get("regime") == "reroster":
        P = disruption_model(pools["reroster"])
        return [P.context(r["seed"]) for r in pools[which]]
    n, m, k = pools["size"]
    return [Context(inst) for inst in rebuild_pool(n, m, k, pools[which])]


def feasible_contexts(P, first_seed, size, cp_time_limit):
    """``size`` re-rostering contexts, from seed ``first_seed`` on, for which CP-SAT finds a
    feasible roster within ``cp_time_limit``; the others are recorded as rejected (X(phi) empty,
    or nothing found in time). Returns (contexts, records, rejected)."""
    from nrrp.cpsat import solve
    ctxs, records, rejected, seed = [], [], [], first_seed
    while len(ctxs) < size:
        ctx = P.context(seed)
        res = solve(ctx, BURKE_CURTOIS, time_limit=cp_time_limit, workers=4, seed=0)
        if res.status in ("UNKNOWN", "FEASIBLE") and res.wall_time < 0.9 * cp_time_limit:
            raise KeyboardInterrupt                      # Ctrl+C, caught by CP-SAT (see feasible_pool)
        base = {"seed": seed, "name": ctx.name, "d_xi": ctx.d_xi, "absent_days": int(ctx.absent.sum()),
                "cp_status": res.status, "cp_seconds": round(res.wall_time, 3)}
        if res.has_solution and res.feasible:
            ctxs.append(ctx)
            records.append({**base, "J_cp": res.J, "cp_bound": res.bound})
        else:
            rejected.append(base)
        print(f"    {ctx.name:18s} d_xi {ctx.d_xi:2d}  absent days {base['absent_days']:2d}  CP-SAT {res.status:10s} "
              f"{'' if not (res.has_solution and res.feasible) else f'J {res.J:7.1f}'}  {res.wall_time:6.1f} s", flush=True)
        seed += 1
        if seed - first_seed > 20 * size:
            raise RuntimeError(f"fewer than {size} feasible contexts in {20 * size} seeds")
    return ctxs, records, rejected


def run_episode(env, model, idx, deterministic, rng=None):
    obs, _ = env.reset(seed=None if rng is None else int(rng.integers(2**31 - 1)), options={"instance": idx})
    done, info = False, {}
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=deterministic)
        obs, _, done, _, info = env.step(int(a))
    ev = info["evaluation"]
    return ev["J"], ev["feasible"], ev["violations"], env.roster


def evaluate(model, env, records, k_samples=8, seed=0):
    """Per held-out instance: deterministic rollout and the best of k stochastic rollouts
    (best = feasible first, then lowest J), under the published profile."""
    rng = np.random.default_rng(seed)
    rows = []
    for idx, rec in enumerate(records):
        t0 = time.perf_counter()
        J_det, f_det, v_det, _ = run_episode(env, model, idx, True)
        ms = 1000 * (time.perf_counter() - t0)
        samples = [run_episode(env, model, idx, False, rng)[:2] for _ in range(k_samples)]
        J_best, f_best = min(samples, key=lambda t: (not t[1], t[0]))
        J_cp = rec["J_cp"]
        gap = lambda j: (j - J_cp) / max(1.0, abs(J_cp))  # noqa: E731
        rows.append({"instance": rec["name"], "J_cp": J_cp, "cp_status": rec["cp_status"],
                     "J_det": J_det, "feasible_det": f_det, "gap_det": gap(J_det),
                     "J_best_of_k": J_best, "feasible_best_of_k": f_best, "gap_best_of_k": gap(J_best),
                     "k": k_samples, "violations_det": json.dumps(v_det),
                     "policy_ms_per_roster": round(ms, 2)})
    return rows


def summary(rows) -> dict:
    a = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    return {"J_det": a("J_det"), "J_stoch": a("J_best_of_k"), "J_cp": a("J_cp"), "gap_det": a("gap_det"),
            "gap_stoch": a("gap_best_of_k"), "feasible_det": a("feasible_det"),
            "feasible_stoch": a("feasible_best_of_k")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name: runs/<name>/ and results/train/<name>/")
    ap.add_argument("--size", type=int, nargs=3, default=[5, 7, 3], metavar=("N", "M", "K"),
                    help="employees, days, shift types of the synthetic pools (construction regime)")
    ap.add_argument("--reroster", type=Path, metavar="INSTANCE",
                    help="re-rostering regime on this Burke & Curtois instance file (e.g. data/raw/Instance5.txt): "
                         "x° fixed, a fresh disruption xi ~ P every episode")
    ap.add_argument("--published", type=Path, metavar="ROSTER",
                    help="x° for --reroster (default: the best published solution in solutions/ next to the instance)")
    ap.add_argument("--n-absent", type=int, nargs=2, default=[1, 2], metavar=("LO", "HI"),
                    help="P: number of absent employees, uniform on LO..HI")
    ap.add_argument("--absence-days", type=int, nargs=2, default=[1, 7], metavar=("LO", "HI"),
                    help="P: length of each absence in days, uniform on LO..HI, from d_xi")
    ap.add_argument("--mu", type=float, default=1.0, help="P: weight of the instability Delta(x, x°) in J")
    ap.add_argument("--published-prior", type=float, default=None, metavar="B0",
                    help="re-rostering: a learned prior b (initially B0) in the logits towards the employee x° "
                         "had on the slot, or the empty action if none is allowed (nrrp/policy.py), so that the "
                         "first policy reproduces x° and training learns where to deviate")
    ap.add_argument("--train-pool", type=int, default=64, help="training instances")
    ap.add_argument("--eval-pool", type=int, default=16, help="held-out instances")
    ap.add_argument("--train-seed0", type=int, default=0, help="first seed tried for the training pool")
    ap.add_argument("--eval-seed0", type=int, default=100000, help="first seed tried for the held-out pool")
    ap.add_argument("--val-pool", type=int, default=0,
                    help="re-rostering: validation contexts from seed --val-seed0 on (no CP-SAT needed, never in the "
                         "held-out pool); at every evaluation the deterministic policy is scored on them and the best "
                         "checkpoint (fewest infeasible, then lowest mean J) is kept as runs/<name>/best.zip")
    ap.add_argument("--val-seed0", type=int, default=200000, help="first seed of the validation contexts")
    ap.add_argument("--timesteps", type=int, default=200_000, help="total environment steps (cumulative on --resume)")
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--save-every", type=int, default=20_000)
    ap.add_argument("--k-samples", type=int, default=8, help="stochastic rollouts per held-out instance")
    ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--reward-scale", type=float, default=0.01)
    ap.add_argument("--w-min-minutes", type=float, default=0.2, help="shaping weight per minute short of H^min")
    ap.add_argument("--w-min-consecutive", type=float, default=100.0, help="shaping weight per day short of C^min")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="PPO discount in training (1 = the undiscounted return c^con - J of Remark 7); "
                         "evaluation is always the true J")
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant",
                    help="linear: lr decays to 0 over --timesteps")
    ap.add_argument("--target-kl", type=float, default=None, help="stop an update's epochs early past this KL")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--features", choices=["table2", "extended"], default="table2",
                    help="state encoding: Table 2, or Table 2 plus the H^ter features of RosterEnv.EXTENDED")
    ap.add_argument("--separate-critic", action="store_true",
                    help="give the critic its own encoder instead of sharing the actor's")
    ap.add_argument("--pools-from", metavar="RUN",
                    help="copy runs/<RUN>/pools.json instead of building the pools (same instances, no CP-SAT)")
    ap.add_argument("--init-from", metavar="RUN",
                    help="start from the policy of runs/<RUN>/latest.zip (any number of employees, same "
                         "shift types and --hidden); PPO's step count starts at 0")
    ap.add_argument("--seed", type=int, default=0, help="seed of PPO and of the stochastic evaluation")
    ap.add_argument("--label", default="PPO + mask", help="configuration label, groups seeds in the grid plot")
    ap.add_argument("--cp-time-limit", type=float, default=10.0, help="CP-SAT seconds per pool instance")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name>/latest.zip")
    ap.add_argument("--evaluate-only", action="store_true", help="evaluate runs/<name>/latest.zip and plot")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results/train"))
    args = ap.parse_args()

    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback

    from nrrp.policy import EmployeePolicy

    run = args.runs / args.name
    run.mkdir(parents=True, exist_ok=True)
    n, m, k = args.size
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    P = None                                   # the disruption model, re-rostering regime only

    # pools: built once, then read back on --resume / --evaluate-only so that they never change
    pools_file = run / "pools.json"
    if args.pools_from and not pools_file.exists():
        src = args.runs / args.pools_from / "pools.json"
        if not src.exists():
            sys.exit(f"{src} not found")
        pools_file.write_text(src.read_text())
        print(f"pools copied from {src}")
    if pools_file.exists():
        pools = json.loads(pools_file.read_text())
        if pools.get("regime") == "reroster":
            P = disruption_model(pools["reroster"])
            n, m, k = P.instance.n_emp, P.instance.n_days, P.instance.n_shifts
            if args.reroster and Path(pools["reroster"]["instance"]).stem != args.reroster.stem:
                sys.exit(f"{pools_file} is for {pools['reroster']['instance']}, not {args.reroster}")
            eval_ctxs = pool_contexts(pools)
        else:
            if args.reroster:
                sys.exit(f"{pools_file} is a construction pool; --reroster needs a run of its own")
            if tuple(pools["size"]) != (n, m, k):
                sys.exit(f"{pools_file} is for size {pools['size']}, not {[n, m, k]}")
            train_insts = rebuild_pool(n, m, k, pools["train"])
            eval_ctxs = [Context(inst) for inst in rebuild_pool(n, m, k, pools["eval"])]
        print(f"pools read from {pools_file}")
    elif args.reroster:
        if args.resume or args.evaluate_only:
            sys.exit(f"{pools_file} not found: nothing to resume")
        published = args.published or best_published(args.reroster)
        spec = {"instance": str(args.reroster), "published": str(published), "n_absent": list(args.n_absent),
                "days": list(args.absence_days), "mu": args.mu, "first_day": 1}
        P = disruption_model(spec)
        ev0 = evaluate_roster(P.published, Context(P.instance), BURKE_CURTOIS)
        if not ev0["feasible"]:
            sys.exit(f"the published roster {published} violates hard constraints: {ev0['violations']}")
        n, m, k = P.instance.n_emp, P.instance.n_days, P.instance.n_shifts
        t0 = time.time()
        print(f"re-rostering {P.instance.name} ({n}x{m}x{k}), x° = {published} (J {ev0['J']:g}); "
              f"P: {args.n_absent[0]}-{args.n_absent[1]} absent for {args.absence_days[0]}-{args.absence_days[1]} days, "
              f"mu {args.mu:g}", flush=True)
        print("building the held-out pool of contexts with CP-SAT (training draws fresh ones) ...", flush=True)
        try:
            eval_ctxs, eval_rec, rejected = feasible_contexts(P, args.eval_seed0, args.eval_pool, args.cp_time_limit)
        except KeyboardInterrupt:
            sys.exit("interrupted while building pools: nothing written (the same command starts over)")
        pools = {"regime": "reroster", "reroster": spec, "size": [n, m, k], "train": [], "eval": eval_rec,
                 "rejected": rejected}
        pools_file.write_text(json.dumps(pools, indent=1) + "\n")
        print(f"  {len(eval_ctxs)} held-out contexts, {len(rejected)} rejected ({time.time() - t0:.0f} s)")
    else:
        if args.resume or args.evaluate_only:
            sys.exit(f"{pools_file} not found: nothing to resume")
        t0 = time.time()
        print(f"building pools of feasible {n}x{m}x{k} instances with CP-SAT ...", flush=True)
        try:
            train_insts, train_rec = feasible_pool(n, m, k, args.train_seed0, args.train_pool,
                                                   args.cp_time_limit, "train")
            eval_insts, eval_rec = feasible_pool(n, m, k, args.eval_seed0, args.eval_pool,
                                                 args.cp_time_limit, "eval")
        except KeyboardInterrupt:
            sys.exit("interrupted while building pools: nothing written (the same command starts over)")
        pools = {"size": [n, m, k], "train": train_rec, "eval": eval_rec}
        pools_file.write_text(json.dumps(pools, indent=1) + "\n")
        eval_ctxs = [Context(inst) for inst in eval_insts]
        print(f"  {len(train_insts)} training, {len(eval_insts)} held-out instances ({time.time() - t0:.0f} s)")

    if args.published_prior is not None and P is None and not (args.resume or args.evaluate_only):
        sys.exit("--published-prior needs --reroster: in construction there is no x° to follow")
    train_profile = burke_curtois_training(w_min_minutes=args.w_min_minutes,
                                           w_min_consecutive=args.w_min_consecutive)
    if args.resume or args.evaluate_only:      # the encoding the model was trained with
        prev = json.loads((run / "config.json").read_text())["args"] if (run / "config.json").exists() else {}
        args.features = prev.get("features", "table2")
    if P is not None:                          # re-rostering: one instance, a fresh xi ~ P every episode
        env = GymRosterEnv([P.instance], train_profile, contexts=P, eval_profile=BURKE_CURTOIS,
                           kappa=args.kappa, reward_scale=args.reward_scale, features=args.features)
    else:
        env = GymRosterEnv(train_insts, train_profile, eval_profile=BURKE_CURTOIS,
                           kappa=args.kappa, reward_scale=args.reward_scale, features=args.features)
    eval_env = GymRosterEnv([c.instance for c in eval_ctxs], train_profile, contexts=eval_ctxs,
                            eval_profile=BURKE_CURTOIS, kappa=args.kappa, reward_scale=args.reward_scale,
                            features=args.features)
    val_env = None                             # validation contexts, for the best checkpoint
    if args.val_pool > 0:
        if P is None:
            sys.exit("--val-pool is for re-rostering runs (--reroster)")
        val_ctxs = [P.context(args.val_seed0 + j) for j in range(args.val_pool)]
        val_env = GymRosterEnv([c.instance for c in val_ctxs], train_profile, contexts=val_ctxs,
                               eval_profile=BURKE_CURTOIS, kappa=args.kappa, reward_scale=args.reward_scale,
                               features=args.features)

    config = {"args": {k_: (str(v) if isinstance(v, Path) else v) for k_, v in vars(args).items()},
              "machine": versions(), "train_profile": train_profile.describe(),
              "regime": "reroster" if P is not None else "construction",
              "disruption": P.describe() if P is not None else None,
              "eval_profile": BURKE_CURTOIS.name, "gamma": args.gamma,
              "started": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    (run / ("config.json" if not (args.resume or args.evaluate_only) else "config_resume.json")).write_text(
        json.dumps(config, indent=2) + "\n")

    latest = run / "latest.zip"
    if args.resume or args.evaluate_only:
        if not latest.exists():
            sys.exit(f"{latest} not found")
        model = MaskablePPO.load(latest, env=env, device="cpu")
        print(f"loaded {latest} at {model.num_timesteps} steps")
    else:
        lr = args.lr if args.lr_schedule == "constant" else (lambda frac, lr0=args.lr: lr0 * frac)
        model = MaskablePPO(
            EmployeePolicy, env, learning_rate=lr, n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef, vf_coef=args.vf_coef, target_kl=args.target_kl,
            policy_kwargs=dict(n_emp=n, n_shifts=k, hidden=tuple(args.hidden), value_hidden=(args.hidden[-1],),
                               separate_critic=args.separate_critic, features=args.features,
                               published_prior=args.published_prior),
            tensorboard_log=str(run / "tb"), seed=args.seed, verbose=0, device="cpu")
        if args.init_from:
            from stable_baselines3.common.save_util import load_from_zip_file

            from nrrp.policy import transfer_state
            src = args.runs / args.init_from / "latest.zip"
            if not src.exists():
                sys.exit(f"{src} not found")
            data, params, _ = load_from_zip_file(src, device="cpu")
            try:
                copied = transfer_state(model.policy, params["policy"])
            except ValueError as err:
                sys.exit(f"--init-from {args.init_from}: {err}")
            src_n = data["policy_kwargs"].get("n_emp")
            print(f"policy initialised from {src} ({len(copied)} tensors; trained with {src_n} employees "
                  f"for {data.get('num_timesteps', 0):,d} steps)")

    progress = run / "progress.csv"

    progress_inst = run / "progress_instances.csv"
    validation, best_zip, best_json = run / "validation.csv", run / "best.zip", run / "best.json"

    def log_validation(steps):
        """Score the deterministic policy on the validation contexts; keep the best checkpoint."""
        vals = [run_episode(val_env, model, j, True)[:2] for j in range(args.val_pool)]
        score = [float(np.mean([not f for _, f in vals])), float(np.mean([j for j, _ in vals]))]
        best = json.loads(best_json.read_text()) if best_json.exists() else None
        better = best is None or tuple(score) < tuple(best["score"])     # fewer infeasible, then lower J
        if better:
            model.save(best_zip)
            best_json.write_text(json.dumps({"steps": steps, "score": score}) + "\n")
        new = not validation.exists()
        with validation.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["steps", "J_val", "feasible_val", "best"])
            w.writerow([steps, score[1], 1 - score[0], better])
        print(f"  {'':>9s}        {'':>7s}     validation: det {score[1]:8.1f}  feasible {1 - score[0]:.0%}"
              f"{'   <- best so far, saved best.zip' if better else ''}", flush=True)

    def log_eval(steps, wall):
        rows = evaluate(model, eval_env, pools["eval"], k_samples=args.k_samples, seed=args.seed)
        s = summary(rows)
        new = not progress_inst.exists()
        with progress_inst.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=INSTANCE_FIELDS)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({"steps": steps, **{key: r[key] for key in INSTANCE_FIELDS[1:]}})
        new = not progress.exists()
        with progress.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=EVAL_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"steps": steps, "wall_seconds": round(wall, 1), "J_det": s["J_det"], "J_stoch": s["J_stoch"],
                        "J_cp": s["J_cp"], "gap_det": s["gap_det"], "gap_stoch": s["gap_stoch"],
                        "feasible_det": s["feasible_det"], "feasible_stoch": s["feasible_stoch"],
                        "episodes_per_second": ""})
        logger = getattr(model, "_logger", None)       # sb3 creates it when learn() starts
        if logger is not None:
            for key, val in s.items():
                logger.record(f"eval/{key}", val)
        print(f"  {steps:>9,d} steps  {wall:7.0f} s   held-out J: det {s['J_det']:8.1f}  best-of-{args.k_samples} "
              f"{s['J_stoch']:8.1f}  CP-SAT {s['J_cp']:7.1f}   feasible: det {s['feasible_det']:.0%} "
              f"best-of-k {s['feasible_stoch']:.0%}", flush=True)
        if val_env is not None:
            log_validation(steps)
        return rows

    class Periodic(BaseCallback):
        def __init__(self, t_start):
            super().__init__()
            self.t_start, self.next_eval, self.next_save = t_start, 0, 0

        def _on_training_start(self):
            self.next_eval = (self.num_timesteps // args.eval_every + 1) * args.eval_every
            self.next_save = (self.num_timesteps // args.save_every + 1) * args.save_every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_save:
                self.model.save(latest)
                self.next_save += args.save_every
            if self.num_timesteps >= self.next_eval:
                log_eval(self.num_timesteps, time.time() - self.t_start)
                self.next_eval += args.eval_every
            return True

    if not args.evaluate_only:
        remaining = args.timesteps - model.num_timesteps
        if remaining <= 0:
            print(f"already at {model.num_timesteps} steps (>= --timesteps {args.timesteps})")
        else:
            print(f"training {remaining:,d} steps (Ctrl+C saves {latest} and stops)", flush=True)
            if model.num_timesteps == 0:
                log_eval(0, 0.0)
            interrupted = False
            previous = signal.getsignal(signal.SIGINT)
            t0 = time.time()
            try:
                model.learn(total_timesteps=remaining, callback=Periodic(t0),
                            reset_num_timesteps=False, tb_log_name="ppo", progress_bar=False)
            except KeyboardInterrupt:
                interrupted = True
            finally:
                signal.signal(signal.SIGINT, previous)
                model.save(latest)
                print(f"saved {latest} at {model.num_timesteps:,d} steps ({time.time() - t0:.0f} s)", flush=True)
            if interrupted:
                print("interrupted: continue with --resume")
                return

    # final evaluation, recorded with the experiment results
    out = args.results / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows = log_eval(model.num_timesteps, 0.0) if args.evaluate_only else evaluate(
        model, eval_env, pools["eval"], k_samples=args.k_samples, seed=args.seed)
    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    for f in (progress, progress_inst, validation):
        if f.exists():
            (out / f.name).write_text(f.read_text())
    extra = {}
    if best_zip.exists():                      # the best checkpoint on validation, on the held-out pool
        best = json.loads(best_json.read_text())
        best_model = MaskablePPO.load(best_zip, env=env, device="cpu")
        rows_best = evaluate(best_model, eval_env, pools["eval"], k_samples=args.k_samples, seed=args.seed)
        with (out / "results_best.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
            w.writeheader()
            w.writerows(rows_best)
        extra = {"best_steps": best["steps"], "best_validation": best["score"], "summary_best": summary(rows_best)}
        sb = extra["summary_best"]
        print(f"\nbest checkpoint on validation: {best['steps']:,d} steps; on the held-out pool J det "
              f"{sb['J_det']:.1f}, feasible {sb['feasible_det']:.0%} (results_best.csv)")
    (out / "run.json").write_text(json.dumps({**config, "steps": model.num_timesteps,
                                              "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                              "summary": summary(rows), **extra}, indent=2) + "\n")
    s = summary(rows)
    print(f"\nfinal, {len(rows)} held-out instances, {model.num_timesteps:,d} steps:")
    print(f"  mean J: policy deterministic {s['J_det']:.1f}, best of {args.k_samples} {s['J_stoch']:.1f}, "
          f"CP-SAT {s['J_cp']:.1f}")
    print(f"  feasible under {BURKE_CURTOIS.name}: deterministic {s['feasible_det']:.0%}, "
          f"best of {args.k_samples} {s['feasible_stoch']:.0%}")
    print(f"  written {out}/results.csv; plot with: python scripts/plot_results.py train {out}")


if __name__ == "__main__":
    main()
