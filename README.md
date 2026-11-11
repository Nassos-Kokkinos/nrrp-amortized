# nrrp-amortized

Code for the paper on the Nurse Re-rostering Problem (NRRP) formulated as an
amortized optimization problem. A disruption (absent employees from day d_xi
on) hits a published roster x°, and a new roster must be produced quickly.
The repository contains:

- the instance model, the constraints and the objective J of the paper, checked
  against all 50 published Burke & Curtois solutions;
- a CP-SAT model of the same problem, for construction and re-rostering;
- a constructive MDP with action masking, trained with masked PPO, and a CP-SAT
  repair operator (Algorithm 1) that makes the policy's roster feasible within a
  time budget;
- the re-rostering regime: a disruption generator P on a published roster,
  held-out and validation contexts, and an optional prior towards x°;
- a learned large-neighbourhood search (LNS): a policy that chooses, step by
  step, which employees and days CP-SAT re-optimises;
- an evaluation in which every method gets the same total response time
  T_infer (policy inference + model building + CP-SAT search).

## Main result

At equal total response time, on Burke & Curtois Instances 5, 8 and 12, the
repair of x° without any learning is as good as or better than both learned
methods. CP-SAT hinted with x° is almost identical to it, and CP-SAT from
scratch catches up after the first fraction of a second. No learned
configuration reaches the success criterion fixed before the last runs (60% of
the held-out contexts won against the repair of x° at the same time); the best
is 5 wins out of 16. Figures and the per-row counts are in `results/figures/`.

| Instance (disruptions) | CP-SAT, 60 s | repair of x° | constructive RL + repair | learned LNS |
|---|---|---|---|---|
| 5 (small), T = 0.1 s | 1385 | 1417 | 1503 | – |
| 8 (big), T = 0.56 s | 2251 | 2316 | – | 2387 |
| 12 (big), T = 2.77 s | 4748 | 4712 | – | 5075 |

Mean J over 16 held-out contexts; lower is better. Timings were measured on an
Apple silicon Mac (arm64, 10 cores, macOS 26.3, Python 3.14, OR-Tools 9.15,
8 CP-SAT workers).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,cp,plot,rl]"
```

Python 3.9 or newer (3.10 or newer for `rl`). The core needs only numpy; the
extras add OR-Tools (`cp`, for `nrrp/cpsat.py`), matplotlib (`plot`) and
PyTorch, Gymnasium, stable-baselines3, sb3-contrib, TensorBoard (`rl`, for
training). OR-Tools has no wheel for Python 3.15 yet (October 2026): use
Python 3.14 for the venv. Tests of a missing extra are skipped.

Instance and solution files are not in the repository; see
[data/README.md](data/README.md) for where to get them and where to put them.

## Reproducing the paper

Every committed run has its full parameters in `results/<experiment>/<run>/run.json`;
the commands below are those parameters written out. Checkpoints and the pool
files (`runs/<name>/`, including `pools.json`) are not committed: training
rebuilds the pools from their seeds (training pool from seed 0, held-out pool
from 100000, validation pool from 200000).

**1. Tests and the model against the benchmark**

```bash
pytest
python scripts/check_solutions.py data/raw/solutions/*.roster      # J of all 50 published solutions
python scripts/solve_cp.py data/raw/Instance*.txt --run mac-60s     # CP-SAT, 60 s per instance
```

**2. Construction on synthetic instances (40×28×3)**

```bash
python scripts/train.py --name 25x7-F-ext-sep --size 25 7 3 --timesteps 2000000 \
    --features extended --separate-critic --ent-coef 0.001
python scripts/train.py --name mac-40x28-F-init --size 40 28 3 --timesteps 3000000 --eval-every 200000 \
    --features extended --separate-critic --ent-coef 0.001 --init-from 25x7-F-ext-sep --cp-time-limit 60
for s in 0 1 2; do
  python scripts/eval_repair.py --name mac-40x28-F-init --budgets 0 0.1 0.25 0.5 1 2 5 --seed $s \
      --out-name mac-40x28-F-init-anytime-s$s
  python scripts/eval_repair.py --name mac-40x28-F-init --budgets 0 0.1 0.25 0.5 1 2 5 --seed $s \
      --source random --out-name mac-40x28-F-init-random-anytime-s$s
done
```

(The committed seed-0 folders are named without the `-s0` suffix.)

**3. Why absences are credited (Remark 3)**

How many disruptions leave no feasible roster, with the absence credit and with
the benchmark's constraints read literally:

```bash
python scripts/feasibility_rate.py --instance data/raw/Instance5.txt --n-absent 1 2 --absence-days 1 7
python scripts/feasibility_rate.py --instance data/raw/Instance5.txt --n-absent 1 2 --absence-days 1 1
```

**4. Constructive re-rostering (Instances 5 and 8)**

x° is the published solution with the lowest value (`data/raw/solutions/`), which
is not always the best known value. Small disruptions: 1–2 absent employees for
1–7 days; big: 3–5 for 3–7 days.

```bash
B=(--budgets 0 0.1 0.25 0.5 1 2 5)
C=(--features extended --separate-critic --ent-coef 0.001 --timesteps 2000000)
python scripts/train.py --name bc5-rr --reroster data/raw/Instance5.txt --n-absent 1 2 --absence-days 1 7 \
    --cp-time-limit 60 "${C[@]}"
python scripts/train.py --name bc5-rr-prior5 --reroster data/raw/Instance5.txt --pools-from bc5-rr \
    --published-prior 5 "${C[@]}"
python scripts/eval_repair.py --name bc5-rr-prior5 "${B[@]}"                      # policy + repair
python scripts/eval_repair.py --name bc5-rr "${B[@]}" --source published          # repair of x°, CP-SAT baselines
python scripts/train.py --name bc8-big-p5 --reroster data/raw/Instance8.txt --n-absent 3 5 --absence-days 3 7 \
    --published-prior 5 --lr 3e-5 --val-pool 16 --cp-time-limit 60 "${C[@]}"
python scripts/eval_repair.py --name bc8-big-p5 "${B[@]}" --checkpoint best
python scripts/eval_repair.py --name bc8-big-p5 "${B[@]}" --source published
```

**5. Learned LNS (Instances 8 and 12, big disruptions)**

```bash
python scripts/train_lns.py --name bc8-lns1 --pools-from bc8-big-p5 --step-time 0.1 --timesteps 50000
python scripts/train_lns.py --name bc8-lns2 --pools-from bc8-big-p5 --step-time 0.1 --timesteps 50000 \
    --release both
python scripts/train_lns.py --name bc12-lns --reroster data/raw/Instance12.txt --n-absent 3 5 --absence-days 3 7 \
    --step-time 0.5 --timesteps 20000 --eval-every 4000 --release both
python scripts/eval_lns.py --name bc8-lns1 --steps-list 2 4 8 16
python scripts/eval_lns.py --name bc8-lns2 --steps-list 2 4 8 16
python scripts/eval_lns.py --name bc12-lns --steps-list 2 4 8
```

`eval_lns.py` runs the learned LNS, a random LNS, the repair of x°, CP-SAT from
scratch and CP-SAT hinted with x°, all with the same time per context.

**6. Figures**

```bash
python scripts/plot_paper.py      # results/figures/fig{1,2,3}_*.{pdf,png}, win_tie_loss.csv
```

It reads only committed results; nothing is re-run.

## Commands

| Command | What it does |
|---|---|
| `pytest` | all tests |
| `python scripts/smoke.py [files]` | the theory validator (state, mask, features, rewards) on synthetic or Burke & Curtois instances |
| `python scripts/bench_env.py [files]` | environment throughput, random masked rollouts, steps/s |
| `python scripts/check_solutions.py data/raw/solutions/*.roster` | J and feasibility of each published solution against its published value |
| `python scripts/solve_cp.py FILES --run NAME [--time-limit 60]` | CP-SAT on each instance; results to `results/cp_baseline/NAME/` |
| `python scripts/train.py --name NAME --size n m k ...` | MaskablePPO on a pool of synthetic instances (construction); Ctrl+C saves, `--resume` continues |
| `python scripts/train.py --name NAME --reroster INSTANCE ...` | the same in the re-rostering regime: fresh disruption per episode, held-out contexts; `--published-prior B0`, `--val-pool N` (best checkpoint on validation contexts) |
| `python scripts/eval_repair.py --name NAME --budgets ...` | policy, policy + repair, CP-SAT from scratch and hinted with x°, at the same T_infer; `--source published` repairs x° instead of the policy's roster, `--checkpoint best` |
| `python scripts/feasibility_rate.py --instance FILE ...` | share of disruptions with no feasible roster, with and without the absence credit; results to `results/feasibility/` |
| `python scripts/train_lns.py --name NAME ...` | learned LNS; `--release employees` or `both` (employees and days) |
| `python scripts/eval_lns.py --name NAME --steps-list ...` | learned LNS against random LNS, repair of x°, CP-SAT from scratch and hinted, same budget |
| `python scripts/plot_results.py {cp,train,grid} ...` | per-run diagnostic figures |
| `python scripts/plot_paper.py` | the figures of the paper |
| `tensorboard --logdir runs` | training curves while a run is going |

## Layout

| Module | Paper |
|---|---|
| `nrrp/instance.py` | instance data (Table 1); context phi = (Inst, x°, xi), Definition 1; absence credit towards H^min |
| `nrrp/bc_format.py` | Burke & Curtois text format, published sizes and best known values; the `.roster` solution format (x°) |
| `nrrp/constraints.py` | constraints (3)-(10), minimum rest, weekend cap: reference violation, prefix-monotone flag, incremental delta; Lemma 1 as an executable search |
| `nrrp/profile.py` | the hard / penalised / off partition of a family (Remark 1); H^pre and H^ter; evaluation and training profiles |
| `nrrp/objective.py` | J(x; phi) (20), its per-step decomposition and c^con (Prop. 2), evaluation |
| `nrrp/env.py` | the MDP M(phi) of Section 3.5: slots (27), actions (28), mask (32), reward, state features (Table 2) |
| `nrrp/cpsat.py` | CP-SAT model of (15) and of re-rostering: frozen history, absences, stability; `free` / `base` for the repair subproblem |
| `nrrp/repair.py` | the repair operator: neighbourhood (40)-(41), Algorithm 1 in its anytime form; model building counts against the budget |
| `nrrp/disruption.py` | the disruption generator P (Eq. (25)) on a published roster; x° without the absences; Delta(x, x°) |
| `nrrp/gym_env.py` | Gymnasium view of the MDP: fresh contexts from P or a fixed held-out list, `action_masks()` |
| `nrrp/policy.py` | the network of Eq. (39) (shared per-employee MLP), the prior towards x°, and the learned-LNS policy |
| `nrrp/lns_env.py` | the learned LNS as an MDP: release employees and/or days, CP-SAT with hint x_t, telescoping reward |
| `nrrp/generator.py` | synthetic instances and random contexts |
| `nrrp/validate.py` | the smoke test: from-scratch checks of state, mask, features and rewards |

## Notation: paper and code

| Paper | Code |
|---|---|
| n, m, k | `inst.n_emp`, `inst.n_days`, `inst.n_shifts` |
| day d = 1..m | day `d - 1`: days are 0-indexed, so construction (d_xi = 1) is `d_xi = 0` |
| x_{i,d,s} | `roster[i, d] == s`; `roster` is an (n, m) int8 array, `OFF = -1` is the day-off symbol 0 |
| y_{i,d} | `roster[i, d] != OFF` |
| x°, A_xi, d_xi, mu | `ctx.published`, `ctx.absent`, `ctx.d_xi`, `ctx.mu` |
| l_s, F, U | `inst.shift_len`, `inst.forbidden`, `inst.day_off` |
| H^min, H^max | `inst.min_minutes`, `inst.max_minutes`; in a context `ctx.min_minutes` = H^min less the shifts x° had on absent days |
| C^min, C^max, R^min | `inst.min_consec`, `inst.max_consec`, `inst.min_consec_off` |
| N^max_{i,s}, weekend cap | `inst.max_shifts`, `inst.max_weekends` |
| Q_i, kappa_s | `inst.skills_held`, `inst.skill_required`; `inst.eligible[i, s]` is kappa_s in Q_i |
| w^+, w^-, p | `inst.on_req`, `inst.off_req`, p = off_req - on_req |
| rho, w^u, w^o, r | `inst.rho`, `inst.w_under`, `inst.w_over`, `inst.min_cover` |
| H^pre, H^ter | `profile.pre`, `profile.ter` |
| kappa, rho_bar, T | `kappa`, `min(rho + kappa, n)`, `env.T` |
| M(s_t) | `env.action_mask()`; index n is the empty action |
| c_t, c^con = J(x^(0)) | `-reward`, `env.c_con` |
| J(x; phi) | `objective.J(roster, ctx, profile)` |
| Delta(x, x°) | `disruption.changes(roster, ctx)` |
| psi_i, g (Table 2) | rows of `env.E`, vector `env.G`; columns in `RosterEnv.FEATURES` |

## Conventions

- Slots are day-major over d >= d_xi, then shift types in index order, then
  rho_bar = min(rho + kappa, n) slots per (d, s). Prefix-monotonicity is
  defined with respect to this order.
- The reward is -c_t with c_t = J(x^(t+1)) - J(x^(t)) on partial rosters. With
  `include_constants=True` the last step also charges c^con = J(x^(0)) and the
  rewards sum to -J(x^(T)) exactly; with `False` (training, Remark 7) they sum
  to c^con - J(x^(T)).
- Whether a constraint is hard or penalised comes from a profile, never from
  the instance or the environment. Evaluation uses the family's published
  profile (`BURKE_CURTOIS`); training may penalise H^ter as reward shaping
  (`burke_curtois_training(...)`, weights given explicitly).
- An absence counts towards the minimum total minutes (the shifts x° had on the
  absent days are credited), and a work stretch next to an absent day is
  exempt from the minimum stretch, as at the end of the horizon.
- Equal-time comparison: every method gets the same total time T_infer, and the
  CP-SAT model building is counted in it.

## Results

`results/<experiment>/<run>/` holds one run: `results.csv` (written row by row,
so an interrupted run keeps what it finished) and `run.json` (machine, versions,
parameters, summary). Experiments: `cp_baseline`, `train`, `repair`,
`train_lns`, `lns`, `feasibility`, and `figures` for the paper. Runs are committed; runs named
`provisional-*` are not (timings measured where they do not count, e.g. in a
VM). Timings for the paper are measured natively on the Mac above.

## Workflow

One commit per step. Run `pytest` before every commit; the GitHub workflow runs
it again on every push, on Python 3.9 and 3.13.
