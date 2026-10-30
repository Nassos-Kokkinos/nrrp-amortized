# nrrp-amortized

Code for the paper on the Nurse Re-rostering Problem (NRRP) formulated as an
amortized optimization problem: a constructive MDP solved by a policy trained
with masked PPO, with action masking for prefix-monotone hard constraints and a
CP repair step for the rest.

**Status.** Build order steps 1-5 are done: instance model and Burke & Curtois
reader, constraint profiles, the objective J, the MDP environment with its mask,
the smoke test, and the environment benchmark. The objective and the hard
constraints reproduce the benchmark's own values on all 50 published Burke &
Curtois solutions (`scripts/check_solutions.py`). A CP-SAT model of the same
problem (`nrrp/cpsat.py`) solves Instance1-3 to the published optima (607, 828,
1001) and gives x° for instances without a published roster. The policy of
Eq. (39) trains with MaskablePPO over pools of instances (`scripts/train.py`),
evaluated against CP-SAT on held-out instances. The repair operator of
Section 4.3 (neighbourhood (40)-(41), Algorithm 1, `nrrp/repair.py`) turns the
policy's roster into a feasible one within a time budget tau_cp; Proposition 4
is tested. The code follows the paper as agreed on 25/10/2026
(`code-paper-agreement.md` in the project). Not yet: the re-rostering
regime of training, the disruption generator P, experiments.

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

## Commands

| Command | What it does |
|---|---|
| `pytest` | all tests, about 20 s |
| `python scripts/smoke.py` | the theory validator on synthetic instances, as a report |
| `python scripts/smoke.py data/raw/Instance*.txt` | the same on Burke & Curtois files, after checking each file's size against the published table |
| `python scripts/bench_env.py [files]` | environment throughput, random masked rollouts, steps/s |
| `python scripts/check_solutions.py data/raw/solutions/*.roster` | J and feasibility of each published solution against its published value |
| `python scripts/solve_cp.py data/raw/Instance1.txt ... --run NAME` | CP-SAT on each instance; results to `results/cp_baseline/NAME/` |
| `python scripts/plot_results.py cp results/cp_baseline/NAME` | bar charts (PDF for LaTeX, PNG preview) of a CP-SAT run |
| `python scripts/train.py --name NAME --size 5 7 3 --timesteps 200000` | MaskablePPO on a pool of feasible synthetic instances; Ctrl+C saves, `--resume` continues; held-out evaluation against CP-SAT |
| `python scripts/plot_results.py train results/train/NAME` | learning curve, feasibility curve, final gaps of a training run |
| `python scripts/plot_results.py grid results/train/NAME-s0 results/train/NAME-s1 ... --out results/train/grid` | per-instance learning curves of several runs: one panel per held-out instance, one line per seed, one colour per `--label` (cf. Fig. 6 of the PPO paper) |
| `tensorboard --logdir runs` | training curves while a run is going |
| `python scripts/eval_repair.py --name NAME --budgets 0 0.1 0.5 2 [--first-feasible]` | policy alone (tau_cp = 0) against policy with repair, per held-out instance; results to `results/repair/NAME-anytime/` (or `NAME-first/`) |

Instance and solution files are not in the repository; see [data/README.md](data/README.md).

## Layout

| Module | Paper |
|---|---|
| `nrrp/instance.py` | instance data (Table 1); context phi = (Inst, x°, xi), Definition 1 |
| `nrrp/bc_format.py` | Burke & Curtois text format, published sizes and best known values; the `.roster` solution format (x°) |
| `nrrp/constraints.py` | constraints (3)-(10), minimum rest, weekend cap: reference violation, prefix-monotone flag, incremental delta; Lemma 1 as an executable search |
| `nrrp/profile.py` | the hard / penalised / off partition of a family (Remark 1); derives H^pre and H^ter; evaluation and training profiles |
| `nrrp/objective.py` | J(x; phi) (20), its per-step decomposition and c^con (Prop. 2), evaluation |
| `nrrp/env.py` | the MDP M(phi) of Section 3.5: slots (27), actions (28), mask (32), reward, state features (Table 2) |
| `nrrp/cpsat.py` | CP-SAT model of (15) and of re-rostering: every constraint in its profile role, J as the objective; frozen history, absences, stability; `free` (cells or single variables) / `base` for the repair subproblem |
| `nrrp/gym_env.py` | Gymnasium view of the MDP over a pool of same-size instances: Box observation, Discrete(n + 1), `action_masks()`, scaled reward, full J in the last step's info |
| `nrrp/policy.py` | the network of Eq. (39) as a MaskablePPO policy: shared per-employee MLP, learned scalar for the empty action, value from the mean embedding |
| `nrrp/repair.py` | the repair operator: neighbourhood V_w = window ∪ V^∅ ∪ V^ter (40)-(41), Algorithm 1 in its anytime form (keep widening while budget remains, best roster kept; budget tau_cp; fallback with the violated constraints); `anytime=False` is the first-feasible form of the 25/10 PDF |
| `nrrp/generator.py` | synthetic instances; provisional disruption generator (the P of Eq. (23) is still to be specified) |
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
| H^min, H^max | `inst.min_minutes`, `inst.max_minutes` |
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
  (`burke_curtois_training(...)`, weights given explicitly). Pass the
  evaluation profile as `eval_profile` and the last step's info reports J and
  feasibility under it.

## Results

Training checkpoints and TensorBoard logs go to `runs/<name>/` (git-ignored);
the final evaluation of a training run goes to `results/train/<name>/`.
`results/<experiment>/<run>/` holds one run: `results.csv` (written row by row,
so an interrupted run keeps what it finished), `run.json` (machine, versions,
parameters), and `figures/` from `scripts/plot_results.py`. Runs are committed;
runs named `provisional-*` are not (timings measured where they do not count,
e.g. in a VM). Timings for the paper are measured natively on the target
machine.

## Workflow

One commit per step. Run `pytest` before every commit; the GitHub workflow runs
it again on every push, on Python 3.9 and 3.13.
