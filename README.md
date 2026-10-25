# nrrp-amortized

Code for the paper on the Nurse Re-rostering Problem (NRRP) formulated as an
amortized optimization problem: a constructive MDP solved by a policy trained
with masked PPO, with action masking for prefix-monotone hard constraints and a
CP repair step for the rest.

**Status.** Build order steps 1-5 are done: instance model and Burke & Curtois
reader, constraint profiles, the objective J, the MDP environment with its mask,
the smoke test, and the environment benchmark. The code follows the paper as
agreed on 25/10/2026 (`code-paper-agreement.md` in the project). Not yet:
the policy network (34) and MaskablePPO training, CP repair (OR-Tools),
experiments. Nothing here is an experimental result.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Python 3.9 or newer; the only runtime dependency so far is numpy.

## Commands

| Command | What it does |
|---|---|
| `pytest` | all tests, about 20 s |
| `python scripts/smoke.py` | the theory validator on synthetic instances, as a report |
| `python scripts/smoke.py data/raw/Instance*.txt` | the same on Burke & Curtois files, after checking each file's size against the published table |
| `python scripts/bench_env.py [files]` | environment throughput, random masked rollouts, steps/s |

Instance files are not in the repository; see [data/README.md](data/README.md).

## Layout

| Module | Paper |
|---|---|
| `nrrp/instance.py` | instance data (Table 1); context phi = (Inst, x°, xi), Definition 1 |
| `nrrp/bc_format.py` | Burke & Curtois text format, published sizes and best known values |
| `nrrp/constraints.py` | constraints (3)-(10), minimum rest, weekend cap: reference violation, prefix-monotone flag, incremental delta; Lemma 1 as an executable search |
| `nrrp/profile.py` | the hard / penalised / off partition of a family (Remark 1); derives H^pre and H^ter; evaluation and training profiles |
| `nrrp/objective.py` | J(x; phi) (20), its per-step decomposition and c^con (Prop. 2), evaluation |
| `nrrp/env.py` | the MDP M(phi) of Section 3.5: slots (27), actions (28), mask (32), reward, state features (Table 2) |
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

## Workflow

One commit per step. Run `pytest` before every commit; the GitHub workflow runs
it again on every push, on Python 3.9 and 3.13.
