# nrrp-amortized

Code for the paper on the Nurse Re-rostering Problem (NRRP) formulated as an
amortized optimization problem: a constructive MDP solved by a policy trained
with masked PPO, with action masking for prefix-monotone hard constraints and a
CP repair step for the rest.

**Status.** Build order steps 1-5 are done: instance model and Burke & Curtois
reader, constraint profile, the objective J, the MDP environment with its mask,
the smoke test, and the environment benchmark. Not yet: Gymnasium wrapper and
MaskablePPO training, CP repair (OR-Tools), experiments. Nothing here is an
experimental result.

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
| `nrrp/instance.py` | instance data; context phi = (instance, x°, xi, d_xi, mu) |
| `nrrp/bc_format.py` | Burke & Curtois text format, published sizes |
| `nrrp/constraints.py` | constraint types: reference violation, prefix-monotone flag, incremental delta; Lemma 1 as an executable search |
| `nrrp/profile.py` | ConstraintProfile (Section 3.2): hard / penalised / off; derives the mask set and H^ter |
| `nrrp/objective.py` | J(x; phi), its per-step decomposition, the context constant K(phi) (Prop. 2) |
| `nrrp/env.py` | the constructive MDP (Section 3.5): slots, mask (Prop. 3), reward, incremental features |
| `nrrp/generator.py` | synthetic instances; provisional disruption generator (the P of Eq. (23) is still to be specified) |
| `nrrp/validate.py` | the smoke test: from-scratch checks of state, mask, features and rewards |

## Conventions

- Days are 0-indexed, so construction, d_xi = 1 in the paper, is `d_xi = 0`.
- A roster is an `(I, D)` int8 array: shift index, or `OFF = -1`. The paper's
  x_{ids} is `y[i, d] == s`.
- Slots are day-major over d >= d_xi, then shift types in index order, then
  rho_bar = min(rho + kappa, I) slots per (d, s). Prefix-monotonicity is
  defined with respect to this order.
- Action `a < I` assigns employee a to the current (d, s); `a = I` leaves the
  slot empty and is always allowed.
- The reward is minus the change of J on the partial roster. With
  `include_constants=True` the last step also charges K(phi) = J(x^0) and the
  rewards sum to -J(x) exactly; with `False` (training) they sum to K(phi) - J(x).
  The info dict of the last step reports J either way.
- Whether a constraint is hard or penalised comes from the profile, never from
  the instance or the environment.

## Workflow

One commit per step of the build order. Run `pytest` before every commit; the
GitHub workflow runs it again on every push, on Python 3.9 and 3.13.
