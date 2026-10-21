# nrrp-amortized

Code for the paper on the Nurse Re-rostering Problem (NRRP) formulated as an
amortized optimization problem: a constructive MDP solved by a policy trained
with masked PPO, with action masking for prefix-monotone hard constraints and a
CP repair step for the rest.

Status: under construction. Nothing in this repository is an experimental
result yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest
```
