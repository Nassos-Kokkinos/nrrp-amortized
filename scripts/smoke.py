#!/usr/bin/env python3
"""Smoke test: the theory validator, as a report.

  python scripts/smoke.py                                  synthetic instances
  python scripts/smoke.py data/raw/Instance1.txt ...       Burke & Curtois files
  python scripts/smoke.py data/raw/Instance*.txt --episodes 3

Each file is parsed and its size compared with the published table. Every
(instance, profile, regime) is then run with random masked rollouts. On small
instances every step is also checked against from-scratch recomputation
(state, mask, features, reward); above PER_STEP_LIMIT cells only the
end-of-episode checks run. Exit code 0 iff everything passes.
"""
import argparse
import platform
import sys
import time

import numpy as np

from nrrp.bc_format import FormatError, published_mismatch, read_bc
from nrrp.generator import synthetic_instance
from nrrp.validate import TEST_PROFILES, Report, smoke

PER_STEP_LIMIT = 5_000  # I * D * S: Instance1-9 and 16 get per-step checks

LEGEND = """
columns: ep, steps = episodes and steps run
  Prop2 Σ   worst |sum of rewards + J - target| / max(1, J), target 0 with K included, K without
  Prop2 t   worst per-step |reward + change of J|, J from scratch ("off" on large instances)
  Prop3     total violation of masked hard constraints in the final rosters (must be 0)
  mask      steps where the mask differed from its definition
  state     incremental-state arrays that differed from the from-scratch state
  features  largest difference between the observation and its from-scratch version
  H^ter     employees whose final row violates H^ter (random policy; work left for repair)
profiles (weights are arbitrary test values): burke_curtois = as published, all hard;
  soft_ter = H^ter penalised; all_soft = everything except absences penalised
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="instance files in the Burke & Curtois text format")
    ap.add_argument("--episodes", type=int, help="per configuration (default: 5 synthetic, 2 for files)")
    ap.add_argument("--kappa", type=int, default=1, help="rho_bar = rho + kappa slots per (day, shift)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profiles", nargs="+", choices=list(TEST_PROFILES), default=list(TEST_PROFILES))
    args = ap.parse_args()

    print(f"{platform.platform()} | Python {platform.python_version()} | numpy {np.__version__}")
    ok = True
    instances = []
    if args.files:
        for f in args.files:
            try:
                inst = read_bc(f)
            except (OSError, FormatError) as e:
                print(f"FAIL  {f}: {e}")
                ok = False
                continue
            problem = published_mismatch(inst)
            print(f"{'FAIL' if problem else 'ok  '}  {f}: {inst.size}" + (f"  <- {problem}" if problem else ""))
            ok &= problem is None
            instances.append(inst)
    else:
        instances = [synthetic_instance(*size, seed=args.seed) for size in [(5, 7, 3), (12, 14, 3), (20, 28, 4)]]
    episodes = args.episodes or (2 if args.files else 5)

    print("\n" + Report.HEADER)
    t_all = time.perf_counter()
    for inst in instances:
        per_step = inst.n_emp * inst.n_days * inst.n_shifts <= PER_STEP_LIMIT
        for pname in args.profiles:
            for regime in ("construction", "rerostering") if inst.n_days >= 2 else ("construction",):
                t0 = time.perf_counter()
                r = smoke(inst, TEST_PROFILES[pname], regime=regime, episodes=episodes,
                          kappa=args.kappa, per_step=per_step, seed=args.seed,
                          label=f"{inst.name} / {pname} / {regime}")
                print(f"{r.line()} {time.perf_counter() - t0:6.1f}s", flush=True)
                ok &= r.ok
    print(LEGEND)
    print(f"{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'} ({time.perf_counter() - t_all:.0f} s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
