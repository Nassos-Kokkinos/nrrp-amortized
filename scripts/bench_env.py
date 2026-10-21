#!/usr/bin/env python3
"""Environment throughput: random masked rollouts in steps per second. No learning.

  python scripts/bench_env.py                              synthetic sizes
  python scripts/bench_env.py data/raw/Instance*.txt       Burke & Curtois files

Measures the environment alone (reset, mask, features, reward, observation
copy), construction regime, the published Burke & Curtois profile. PPO adds the
network on top, so this is an upper bound on training throughput: it tells
whether the environment can become the bottleneck.
"""
import argparse
import platform
import time

import numpy as np

from nrrp.bc_format import read_bc
from nrrp.env import RosterEnv
from nrrp.generator import synthetic_instance
from nrrp.profile import BURKE_CURTOIS


def bench(inst, seconds: float, kappa: int, seed: int = 0):
    env = RosterEnv(inst, BURKE_CURTOIS, kappa=kappa, seed=seed)
    rng = np.random.default_rng(seed)

    def episode():
        env.reset()
        n, done = 0, False
        while not done:
            allowed = np.flatnonzero(env.mask)
            done = env.step(int(allowed[rng.integers(len(allowed))]))[2]
            n += 1
        return n

    episode()  # warm-up
    steps, episodes, t0 = 0, 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds or episodes < 2:
        steps += episode()
        episodes += 1
    dt = time.perf_counter() - t0
    return env.T, env.obs_dim, steps / dt, dt / episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--seconds", type=float, default=3.0, help="time per instance")
    ap.add_argument("--kappa", type=int, default=1)
    args = ap.parse_args()

    if args.files:
        instances = [read_bc(f) for f in args.files]
    else:
        instances = [synthetic_instance(*s, seed=0) for s in
                     [(5, 7, 3), (12, 7, 3), (25, 7, 3), (25, 28, 4), (50, 28, 6), (100, 182, 8)]]
    print(f"{platform.platform()} | {platform.processor() or platform.machine()} | "
          f"Python {platform.python_version()} | numpy {np.__version__}\n")
    print(f"{'instance':<22} {'I x D x S':>14} {'slots/ep':>9} {'obs dim':>8} {'steps/s':>10} {'s/episode':>10}")
    for inst in instances:
        T, dim, sps, spe = bench(inst, args.seconds, args.kappa)
        size = f"{inst.n_emp}x{inst.n_days}x{inst.n_shifts}"
        print(f"{inst.name:<22} {size:>14} {T:>9,} {dim:>8,} {sps:>10,.0f} {spe:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
