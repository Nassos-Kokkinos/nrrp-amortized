"""Synthetic instances and re-rostering contexts.

``synthetic_instance`` lets the whole pipeline run without the benchmark files.
Its parameter ranges imitate the Burke & Curtois instances (8-hour shifts,
cover weights 100 under / 1 over, sparse requests of weight 1-3); they are not
calibrated to any real ward.

``random_context`` is PROVISIONAL. The disruption distribution is the P of
Eq. (23) and its parameters must be fixed and reported in Section 5; the one
here exists only so the environment can be tested in the re-rostering regime.
"""
from __future__ import annotations

import numpy as np

from .instance import OFF, Context, Instance


def synthetic_instance(n_emp: int = 12, n_days: int = 14, n_shifts: int = 3,
                       seed: int = 0, name: str | None = None, *, skills: bool = False,
                       min_cover: bool = False) -> Instance:
    """``skills``: two skills, the last shift type needs the second, which about
    half the employees hold. ``min_cover``: a hard minimum r = floor(0.6 rho).
    Both are absent from the Burke & Curtois family and off by default."""
    if n_emp < 1 or n_days < 1 or n_shifts < 1:
        raise ValueError("need at least one employee, day and shift")
    rng = np.random.default_rng(seed)
    I, D, S = n_emp, n_days, n_shifts
    base = ["D", "E", "L", "N"]
    shift_ids = base[:S] if S <= len(base) else base + [f"S{k}" for k in range(len(base), S)]
    shift_len = rng.choice([360, 480, 480, 600, 720], S)
    a = np.arange(S)
    forbidden = a[None, :] < a[:, None]  # a later shift may not be followed by an earlier one
    weeks = D / 7
    avg = float(shift_len.mean())
    max_minutes = np.round(rng.choice([4.0, 4.5, 5.0], I) * weeks * avg).astype(np.int64)
    min_minutes = np.round(max_minutes * rng.uniform(0.6, 0.8, I)).astype(np.int64)
    max_shifts = np.where(rng.random((I, S)) < 0.75, D, rng.integers(0, D // 2 + 1, (I, S)))
    max_consec = rng.choice([4, 5, 6], I)
    min_consec = rng.choice([1, 2, 2, 3], I)
    min_consec_off = rng.choice([1, 2, 2, 3], I)
    k = np.arange(D)
    n_weekends = len(np.unique((k // 7)[k % 7 >= 5]))
    max_weekends = np.maximum(1, np.ceil(n_weekends * rng.uniform(0.4, 0.8, I))).astype(np.int64)
    day_off = rng.random((I, D)) < 1 / 14
    on_req = (rng.random((I, D, S)) < 0.04) * rng.integers(1, 4, (I, D, S))
    off_req = (rng.random((I, D, S)) < 0.04) * rng.integers(1, 4, (I, D, S))
    per_day = ((min_minutes + max_minutes) / 2 / avg).sum() / D * 0.9
    share = np.linspace(1.4, 0.6, S)
    share /= share.sum()
    rho = np.maximum(0, np.rint(per_day * share + rng.normal(0, 0.6, (D, S)))).astype(np.int64)
    extra = {}
    if skills:
        held = np.ones((I, 2), bool)
        held[:, 1] = rng.random(I) < 0.5
        required = np.full(S, 0, np.int64)
        required[0] = -1
        required[-1] = 1 if S > 1 else -1
        extra.update(skills_held=held, skill_required=required)
    if min_cover:
        extra["min_cover"] = np.floor(0.6 * rho).astype(np.int64)
    return Instance(
        name=name or f"syn{I}x{D}x{S}s{seed}", employee_ids=[f"E{i:02d}" for i in range(I)],
        shift_ids=shift_ids, n_days=D, shift_len=shift_len, forbidden=forbidden,
        max_shifts=max_shifts, max_minutes=max_minutes, min_minutes=min_minutes,
        max_consec=max_consec, min_consec=min_consec, min_consec_off=min_consec_off,
        max_weekends=max_weekends, day_off=day_off, on_req=on_req, off_req=off_req,
        rho=rho, w_under=np.full((D, S), 100.0), w_over=np.full((D, S), 1.0), **extra)


def random_roster(instance: Instance, profile, rng: np.random.Generator, kappa: int = 1) -> np.ndarray:
    """A roster from a uniformly random masked construction.

    By Prop. 3 it satisfies H^pre of the profile, so its frozen prefix meets the
    assumption of Prop. 3. It ignores H^ter and the objective. Remark 5 asks for a
    solution of the construction problem (13) instead; that comes with CP-SAT.
    """
    from .env import RosterEnv  # imported here: the environment depends on this package's core

    env = RosterEnv(instance, profile, kappa=kappa)
    env.reset()
    done = False
    while not done:
        allowed = np.flatnonzero(env.action_mask())
        _, _, done, _ = env.step(int(allowed[rng.integers(len(allowed))]))
    return env.roster.copy()


def random_context(instance: Instance, profile, rng: np.random.Generator, *, mu: float = 1.0,
                   n_absent: int = 1, max_absence: int = 3, d_xi: int | None = None,
                   kappa: int = 1) -> Context:
    """PROVISIONAL disruption generator (see the module docstring).

    Published roster: :func:`random_roster`. Disruption: d_xi uniform on
    1..D-1; ``n_absent`` employees, preferably among those working on d_xi in
    the published roster, each absent for 1..max_absence days from d_xi.
    """
    D = instance.n_days
    if D < 2:
        raise ValueError("re-rostering needs at least two days")
    published = random_roster(instance, profile, rng, kappa)
    if d_xi is None:
        d_xi = int(rng.integers(1, D))
    working = np.flatnonzero(published[:, d_xi] != OFF)
    pool = working if len(working) >= n_absent else np.arange(instance.n_emp)
    absent = np.zeros((instance.n_emp, D), bool)
    for i in rng.choice(pool, size=min(n_absent, len(pool)), replace=False):
        absent[i, d_xi:d_xi + int(rng.integers(1, max_absence + 1))] = True
    return Context(instance, d_xi=d_xi, published=published, absent=absent, mu=mu,
                   name=f"{instance.name}@d{d_xi}")
