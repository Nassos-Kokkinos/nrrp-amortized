"""The disruption generator P of Remark 5 and Eq. (25): re-rostering contexts.

A context is phi = (Inst, x°, xi) with xi = (A_xi, d_xi) (Definition 1). For one
organisation the instance and the published roster x° are fixed, and only the
disruption varies; :class:`DisruptionModel` is the distribution of xi:

* d_xi uniform on ``first_day``..m-1 (0-indexed days; d_xi = 0 would leave no
  history, which is the construction regime);
* a number of absent employees uniform on ``n_absent`` = (lo, hi), chosen among
  the employees x° has working on d_xi (short-notice absence: the absence is
  reported on the day it starts), or among all employees if too few work then;
* for each, an absence of a length uniform on ``days`` = (lo, hi), from d_xi,
  clipped to the horizon;
* ``mu``: the weight of the instability Delta(x, x°) in J (Eq. (20)).

``sample(rng)`` draws a context; ``context(seed)`` draws the context of a seed,
so that a pool of held-out contexts is stored as its seeds and rebuilt exactly.

A disruption can make X(phi) empty: under the Burke & Curtois semantics the
minimum total minutes (5) and the minimum stretch (9) are hard, and an absent
employee may no longer be able to meet them. Training draws from P as it is;
held-out pools keep the contexts for which CP-SAT finds a feasible roster, and
report how many were rejected (decision of 10/11/2026).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import OFF, Context, Instance, roster_to_binary


@dataclass(frozen=True)
class DisruptionModel:
    instance: Instance
    published: np.ndarray
    n_absent: tuple = (1, 2)
    days: tuple = (1, 7)
    mu: float = 1.0
    first_day: int = 1

    def __post_init__(self):
        inst = self.instance
        object.__setattr__(self, "published", np.asarray(self.published, dtype=np.int8))
        if self.published.shape != (inst.n_emp, inst.n_days):
            raise ValueError(f"published roster shape {self.published.shape}, instance needs "
                             f"{(inst.n_emp, inst.n_days)}")
        for name in ("n_absent", "days"):
            lo, hi = getattr(self, name)
            if not 1 <= lo <= hi:
                raise ValueError(f"{name} must be (lo, hi) with 1 <= lo <= hi, not {(lo, hi)}")
            object.__setattr__(self, name, (int(lo), int(hi)))
        if self.n_absent[1] > inst.n_emp:
            raise ValueError("more absent employees than employees")
        if not 1 <= self.first_day < inst.n_days:
            raise ValueError(f"first_day must be in 1..{inst.n_days - 1}")
        if self.mu < 0:
            raise ValueError("mu must be non-negative")

    def sample(self, rng: np.random.Generator, name: str | None = None) -> Context:
        inst, x0 = self.instance, self.published
        D = inst.n_days
        d_xi = int(rng.integers(self.first_day, D))
        a = int(rng.integers(self.n_absent[0], self.n_absent[1] + 1))
        working = np.flatnonzero(x0[:, d_xi] != OFF)
        pool = working if len(working) >= a else np.arange(inst.n_emp)
        absent = np.zeros((inst.n_emp, D), bool)
        for i in rng.choice(pool, size=a, replace=False):
            length = int(rng.integers(self.days[0], self.days[1] + 1))
            absent[i, d_xi: min(d_xi + length, D)] = True
        return Context(inst, d_xi=d_xi, published=x0, absent=absent, mu=self.mu,
                       name=name or f"{inst.name}@d{d_xi}")

    def context(self, seed: int) -> Context:
        return self.sample(np.random.default_rng(seed), name=f"{self.instance.name}-xi{seed}")

    def __call__(self, rng: np.random.Generator, instance: Instance | None = None) -> Context:
        """The ``contexts`` callable of :class:`nrrp.gym_env.GymRosterEnv`: one fresh context per episode."""
        if instance is not None and instance is not self.instance:
            raise ValueError("this disruption model is for another instance")
        return self.sample(rng)

    def describe(self) -> dict:
        return {"instance": self.instance.name, "n_absent": list(self.n_absent), "days": list(self.days),
                "mu": self.mu, "first_day": self.first_day}


def without_absences(ctx: Context) -> np.ndarray:
    """x° with the assignments of absent employees removed: the starting point of the
    classical re-rostering baseline (repair the published roster, no learned policy).
    Satisfies the history constraint, since absences only mark days d >= d_xi."""
    x = ctx.published.copy()
    x[ctx.absent] = OFF
    return x


def changes(roster: np.ndarray, ctx: Context) -> int:
    """Delta(x, x°) of Eq. (18): assignments created plus assignments withdrawn."""
    S = ctx.instance.n_shifts
    return int((roster_to_binary(roster, S) != roster_to_binary(ctx.published, S)).sum())
