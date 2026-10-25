"""Constraint profile: which constraints are hard, which penalised, for one instance family.

Remark 1: the hard/soft partition is a property of the instance family, not of
the method. A profile gives every constraint type in
:data:`nrrp.constraints.CONSTRAINTS` exactly one role (hard, penalised with a
weight per violation unit, or off), and two sets follow from it:

* ``pre`` = H^pre, hard and prefix-monotone, enforced by the mask (32) (Prop. 3);
* ``ter`` = H^ter, hard and not prefix-monotone, left to the repair operator.

Penalised constraints enter the objective J with their weights. Constraint (3),
at most one shift per day, is not listed: the roster representation enforces it
and the environment masks employees already assigned on the current day.

Training and evaluation may use different profiles. Evaluation uses the
family's published partition; training may penalise the H^ter constraints as
reward shaping, which changes the reward the policy learns from but not the
problem it is evaluated on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constraints import ALWAYS_HARD, CONSTRAINTS


@dataclass(frozen=True)
class ConstraintProfile:
    name: str
    hard: tuple = ()
    penalised: tuple = ()  # (name, weight) pairs; a dict is accepted
    off: tuple = ()
    note: str = field(default="", compare=False)

    def __post_init__(self):
        pen = self.penalised.items() if isinstance(self.penalised, dict) else self.penalised
        object.__setattr__(self, "penalised", tuple((str(n), float(w)) for n, w in pen))
        object.__setattr__(self, "hard", tuple(self.hard))
        object.__setattr__(self, "off", tuple(self.off))
        roles = list(self.hard) + [n for n, _ in self.penalised] + list(self.off)
        unknown = sorted(set(roles) - set(CONSTRAINTS))
        if unknown:
            raise ValueError(f"profile {self.name}: unknown constraint(s) {unknown}")
        twice = sorted({n for n in roles if roles.count(n) > 1})
        if twice:
            raise ValueError(f"profile {self.name}: constraint(s) given more than one role: {twice}")
        missing = [n for n in CONSTRAINTS if n not in roles]
        if missing:
            raise ValueError(f"profile {self.name}: no role for {missing}")
        for n in ALWAYS_HARD:
            if n not in self.hard:
                raise ValueError(f"profile {self.name}: {n} must be hard")
        for n, w in self.penalised:
            if not (w > 0 and math.isfinite(w)):
                raise ValueError(f"profile {self.name}: weight of {n} must be positive, got {w}")

    @property
    def pre(self) -> tuple:
        """H^pre: hard and prefix-monotone, in registry order. Enforced by the mask."""
        return tuple(c for n, c in CONSTRAINTS.items() if n in self.hard and c.prefix_monotone)

    @property
    def ter(self) -> tuple:
        """H^ter: hard and not prefix-monotone. Left to the repair operator."""
        return tuple(c for n, c in CONSTRAINTS.items() if n in self.hard and not c.prefix_monotone)

    @property
    def penalties(self) -> tuple:
        """(constraint, weight) for every penalised constraint, in registry order."""
        w = dict(self.penalised)
        return tuple((c, w[n]) for n, c in CONSTRAINTS.items() if n in w)

    def with_penalised(self, weights: dict, name: str | None = None) -> "ConstraintProfile":
        """A copy in which the given constraints are penalised with the given weights."""
        hard = tuple(n for n in self.hard if n not in weights)
        off = tuple(n for n in self.off if n not in weights)
        pen = {**{n: w for n, w in self.penalised if n not in weights}, **weights}
        return ConstraintProfile(name or f"{self.name}+penalised", hard=hard, penalised=pen, off=off)

    def describe(self) -> str:
        names = lambda cs: ", ".join(c.name for c in cs) or "-"  # noqa: E731
        pen = ", ".join(f"{c.name} (w={w:g} per {c.unit})" for c, w in self.penalties) or "-"
        return (f"profile {self.name}\n  H^pre (hard, prefix-monotone, masked): {names(self.pre)}\n"
                f"  H^ter (hard, not prefix-monotone, repaired): {names(self.ter)}\n"
                f"  penalised: {pen}\n  off: {', '.join(self.off) or '-'}")


BURKE_CURTOIS = ConstraintProfile(
    "burke_curtois",
    hard=tuple(n for n in CONSTRAINTS if n != "min_coverage"),
    off=("min_coverage",),
    note="Format guide of schedulingbenchmarks.org/nrp: every constraint is hard and cover is "
         "soft only, so (4) is vacuous. The evaluation profile for this family.")


def burke_curtois_training(*, w_min_minutes: float, w_min_consecutive: float) -> ConstraintProfile:
    """Training profile for the Burke & Curtois family: H^ter penalised as reward shaping.

    The weights are hyperparameters, reported in Section 5; there are no defaults
    on purpose.
    """
    return BURKE_CURTOIS.with_penalised(
        {"min_total_minutes": w_min_minutes, "min_consecutive_shifts": w_min_consecutive},
        name="burke_curtois_training")


PROFILES = {p.name: p for p in (BURKE_CURTOIS,)}
