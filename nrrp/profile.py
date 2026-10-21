"""Constraint profile: which constraints are hard, which penalised, for one instance family.

Section 3.2: the hard/soft partition is a property of the instance family, not
of the method. A profile assigns every constraint type in
:data:`nrrp.constraints.CONSTRAINTS` exactly one role (hard, penalised with a
weight per violation unit, or off), and two sets follow from it:

* ``masked`` = hard and prefix-monotone, enforced by the action mask (Prop. 3);
* ``ter``    = hard and not prefix-monotone, the H^ter left to the repair step.

Penalised constraints enter the objective J with their weights. "At most one
shift per employee and day" is not listed: the roster representation enforces
it, and the environment masks employees already assigned on the current day.
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
    def masked(self) -> tuple:
        """Hard and prefix-monotone, in registry order: enforced by the mask."""
        return tuple(c for n, c in CONSTRAINTS.items() if n in self.hard and c.prefix_monotone)

    @property
    def ter(self) -> tuple:
        """H^ter: hard and not prefix-monotone, left to the repair step."""
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
        return (f"profile {self.name}\n  mask  (hard, prefix-monotone): {names(self.masked)}\n"
                f"  H^ter (hard, not prefix-monotone): {names(self.ter)}\n  penalised: {pen}\n"
                f"  off: {', '.join(self.off) or '-'}")


BURKE_CURTOIS = ConstraintProfile(
    "burke_curtois", hard=tuple(CONSTRAINTS),
    note="Format guide of schedulingbenchmarks.org/nrp: every constraint is hard; "
         "the objective is shift requests plus cover.")

PROFILES = {p.name: p for p in (BURKE_CURTOIS,)}
