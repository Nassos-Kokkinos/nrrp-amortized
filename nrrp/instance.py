"""Instance and context (the paper's phi).

Conventions used throughout the package:

* employees i = 0..I-1, days d = 0..D-1, shift types s = 0..S-1;
* days are 0-indexed, so construction, the paper's d_xi = 1, is ``d_xi = 0`` here;
* a roster is an int8 array ``y`` of shape (I, D) holding the shift worked by
  employee i on day d, or ``OFF`` (-1). The paper's binary variables are
  x_{ids} = [y_{id} = s] (:func:`roster_to_binary`). Storing one shift per (i, d)
  makes "at most one shift per employee and day" hold by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

OFF = -1


def roster_to_binary(y: np.ndarray, n_shifts: int) -> np.ndarray:
    """x_{ids} = [y_{id} = s], shape (I, D, S), dtype bool."""
    return np.asarray(y)[:, :, None] == np.arange(n_shifts)[None, None, :]


def binary_to_roster(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`roster_to_binary`; refuses two shifts on one day."""
    x = np.asarray(x, dtype=bool)
    if (x.sum(axis=2) > 1).any():
        raise ValueError("more than one shift for an employee on a day")
    y = np.full(x.shape[:2], OFF, dtype=np.int8)
    i, d, s = np.nonzero(x)
    y[i, d] = s
    return y


def empty_roster(n_emp: int, n_days: int) -> np.ndarray:
    return np.full((n_emp, n_days), OFF, dtype=np.int8)


def _arr(a, dtype, shape, name):
    a = np.array(a, dtype=dtype)  # always a copy: instances never alias caller arrays
    if a.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    return a


@dataclass(eq=False)
class Instance:
    """Static data of one instance, in the Burke & Curtois data model.

    Arrays (I employees, D days, S shift types):

    ==============  =========  ==================================================
    shift_len       (S,)       shift length in minutes
    forbidden       (S, S)     forbidden[a, b]: shift b may not be worked the day
                               after shift a
    max_shifts      (I, S)     maximum number of shifts of each type
    max_minutes     (I,)       maximum total minutes over the horizon
    min_minutes     (I,)       minimum total minutes over the horizon
    max_consec      (I,)       maximum consecutive working days
    min_consec      (I,)       minimum consecutive working days
    min_consec_off  (I,)       minimum consecutive days off
    max_weekends    (I,)       maximum number of worked weekends
    day_off         (I, D)     fixed days off
    on_req          (I, D, S)  weight of the request to work s on d (0 = none)
    off_req         (I, D, S)  weight of the request not to work s on d
    rho             (D, S)     cover requirement
    w_under         (D, S)     weight per missing employee
    w_over          (D, S)     weight per surplus employee
    ==============  =========  ==================================================

    Whether a constraint is hard or penalised is not part of the instance: it
    is declared by a :class:`nrrp.profile.ConstraintProfile`.
    """

    name: str
    employee_ids: tuple
    shift_ids: tuple
    n_days: int
    shift_len: np.ndarray
    forbidden: np.ndarray
    max_shifts: np.ndarray
    max_minutes: np.ndarray
    min_minutes: np.ndarray
    max_consec: np.ndarray
    min_consec: np.ndarray
    min_consec_off: np.ndarray
    max_weekends: np.ndarray
    day_off: np.ndarray
    on_req: np.ndarray
    off_req: np.ndarray
    rho: np.ndarray
    w_under: np.ndarray
    w_over: np.ndarray
    first_weekday: int = 0  # 0 = Monday; all Burke & Curtois instances start on a Monday

    def __post_init__(self):
        self.employee_ids = tuple(self.employee_ids)
        self.shift_ids = tuple(self.shift_ids)
        I, D, S = len(self.employee_ids), int(self.n_days), len(self.shift_ids)
        self.n_days = D
        self.shift_len = _arr(self.shift_len, np.int64, (S,), "shift_len")
        self.forbidden = _arr(self.forbidden, bool, (S, S), "forbidden")
        self.max_shifts = _arr(self.max_shifts, np.int64, (I, S), "max_shifts")
        for f in ("max_minutes", "min_minutes", "max_consec", "min_consec",
                  "min_consec_off", "max_weekends"):
            setattr(self, f, _arr(getattr(self, f), np.int64, (I,), f))
        self.day_off = _arr(self.day_off, bool, (I, D), "day_off")
        self.on_req = _arr(self.on_req, np.float64, (I, D, S), "on_req")
        self.off_req = _arr(self.off_req, np.float64, (I, D, S), "off_req")
        self.rho = _arr(self.rho, np.int64, (D, S), "rho")
        self.w_under = _arr(self.w_under, np.float64, (D, S), "w_under")
        self.w_over = _arr(self.w_over, np.float64, (D, S), "w_over")
        self.validate()

    @property
    def n_emp(self) -> int:
        return len(self.employee_ids)

    @property
    def n_shifts(self) -> int:
        return len(self.shift_ids)

    @property
    def size(self) -> str:
        return f"{self.n_emp} emp x {self.n_days} days x {self.n_shifts} shifts"

    @cached_property
    def weekend_of_day(self) -> np.ndarray:
        """(D,) index of the weekend a Saturday or Sunday belongs to, -1 on weekdays."""
        k = np.arange(self.n_days) + self.first_weekday
        return np.where(k % 7 >= 5, k // 7, -1).astype(np.int64)

    def validate(self) -> None:
        if self.n_emp == 0 or self.n_shifts == 0 or self.n_days == 0:
            raise ValueError("empty instance")
        if len(set(self.employee_ids)) != self.n_emp:
            raise ValueError("duplicate employee ids")
        if len(set(self.shift_ids)) != self.n_shifts:
            raise ValueError("duplicate shift ids")
        if (self.shift_len <= 0).any():
            raise ValueError("shift lengths must be positive")
        if (self.min_minutes > self.max_minutes).any():
            raise ValueError("min_minutes > max_minutes for some employee (column order?)")
        if (self.min_consec > self.max_consec).any():
            raise ValueError("min_consec > max_consec for some employee (column order?)")
        for f in ("max_shifts", "max_minutes", "min_minutes", "max_consec", "min_consec",
                  "min_consec_off", "max_weekends", "rho"):
            if (getattr(self, f) < 0).any():
                raise ValueError(f"{f} must be non-negative")
        for f in ("on_req", "off_req", "w_under", "w_over"):
            if (getattr(self, f) < 0).any():
                raise ValueError(f"{f} must be non-negative")
        if not 0 <= self.first_weekday <= 6:
            raise ValueError("first_weekday must be in 0..6")


@dataclass(eq=False)
class Context:
    """A context phi = (instance, published roster x°, disruption xi, d_xi, mu).

    * days d < d_xi are frozen to ``y0``: the history no decision can change;
    * ``absent`` (I, D) holds the disruption's unavailabilities, only on d >= d_xi;
    * ``rho`` is the cover requirement after the disruption (default: the
      instance's); it may differ from the instance's only on d >= d_xi;
    * ``mu`` weighs stability, the Hamming distance to ``y0`` on d >= d_xi.

    Construction is the special case d_xi = 0, no disruption, mu = 0, and
    ``y0`` the empty roster (the default).
    """

    instance: Instance
    d_xi: int = 0
    y0: np.ndarray | None = None
    absent: np.ndarray | None = None
    rho: np.ndarray | None = None
    mu: float = 0.0
    name: str = ""

    def __post_init__(self):
        inst = self.instance
        I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
        self.d_xi = int(self.d_xi)
        if not 0 <= self.d_xi < D:
            raise ValueError(f"d_xi={self.d_xi} outside 0..{D - 1}")
        if self.y0 is None:
            if self.d_xi > 0 or self.mu != 0:
                raise ValueError("re-rostering (d_xi > 0 or mu > 0) needs a published roster y0")
            self.y0 = empty_roster(I, D)
        self.y0 = _arr(self.y0, np.int8, (I, D), "y0")
        if ((self.y0 < OFF) | (self.y0 >= S)).any():
            raise ValueError("y0 holds an invalid shift index")
        self.absent = (np.zeros((I, D), bool) if self.absent is None
                       else _arr(self.absent, bool, (I, D), "absent"))
        if self.absent[:, : self.d_xi].any():
            raise ValueError("a disruption cannot mark days before d_xi")
        self.rho = (inst.rho.copy() if self.rho is None
                    else _arr(self.rho, np.int64, (D, S), "rho"))
        if (self.rho[: self.d_xi] != inst.rho[: self.d_xi]).any():
            raise ValueError("cover before d_xi must equal the instance's")
        if (self.rho < 0).any():
            raise ValueError("negative cover requirement")
        self.mu = float(self.mu)
        if self.mu < 0:
            raise ValueError("mu must be non-negative")
        if not self.name:
            self.name = f"{inst.name}@d{self.d_xi}"

    @property
    def is_construction(self) -> bool:
        return self.d_xi == 0 and self.mu == 0 and not self.absent.any() \
            and bool((self.rho == self.instance.rho).all())
