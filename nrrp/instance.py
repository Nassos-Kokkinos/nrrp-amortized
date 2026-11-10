"""Instance and context (the paper's phi = (Inst, x°, xi)).

Conventions used throughout the package:

* employees i = 0..n-1, days d = 0..m-1, shift types s = 0..k-1. Days are
  0-indexed, so the paper's day d is day d-1 here, and construction, d_xi = 1
  in the paper, is ``d_xi = 0``;
* a roster is an int8 array of shape (n, m), called ``roster`` in the code,
  holding the shift worked by employee i on day d, or ``OFF`` (-1, the paper's
  day-off symbol 0). The paper's binary variables are x_{ids} = [roster[i, d] == s]
  (:func:`roster_to_binary`); its y_{i,d} is ``roster[i, d] != OFF``. Storing
  one shift per (i, d) makes constraint (3) hold by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

OFF = -1


def roster_to_binary(roster: np.ndarray, n_shifts: int) -> np.ndarray:
    """x_{ids} = [roster_{id} = s], shape (n, m, k), dtype bool."""
    return np.asarray(roster)[:, :, None] == np.arange(n_shifts)[None, None, :]


def binary_to_roster(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`roster_to_binary`; refuses two shifts on one day."""
    x = np.asarray(x, dtype=bool)
    if (x.sum(axis=2) > 1).any():
        raise ValueError("more than one shift for an employee on a day")
    roster = np.full(x.shape[:2], OFF, dtype=np.int8)
    i, d, s = np.nonzero(x)
    roster[i, d] = s
    return roster


def empty_roster(n_emp: int, n_days: int) -> np.ndarray:
    return np.full((n_emp, n_days), OFF, dtype=np.int8)


def _arr(a, dtype, shape, name):
    a = np.array(a, dtype=dtype)  # always a copy: instances never alias caller arrays
    if a.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    return a


@dataclass(eq=False)
class Instance:
    """Static data of one instance (Table 1 of the paper).

    Arrays (n employees, m days, k shift types, K skills):

    ==============  =========  =================================================  =========
    field           shape      meaning                                            paper
    ==============  =========  =================================================  =========
    shift_len       (k,)       shift duration in minutes                          l_s
    forbidden       (k, k)     forbidden[a, b]: b may not follow a the next day   F
    max_shifts      (n, k)     cap on assignments of each shift type              N^max_{i,s}
    max_minutes     (n,)       upper working-time bound, minutes                  H^max_i
    min_minutes     (n,)       lower working-time bound, minutes                  H^min_i
    max_consec      (n,)       maximum consecutive working days                   C^max_i
    min_consec      (n,)       minimum consecutive working days                   C^min_i
    min_consec_off  (n,)       minimum consecutive days off                       R^min_i
    max_weekends    (n,)       cap on worked weekends                             -
    day_off         (n, m)     unavailable employee-day pairs                     U
    on_req          (n, m, k)  weight of a request to work s on d (0 = none)      w^+
    off_req         (n, m, k)  weight of a request not to work s on d             w^-
    rho             (m, k)     preferred coverage                                 rho_{d,s}
    w_under         (m, k)     weight per missing employee                        w^u
    w_over          (m, k)     weight per surplus employee                        w^o
    min_cover       (m, k)     minimum coverage (zeros: none)                     r_{d,s}
    skills_held     (n, K)     skills held by each employee                       Q_i
    skill_required  (k,)       skill required by each shift, -1 for none          kappa_s
    ==============  =========  =================================================  =========

    Whether a constraint is hard or penalised is not part of the instance: a
    :class:`nrrp.profile.ConstraintProfile` declares it (Remark 1).
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
    min_cover: np.ndarray | None = None
    skills_held: np.ndarray | None = None
    skill_required: np.ndarray | None = None

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
        self.min_cover = (np.zeros((D, S), np.int64) if self.min_cover is None
                          else _arr(self.min_cover, np.int64, (D, S), "min_cover"))
        self.skill_required = (np.full(S, -1, np.int64) if self.skill_required is None
                               else _arr(self.skill_required, np.int64, (S,), "skill_required"))
        if self.skills_held is None:
            self.skills_held = np.zeros((I, 0), bool)
        self.skills_held = np.array(self.skills_held, dtype=bool)
        if self.skills_held.ndim != 2 or self.skills_held.shape[0] != I:
            raise ValueError(f"skills_held: expected shape ({I}, K), got {self.skills_held.shape}")
        self.validate()

    @property
    def n_emp(self) -> int:
        return len(self.employee_ids)

    @property
    def n_shifts(self) -> int:
        return len(self.shift_ids)

    @property
    def n_skills(self) -> int:
        return self.skills_held.shape[1]

    @property
    def size(self) -> str:
        return f"{self.n_emp} emp x {self.n_days} days x {self.n_shifts} shifts"

    @cached_property
    def weekend_of_day(self) -> np.ndarray:
        """(m,) index of the weekend a Saturday or Sunday belongs to, -1 on weekdays."""
        k = np.arange(self.n_days) + self.first_weekday
        return np.where(k % 7 >= 5, k // 7, -1).astype(np.int64)

    @cached_property
    def eligible(self) -> np.ndarray:
        """(n, k) True iff employee i holds the skill shift s requires (kappa_s in Q_i)."""
        req = self.skill_required
        held = self.skills_held[:, np.maximum(req, 0)] if self.n_skills else np.ones((self.n_emp, len(req)), bool)
        return np.where(req[None, :] < 0, True, held)

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
                  "min_consec_off", "max_weekends", "rho", "min_cover"):
            if (getattr(self, f) < 0).any():
                raise ValueError(f"{f} must be non-negative")
        for f in ("on_req", "off_req", "w_under", "w_over"):
            if (getattr(self, f) < 0).any():
                raise ValueError(f"{f} must be non-negative")
        req = self.skill_required
        if ((req < -1) | (req >= self.n_skills)).any():
            raise ValueError("skill_required refers to an unknown skill")
        if not 0 <= self.first_weekday <= 6:
            raise ValueError("first_weekday must be in 0..6")


@dataclass(eq=False)
class Context:
    """A context phi = (Inst, x°, xi) with xi = (A_xi, d_xi) (Definition 1, Eq. (14)).

    * ``published`` is x°, the published roster, as a roster array; days
      d < d_xi are frozen to it (the history constraint of Eq. (15));
    * ``absent`` (n, m) is A_xi, allowed only on d >= d_xi;
    * ``mu`` weighs the instability Delta(x, x°).

    Absences are credited (decision of 10/11/2026), as sick leave counts
    towards contractual hours:

    * ``credit`` (n,): the minutes of the shifts x° had on the absent days of
      each employee; the lower bound of (5) becomes ``min_minutes`` =
      max(H^min_i - credit_i, 0);
    * an absent day acts like an end of the horizon for the minimum stretch (9):
      a work stretch next to an absence is exempt, as one touching day 1 or
      day m already is.

    Construction is the special case d_xi = 0, A_xi empty, mu = 0, and the
    empty published roster (the default).
    """

    instance: Instance
    d_xi: int = 0
    published: np.ndarray | None = None
    absent: np.ndarray | None = None
    mu: float = 0.0
    name: str = ""

    def __post_init__(self):
        inst = self.instance
        I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
        self.d_xi = int(self.d_xi)
        if not 0 <= self.d_xi < D:
            raise ValueError(f"d_xi={self.d_xi} outside 0..{D - 1}")
        if self.published is None:
            if self.d_xi > 0 or self.mu != 0:
                raise ValueError("re-rostering (d_xi > 0 or mu > 0) needs a published roster")
            self.published = empty_roster(I, D)
        self.published = _arr(self.published, np.int8, (I, D), "published")
        if ((self.published < OFF) | (self.published >= S)).any():
            raise ValueError("the published roster holds an invalid shift index")
        self.absent = (np.zeros((I, D), bool) if self.absent is None
                       else _arr(self.absent, bool, (I, D), "absent"))
        if self.absent[:, : self.d_xi].any():
            raise ValueError("a disruption cannot mark days before d_xi")
        lost = self.absent & (self.published != OFF)            # shifts of x° the absences cancel
        self.credit = np.where(lost, inst.shift_len[np.where(lost, self.published, 0)], 0).sum(axis=1)
        self.mu = float(self.mu)
        if self.mu < 0:
            raise ValueError("mu must be non-negative")
        if not self.name:
            self.name = f"{inst.name}@d{self.d_xi}"

    @property
    def min_minutes(self) -> np.ndarray:
        """(n,) the lower bound of (5) in this context: H^min_i less the credited absences."""
        return np.maximum(self.instance.min_minutes - self.credit, 0)

    @property
    def is_construction(self) -> bool:
        return self.d_xi == 0 and self.mu == 0 and not self.absent.any()
