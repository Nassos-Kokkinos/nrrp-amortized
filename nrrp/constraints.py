"""Constraint types: the hard constraints (3)-(10) of the paper, the minimum-rest
constraint, and the weekend cap.

Each row type (a constraint on one employee's row) defines three things, side by side:

``violation_row(i, row, ctx)``
    The reference definition: the violation amount of employee i's row (shift
    index per day, or OFF), in the unit given by ``unit``; 0 iff satisfied.
    Rosters may be partial: undecided slots read as OFF.

``prefix_monotone``
    True iff every truncation of a satisfying roster under the environment's
    day-major slot order (undecided slots empty) still satisfies the
    constraint (equivalently, Definition 2: a violated partial roster has no
    satisfying completion). Lemma 1 sorts the constraints by this property; the
    hard prefix-monotone ones form H^pre and are enforced by the mask.
    :func:`find_truncation_counterexample` tests the flag.

``delta(st, d, s)``
    The change in each employee's violation if that employee were assigned
    shift s on the current day d of a partial roster in day-major order, where
    days < d are decided, day d is still empty for the employee, and days > d
    are empty. Vectorised over employees; ``s`` is a shift index or an (I,)
    array of them. It serves twice: for a penalised constraint it is the cost
    increment the reward charges; for a hard prefix-monotone constraint the
    mask is ``delta == 0`` (the prefix satisfies it, so the assignment keeps it
    satisfied iff the increment is zero).

Minimum coverage (4) is a column constraint, on the staffing of a (d, s) pair;
it has ``scope = "column"``, a (m, k) violation array, and per-fill increments
instead of a per-employee delta. It is not prefix-monotone.

``st`` is any object with the attributes of :class:`RosterState`, which defines
them from a partial roster. The environment maintains the same attributes
incrementally; the smoke test checks the two agree.

Boundary conventions (format guide, schedulingbenchmarks.org/nrp): for shift
rotation and maximum consecutive shifts, the day before the horizon and the
day after it are days off. For minimum consecutive shifts and minimum
consecutive days off, an unbounded run is assumed on both sides of the
horizon, so a run touching either end is exempt.
"""
from __future__ import annotations

import numpy as np

from .instance import OFF, Context


def true_runs(flags) -> tuple[np.ndarray, np.ndarray]:
    """Maximal runs of True: (starts, stops), stop exclusive."""
    f = np.concatenate(([0], np.asarray(flags, dtype=np.int8), [0]))
    dif = np.diff(f)
    return np.flatnonzero(dif == 1), np.flatnonzero(dif == -1)


def trailing_run(flags: np.ndarray, d: int) -> np.ndarray:
    """(I,) length of the run of True ending at day d-1 in each row of an (I, D) array."""
    if d == 0:
        return np.zeros(flags.shape[0], np.int64)
    rev = flags[:, :d][:, ::-1]
    first_false = np.argmax(~rev, axis=1)
    return np.where(rev.all(axis=1), d, first_false).astype(np.int64)


class RosterState:
    """The state the deltas read, defined from a partial roster ``y`` at day ``d``.

    ==============  =======  ===================================================
    prev            (I,)     shift worked on d-1 (OFF on day 0)
    count           (I, S)   shifts of each type in the roster so far
    minutes         (I,)     minutes worked so far
    run_work        (I,)     consecutive working days ending at d-1
    run_off         (I,)     consecutive days off ending at d-1
    weekends        (I,)     weekends with at least one worked day so far
    last_wk         (I,)     index of the last such weekend, -1 if none
    forbidden_ext   (S+1,S)  ``forbidden`` with a row of False appended, so that
                             indexing with prev = OFF = -1 forbids nothing
    ==============  =======  ===================================================
    """

    def __init__(self, ctx: Context, roster: np.ndarray, d: int):
        inst = ctx.instance
        I, S = inst.n_emp, inst.n_shifts
        self.inst, self.ctx, self.d = inst, ctx, d
        self.rows = np.arange(I)
        self.forbidden_ext = extended_forbidden(inst.forbidden)
        roster = np.asarray(roster)
        w = roster != OFF
        self.prev = roster[:, d - 1].astype(np.int64) if d > 0 else np.full(I, OFF, np.int64)
        self.count = (roster[:, :, None] == np.arange(S)).sum(axis=1).astype(np.int64)
        self.minutes = np.where(w, inst.shift_len[np.where(w, roster, 0)], 0).sum(axis=1).astype(np.int64)
        self.run_work = trailing_run(w, d)
        self.run_off = trailing_run(~w, d)
        wk = inst.weekend_of_day
        self.weekends = np.zeros(I, np.int64)
        self.last_wk = np.full(I, -1, np.int64)
        for i in range(I):
            worked = np.unique(wk[w[i] & (wk >= 0)])
            self.weekends[i] = len(worked)
            if len(worked):
                self.last_wk[i] = worked[-1]


def extended_forbidden(forbidden: np.ndarray) -> np.ndarray:
    return np.vstack([forbidden, np.zeros((1, forbidden.shape[1]), bool)])


class Constraint:
    name = ""
    paper = ""          # where the paper states it
    scope = "row"
    prefix_monotone = False
    unit = ""
    why = ""            # one-line argument for the prefix_monotone flag

    def violation_row(self, i: int, row: np.ndarray, ctx: Context) -> float:
        raise NotImplementedError

    def violation(self, roster: np.ndarray, ctx: Context) -> np.ndarray:
        """(n,) violation amount of every row."""
        return np.array([self.violation_row(i, roster[i], ctx) for i in range(roster.shape[0])], dtype=float)

    def delta(self, st, d: int, s) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


class ShiftRotation(Constraint):
    name, prefix_monotone, unit = "shift_rotation", True, "forbidden successions"
    paper = "(7)"
    why = "removing assignments cannot create a forbidden pair of consecutive shifts"

    def violation_row(self, i, row, ctx):
        a, b = row[:-1], row[1:]
        both = (a != OFF) & (b != OFF)
        return float(np.count_nonzero(ctx.instance.forbidden[a[both], b[both]]))

    def delta(self, st, d, s):
        return st.forbidden_ext[st.prev, s].astype(float)


class MaxShiftsPerType(Constraint):
    name, prefix_monotone, unit = "max_shifts_per_type", True, "shifts"
    paper = "(10)"
    why = "upper bound on a count"

    def violation_row(self, i, row, ctx):
        inst = ctx.instance
        cnt = np.bincount(row[row != OFF].astype(np.int64), minlength=inst.n_shifts)
        return float(np.maximum(cnt - inst.max_shifts[i], 0).sum())

    def delta(self, st, d, s):
        return (st.count[st.rows, s] >= st.inst.max_shifts[st.rows, s]).astype(float)


class MaxTotalMinutes(Constraint):
    name, prefix_monotone, unit = "max_total_minutes", True, "minutes"
    paper = "(5), upper bound"
    why = "upper bound on a sum"

    def violation_row(self, i, row, ctx):
        inst = ctx.instance
        total = int(inst.shift_len[row[row != OFF]].sum())
        return float(max(total - int(inst.max_minutes[i]), 0))

    def delta(self, st, d, s):
        m, cap, L = st.minutes, st.inst.max_minutes, st.inst.shift_len[s]
        return (np.maximum(m + L - cap, 0) - np.maximum(m - cap, 0)).astype(float)


class MinTotalMinutes(Constraint):
    name, prefix_monotone, unit = "min_total_minutes", False, "minutes"
    paper = "(5), lower bound"
    why = "lower bound on a sum: a truncation removes minutes"

    def violation_row(self, i, row, ctx):
        inst = ctx.instance
        total = int(inst.shift_len[row[row != OFF]].sum())
        return float(max(int(inst.min_minutes[i]) - total, 0))

    def delta(self, st, d, s):
        m, floor, L = st.minutes, st.inst.min_minutes, st.inst.shift_len[s]
        return (np.maximum(floor - m - L, 0) - np.maximum(floor - m, 0)).astype(float)


class MaxConsecutiveShifts(Constraint):
    name, prefix_monotone, unit = "max_consecutive_shifts", True, "days over the limit"
    paper = "(8)"
    why = "a truncation can only shorten work runs"

    def violation_row(self, i, row, ctx):
        m = int(ctx.instance.max_consec[i])
        a, b = true_runs(row != OFF)
        return float(np.maximum(b - a - m, 0).sum())

    def delta(self, st, d, s):
        # day d+1 is empty, so the run ending at d-1 grows by one and merges with nothing
        return (st.run_work >= st.inst.max_consec).astype(float)


class MinConsecutiveShifts(Constraint):
    name, prefix_monotone, unit = "min_consecutive_shifts", False, "days short"
    paper = "(9)"
    why = "cutting inside a work run leaves a short run that no longer touches the end"

    def violation_row(self, i, row, ctx):
        m, D = int(ctx.instance.min_consec[i]), len(row)
        a, b = true_runs(row != OFF)
        interior = (a > 0) & (b < D)
        return float(np.maximum(m - (b - a), 0)[interior].sum())

    def delta(self, st, d, s):
        r, m, D = st.run_work, st.inst.min_consec, st.inst.n_days
        # before: the run [d-r, d) is followed by the empty day d
        before = np.where((r > 0) & (r < d), np.maximum(m - r, 0), 0)
        # after: the run [d-r, d+1); interior iff it starts after day 0 and d+1 < D
        after = np.where((r < d) & (d < D - 1), np.maximum(m - r - 1, 0), 0)
        return (after - before).astype(float)


class MinConsecutiveDaysOff(Constraint):
    name, prefix_monotone, unit = "min_consecutive_days_off", True, "days short"
    paper = "(9) with 1 - y, parameter R^min"
    why = ("day-major order plus the end-of-horizon exemption: a truncation turns the run "
           "containing the cut into a trailing off-run, which is exempt; earlier runs are unchanged")

    def violation_row(self, i, row, ctx):
        m, D = int(ctx.instance.min_consec_off[i]), len(row)
        a, b = true_runs(row == OFF)
        interior = (a > 0) & (b < D)
        return float(np.maximum(m - (b - a), 0)[interior].sum())

    def delta(self, st, d, s):
        # assigning d closes the off-run [d-r, d); it is exempt if empty or if it starts at day 0
        r, m = st.run_off, st.inst.min_consec_off
        return np.where((r > 0) & (r < d), np.maximum(m - r, 0), 0).astype(float)


class MaxWeekends(Constraint):
    name, prefix_monotone, unit = "max_weekends", True, "weekends"
    paper = "weekend cap (Remark 1)"
    why = "upper bound on the number of worked weekends"

    def violation_row(self, i, row, ctx):
        wk = ctx.instance.weekend_of_day
        worked = np.unique(wk[(row != OFF) & (wk >= 0)])
        return float(max(len(worked) - int(ctx.instance.max_weekends[i]), 0))

    def delta(self, st, d, s):
        w = st.inst.weekend_of_day[d]
        if w < 0:
            return np.zeros(len(st.rows))
        return ((st.last_wk != w) & (st.weekends >= st.inst.max_weekends)).astype(float)


class DaysOff(Constraint):
    name, prefix_monotone, unit = "days_off", True, "assignments"
    paper = "(6), U"
    why = "forbids individual assignments"

    def violation_row(self, i, row, ctx):
        return float(np.count_nonzero((row != OFF) & ctx.instance.day_off[i]))

    def delta(self, st, d, s):
        return st.inst.day_off[:, d].astype(float)


class Unavailability(Constraint):
    """Absences introduced by the disruption xi. Always hard."""
    name, prefix_monotone, unit = "unavailability", True, "assignments"
    paper = "(6), A_xi in U(phi)"
    why = "forbids individual assignments"

    def violation_row(self, i, row, ctx):
        return float(np.count_nonzero((row != OFF) & ctx.absent[i]))

    def delta(self, st, d, s):
        return st.ctx.absent[:, d].astype(float)


class Skills(Constraint):
    """kappa_s not in Q_i: an employee may not work a shift whose skill they lack."""
    name, prefix_monotone, unit = "skills", True, "assignments"
    paper = "(6), kappa_s not in Q_i"
    why = "forbids individual assignments"

    def violation_row(self, i, row, ctx):
        worked = row != OFF
        return float(np.count_nonzero(~ctx.instance.eligible[i, row[worked].astype(np.int64)]))

    def delta(self, st, d, s):
        return (~st.inst.eligible[st.rows, s]).astype(float)


class MinCoverage(Constraint):
    """sum_i x_{i,d,s} >= r_{d,s}, a constraint on a column (d, s), not on a row."""
    name, prefix_monotone, unit = "min_coverage", False, "employees short"
    paper = "(4)"
    scope = "column"
    why = "a truncation removes staff from a pair"

    def violation(self, roster, ctx):
        """(m, k) shortfall max(r - n, 0) of every pair."""
        inst = ctx.instance
        n = roster_counts(roster, inst.n_shifts)
        return np.maximum(inst.min_cover - n, 0).astype(float)

    def increments(self, ctx: Context, d: int, s: int, n_max: int) -> np.ndarray:
        """(n_max,) change in the shortfall of (d, s) when its staffing goes from f to f + 1."""
        r = int(ctx.instance.min_cover[d, s])
        f = np.arange(n_max + 1)
        return np.diff(np.maximum(r - f, 0)).astype(float)

    def violation_row(self, i, row, ctx):
        raise TypeError("min_coverage constrains a (d, s) pair, not a row")

    def delta(self, st, d, s):
        raise TypeError("min_coverage has per-fill increments, not a per-employee delta")


def roster_counts(roster: np.ndarray, n_shifts: int) -> np.ndarray:
    """(m, k) number of employees on each (d, s)."""
    roster = np.asarray(roster)
    return (roster[:, :, None] == np.arange(n_shifts)).sum(axis=0)


CONSTRAINTS: dict = {c.name: c for c in (
    ShiftRotation(), MaxShiftsPerType(), MaxTotalMinutes(), MinTotalMinutes(),
    MaxConsecutiveShifts(), MinConsecutiveShifts(), MinConsecutiveDaysOff(),
    MaxWeekends(), DaysOff(), Unavailability(), Skills(), MinCoverage(),
)}
ROW_CONSTRAINTS: dict = {n: c for n, c in CONSTRAINTS.items() if c.scope == "row"}
ALWAYS_HARD = ("unavailability",)


# ------------------------------------------------------ Lemma 1 as an executable test

def truncations(row: np.ndarray):
    """All truncations of one row under day-major order.

    Days before the cut day t are kept; day t is kept or not (the cut can fall
    before or after this employee's slot that day); later days are empty.
    """
    D = len(row)
    for t in range(D):
        for keep in (False, True):
            if keep and row[t] == OFF:
                continue
            r = np.full(D, OFF, np.int8)
            r[:t] = row[:t]
            if keep:
                r[t] = row[t]
            yield r


def random_block_row(D: int, S: int, rng: np.random.Generator, p_work: float = 0.6) -> np.ndarray:
    """A row of alternating work and off runs of random length 1..6."""
    row = np.full(D, OFF, np.int8)
    d, work = 0, rng.random() < p_work
    while d < D:
        L = int(rng.integers(1, 7))
        if work:
            same = rng.random() < 0.5
            s0 = int(rng.integers(S))
            for k in range(d, min(d + L, D)):
                row[k] = s0 if same else int(rng.integers(S))
        d += L
        work = (not work) if rng.random() < 0.85 else work
    return row


def find_truncation_counterexample(constraint: Constraint, ctx: Context, rng: np.random.Generator,
                                   n_rows: int = 2000):
    """Search for a satisfying row with a violating truncation.

    Returns (number of satisfying rows examined, counterexample or None); a
    counterexample is (employee, row, truncated row).
    """
    if constraint.scope != "row":
        raise ValueError(f"{constraint.name} is not a row constraint")
    inst = ctx.instance
    n_ok = 0
    for _ in range(n_rows):
        i = int(rng.integers(inst.n_emp))
        row = random_block_row(inst.n_days, inst.n_shifts, rng)
        if constraint.violation_row(i, row, ctx) > 0:
            continue
        n_ok += 1
        for r in truncations(row):
            if constraint.violation_row(i, r, ctx) > 0:
                return n_ok, (i, row, r)
    return n_ok, None
