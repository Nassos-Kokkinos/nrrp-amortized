"""CP-SAT model of the roster problem: construction (15) and, with a frozen prefix,
re-rostering (Definition 1). Needs OR-Tools (``pip install -e ".[cp]"``).

The model is a second statement of the same problem, in OR-Tools: variables
x_{ids}, constraint (3) as at-most-one per (i, d), and every constraint of
:data:`nrrp.constraints.CONSTRAINTS` in the role the profile gives it:

* hard: a constraint of the model;
* penalised: an auxiliary variable bounded below by the violation, in the unit
  of ``constraints.py``, charged w_c per unit;
* off: nothing.

The objective is J (``objective.py``): requests + cover + mu Delta + penalties.
Auxiliary variables are only bounded below, so the CP objective of a feasible
solution is >= J of its roster, with equality at an optimum (every auxiliary is
then tight) and whenever the solver has tightened them. The returned J is always
recomputed by ``objective.J``; tests check that the two agree.

Weights may be non-integer. CP-SAT needs integer coefficients, so all
coefficients are multiplied by the smallest power of ten (up to 10^6) that makes
them integral; the objective is divided back.

Uses: x° for instances without a published roster (Remark 5) and the baseline
T_solve; later the repair subproblem of Algorithm 1, through ``free`` (the cells
that may change, the rest fixed to ``base``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .constraints import CONSTRAINTS
from .instance import OFF, Context, roster_to_binary
from .objective import J as objective_J
from .objective import evaluate

MAX_SCALE_EXPONENT = 6


@dataclass
class CPResult:
    status: str                     # OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID, UNKNOWN
    roster: np.ndarray | None       # (n, m) int8, None without a solution
    objective: float | None         # CP objective, unscaled
    bound: float | None             # best lower bound on J proved by the solver
    J: float | None                 # objective.J of the roster under the model's profile
    feasible: bool | None           # every hard constraint of the profile holds (objective.evaluate)
    wall_time: float                # seconds in the solver (T_solve)
    build_time: float               # seconds to build the model
    scale: int
    info: dict = field(default_factory=dict)

    @property
    def has_solution(self) -> bool:
        return self.roster is not None

    @property
    def gap(self) -> float | None:
        if self.J is None or self.bound is None:
            return None
        return (self.J - self.bound) / max(1.0, abs(self.J))


def _scale_for(values) -> int:
    vals = [float(v) for v in values if v]
    for k in range(MAX_SCALE_EXPONENT + 1):
        s = 10 ** k
        if all(abs(v * s - round(v * s)) <= 1e-9 * max(1.0, abs(v * s)) for v in vals):
            return s
    raise ValueError(f"weights are not multiples of 1e-{MAX_SCALE_EXPONENT}; CP-SAT needs integers")


class RosterModel:
    """The CP-SAT model of one context under one profile.

    ``terms`` maps each term of J (names as in :func:`nrrp.objective.J_terms`) to a
    list of (coefficient, linear expression), and ``term_constants`` to the part of
    the term no variable changes; J is the sum of both over all terms.
    Coefficients are unscaled; the model's objective uses them multiplied by
    ``scale``.
    """

    def __init__(self, ctx: Context, profile, *, free: np.ndarray | None = None,
                 base: np.ndarray | None = None):
        t0 = time.perf_counter()
        self.ctx, self.profile = ctx, profile
        inst = ctx.instance
        self.inst = inst
        I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
        unknown = [c.name for c in CONSTRAINTS.values() if c.name not in ENCODERS]
        if unknown:
            raise NotImplementedError(f"no CP-SAT encoding for constraint(s) {unknown}")

        m = self.model = cp_model.CpModel()
        self.x = np.empty((I, D, S), dtype=object)
        for i in range(I):
            for d in range(D):
                for s in range(S):
                    self.x[i, d, s] = m.NewBoolVar(f"x[{i},{d},{s}]")
        # w[i, d] = y_{i,d}: works on d. Constraint (3): at most one shift per day.
        self.w = np.empty((I, D), dtype=object)
        for i in range(I):
            for d in range(D):
                self.w[i, d] = m.NewBoolVar(f"w[{i},{d}]")
                m.Add(sum(self.x[i, d, :]) == self.w[i, d])

        # variables fixed to a roster: the frozen history d < d_xi (fixed to x°) and, for
        # repair, every variable outside ``free`` (fixed to ``base``). ``free`` is either
        # (n, m), whole cells (all shifts of employee i on day d), or (n, m, k), single
        # variables x_{ids} (the slots left unfilled, V^∅ of Eq. (40), free one shift type)
        fixed = np.zeros((I, D, S), bool)
        fixed[:, : ctx.d_xi, :] = True
        target = roster_to_binary(ctx.published, S)
        if free is not None:
            free = np.asarray(free, bool)
            if free.shape == (I, D):
                free = np.repeat(free[:, :, None], S, axis=2)
            if free.shape != (I, D, S):
                raise ValueError(f"free: expected shape {(I, D)} or {(I, D, S)}, got {free.shape}")
            if base is None:
                raise ValueError("free needs base, the roster the other cells are fixed to")
            base = np.asarray(base)
            if base.shape != (I, D):
                raise ValueError(f"base: expected shape {(I, D)}, got {base.shape}")
            outside = ~free
            outside[:, : ctx.d_xi, :] = False
            fixed |= outside
            target = np.where(outside, roster_to_binary(base, S), target)
        for i, d, s in zip(*np.nonzero(fixed)):
            m.Add(self.x[i, d, s] == int(target[i, d, s]))
        self.fixed_vars = fixed                      # (n, m, k)
        self.fixed = fixed.all(axis=2)               # (n, m): cells with every variable fixed
        self.n_free = int((~fixed).sum())            # |V_w| when built for the repair

        self.terms: dict = {"requests": [], "cover": [], "stability": []}
        self.term_constants: dict = {"requests": 0.0, "cover": 0.0, "stability": 0.0}

        # requests: sum w^+ (1 - x) + w^- x = sum w^+ + sum (w^- - w^+) x
        self.term_constants["requests"] = float(inst.on_req.sum())
        p = inst.off_req - inst.on_req
        for i, d, s in zip(*np.nonzero(p)):
            self.terms["requests"].append((float(p[i, d, s]), self.x[i, d, s]))

        # cover: C_{ds}(n) = w^u max(rho - n, 0) + w^o max(n - rho, 0)
        for d in range(D):
            for s in range(S):
                wu, wo, rho = float(inst.w_under[d, s]), float(inst.w_over[d, s]), int(inst.rho[d, s])
                if not (wu or wo):
                    continue
                n = sum(self.x[:, d, s])
                if wu and rho > 0:
                    u = m.NewIntVar(0, rho, f"under[{d},{s}]")
                    m.Add(u >= rho - n)
                    self.terms["cover"].append((wu, u))
                if wo:
                    o = m.NewIntVar(0, I, f"over[{d},{s}]")
                    m.Add(o >= n - rho)
                    self.terms["cover"].append((wo, o))

        # stability: mu sum_{i, d >= d_xi, s} |x - x°|
        if ctx.mu:
            x0 = roster_to_binary(ctx.published, S)
            for i in range(I):
                for d in range(ctx.d_xi, D):
                    for s in range(S):
                        if x0[i, d, s]:
                            self.term_constants["stability"] += ctx.mu
                            self.terms["stability"].append((-ctx.mu, self.x[i, d, s]))
                        else:
                            self.terms["stability"].append((ctx.mu, self.x[i, d, s]))

        # constraints in their profile roles
        hard = {c.name for c in profile.pre + profile.ter}
        weights = {c.name: w for c, w in profile.penalties}
        for name, c in CONSTRAINTS.items():
            if name in hard:
                ENCODERS[name](self, None)
            elif name in weights:
                key = f"penalty:{name}"
                self.terms[key] = [(weights[name], e) for e in ENCODERS[name](self, True)]

        self.constant = sum(self.term_constants.values())
        coefs = [c for lst in self.terms.values() for c, _ in lst] + list(self.term_constants.values())
        self.scale = _scale_for(coefs)
        obj = []
        for lst in self.terms.values():
            for c, e in lst:
                obj.append(int(round(c * self.scale)) * e)
        m.Minimize(sum(obj) + int(round(self.constant * self.scale)))
        self.build_time = time.perf_counter() - t0

    # ------------------------------------------------------------------ helpers

    def penalty_var(self, expr_lower_bound, ub: int, name: str):
        """An integer v >= max(expr, 0), v <= ub."""
        v = self.model.NewIntVar(0, max(int(ub), 0), name)
        self.model.Add(v >= expr_lower_bound)
        return v

    def hint(self, roster: np.ndarray) -> None:
        """Start the search from a roster (a solution hint; need not be feasible)."""
        roster = np.asarray(roster)
        for (i, d, s), v in np.ndenumerate(self.x):
            self.model.AddHint(v, int(roster[i, d] == s))

    def roster_from(self, solver: cp_model.CpSolver) -> np.ndarray:
        I, D, S = self.x.shape
        r = np.full((I, D), OFF, np.int8)
        for (i, d, s), v in np.ndenumerate(self.x):
            if solver.BooleanValue(v):
                r[i, d] = s
        return r

    def term_values(self, solver: cp_model.CpSolver) -> dict:
        """Value of each term of J in the solver's solution (auxiliaries as the solver set them)."""
        return {k: float(sum(c * solver.Value(e) for c, e in lst)) + self.term_constants.get(k, 0.0)
                for k, lst in self.terms.items()}

    def solve(self, time_limit: float | None = None, workers: int = 8, seed: int = 0,
              log: bool = False, hint: np.ndarray | None = None,
              deterministic_limit: float | None = None) -> CPResult:
        """``time_limit``: wall-clock seconds. ``deterministic_limit``: CP-SAT's
        deterministic time, independent of the machine's clock and load; with one
        worker the run is then reproducible (used by the tests)."""
        if hint is not None:
            self.hint(hint)
        solver = cp_model.CpSolver()
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = float(time_limit)
        if deterministic_limit is not None:
            solver.parameters.max_deterministic_time = float(deterministic_limit)
        solver.parameters.num_workers = int(workers)
        solver.parameters.random_seed = int(seed)
        solver.parameters.log_search_progress = bool(log)
        t0 = time.perf_counter()
        code = solver.Solve(self.model)
        wall = time.perf_counter() - t0
        status = solver.StatusName(code)
        res = CPResult(status=status, roster=None, objective=None, bound=None, J=None, feasible=None,
                       wall_time=wall, build_time=self.build_time, scale=self.scale)
        if code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            res.roster = self.roster_from(solver)
            res.objective = solver.ObjectiveValue() / self.scale
            res.bound = solver.BestObjectiveBound() / self.scale
            res.J = objective_J(res.roster, self.ctx, self.profile)
            res.feasible = evaluate(res.roster, self.ctx, self.profile)["feasible"]
            res.info["terms"] = self.term_values(solver)
        res.info["solver_wall_time"] = solver.WallTime()
        det = getattr(solver, "deterministic_time", None)          # a property in OR-Tools >= 9.10
        res.info["deterministic_time"] = float(det() if callable(det) else det or 0.0)
        return res


def solve(ctx: Context, profile, *, time_limit: float | None = None, workers: int = 8, seed: int = 0,
          log: bool = False, hint: np.ndarray | None = None, free: np.ndarray | None = None,
          base: np.ndarray | None = None, deterministic_limit: float | None = None) -> CPResult:
    """Build and solve; see :class:`RosterModel` and :meth:`RosterModel.solve`."""
    return RosterModel(ctx, profile, free=free, base=base).solve(
        time_limit=time_limit, workers=workers, seed=seed, log=log, hint=hint,
        deterministic_limit=deterministic_limit)


# ---------------------------------------------------------------------------- encoders
#
# ENCODERS[name](model, penalised): with penalised None, add the constraint as hard;
# with penalised True, return the list of linear expressions whose sum is the
# violation (each bounded below by its share of it, tight at an optimum), in the
# unit of constraints.py. Boundary conventions as in constraints.py.

def _enc_shift_rotation(M: RosterModel, pen):
    inst, x, out = M.inst, M.x, []
    I, D, _ = x.shape
    pairs = list(zip(*np.nonzero(inst.forbidden)))
    for i in range(I):
        for d in range(D - 1):
            for a, b in pairs:
                if pen is None:
                    M.model.AddBoolOr([x[i, d, a].Not(), x[i, d + 1, b].Not()])
                else:
                    v = M.model.NewBoolVar(f"rot[{i},{d},{a},{b}]")
                    M.model.Add(v >= x[i, d, a] + x[i, d + 1, b] - 1)
                    out.append(v)
    return out


def _enc_max_shifts_per_type(M: RosterModel, pen):
    inst, x, out = M.inst, M.x, []
    I, D, S = x.shape
    for i in range(I):
        for s in range(S):
            cnt, cap = sum(x[i, :, s]), int(inst.max_shifts[i, s])
            if cap >= D:
                continue
            if pen is None:
                M.model.Add(cnt <= cap)
            else:
                out.append(M.penalty_var(cnt - cap, D - cap, f"maxshifts[{i},{s}]"))
    return out


def _minutes(M: RosterModel, i):
    L = M.inst.shift_len
    return sum(int(L[s]) * M.x[i, d, s] for d in range(M.x.shape[1]) for s in range(M.x.shape[2]))


def _enc_max_total_minutes(M: RosterModel, pen):
    inst, out = M.inst, []
    D, top = M.x.shape[1], int(M.inst.shift_len.max())
    for i in range(M.x.shape[0]):
        cap = int(inst.max_minutes[i])
        if pen is None:
            M.model.Add(_minutes(M, i) <= cap)
        else:
            out.append(M.penalty_var(_minutes(M, i) - cap, D * top, f"maxmin[{i}]"))
    return out


def _enc_min_total_minutes(M: RosterModel, pen):
    inst, out = M.inst, []
    for i in range(M.x.shape[0]):
        floor = int(inst.min_minutes[i])
        if floor <= 0:
            continue
        if pen is None:
            M.model.Add(_minutes(M, i) >= floor)
        else:
            out.append(M.penalty_var(floor - _minutes(M, i), floor, f"minmin[{i}]"))
    return out


def _enc_max_consecutive_shifts(M: RosterModel, pen):
    # violation = sum over work runs of max(len - C, 0) = number of days d whose
    # window d-C..d is all worked (days outside the horizon are off)
    inst, w, out = M.inst, M.w, []
    I, D = w.shape
    for i in range(I):
        C = int(inst.max_consec[i])
        for d in range(C, D):
            win = [w[i, k] for k in range(d - C, d + 1)]
            if pen is None:
                M.model.Add(sum(win) <= C)
            else:
                z = M.model.NewBoolVar(f"maxcons[{i},{d}]")
                M.model.Add(z >= sum(win) - C)
                out.append(z)
    return out


def _interior_run_patterns(M: RosterModel, i, m_min, work: bool, pen, tag):
    """Runs of worked (work=True) or off days of length L < m_min that start after day 0
    and end before the last day; each costs m_min - L (constraints (9), (10))."""
    w, out = M.w, []
    D = w.shape[1]
    on = (lambda v: v) if work else (lambda v: v.Not())    # literal "day is in the run"
    off = (lambda v: v.Not()) if work else (lambda v: v)   # literal "day is outside the run"
    for L in range(1, m_min):
        for a in range(1, D - L):           # run a..a+L-1, neighbours a-1 and a+L inside the horizon
            run = [on(w[i, k]) for k in range(a, a + L)]
            left, right = off(w[i, a - 1]), off(w[i, a + L])
            # pattern holds iff left and all of run and right
            if pen is None:
                M.model.AddBoolOr([left.Not(), right.Not()] + [r.Not() for r in run])
            else:
                z = M.model.NewBoolVar(f"{tag}[{i},{a},{L}]")
                lits = [left, right] + run
                M.model.Add(z >= sum(lits) - (len(lits) - 1))
                out.append((m_min - L) * z)
    return out


def _enc_min_consecutive_shifts(M: RosterModel, pen):
    out = []
    for i in range(M.x.shape[0]):
        out += _interior_run_patterns(M, i, int(M.inst.min_consec[i]), True, pen, "mincons")
    return out


def _enc_min_consecutive_days_off(M: RosterModel, pen):
    out = []
    for i in range(M.x.shape[0]):
        out += _interior_run_patterns(M, i, int(M.inst.min_consec_off[i]), False, pen, "minoff")
    return out


def _enc_max_weekends(M: RosterModel, pen):
    inst, w, out = M.inst, M.w, []
    I, D = w.shape
    wk = inst.weekend_of_day
    weekends = [np.flatnonzero(wk == k) for k in np.unique(wk[wk >= 0])]
    if not weekends:
        return out
    for i in range(I):
        worked = []
        for k, days in enumerate(weekends):
            v = M.model.NewBoolVar(f"wkend[{i},{k}]")
            for d in days:
                M.model.Add(v >= w[i, d])
            worked.append(v)
        cap = int(inst.max_weekends[i])
        if cap >= len(weekends):
            continue
        if pen is None:
            M.model.Add(sum(worked) <= cap)
        else:
            out.append(M.penalty_var(sum(worked) - cap, len(weekends) - cap, f"maxwk[{i}]"))
    return out


def _forbid_cells(M: RosterModel, mask, pen):
    """Individual (i, d) cells that may not be worked: days off, absences."""
    out = []
    for i, d in zip(*np.nonzero(mask)):
        if pen is None:
            M.model.Add(M.w[i, d] == 0)
        else:
            out.append(M.w[i, d])
    return out


def _enc_days_off(M: RosterModel, pen):
    return _forbid_cells(M, M.inst.day_off, pen)


def _enc_unavailability(M: RosterModel, pen):
    return _forbid_cells(M, M.ctx.absent, pen)


def _enc_skills(M: RosterModel, pen):
    out = []
    I, D, S = M.x.shape
    for i, s in zip(*np.nonzero(~M.inst.eligible)):
        for d in range(D):
            if pen is None:
                M.model.Add(M.x[i, d, s] == 0)
            else:
                out.append(M.x[i, d, s])
    return out


def _enc_min_coverage(M: RosterModel, pen):
    inst, x, out = M.inst, M.x, []
    _, D, S = x.shape
    for d in range(D):
        for s in range(S):
            r = int(inst.min_cover[d, s])
            if r <= 0:
                continue
            n = sum(x[:, d, s])
            if pen is None:
                M.model.Add(n >= r)
            else:
                out.append(M.penalty_var(r - n, r, f"mincov[{d},{s}]"))
    return out


ENCODERS = {
    "shift_rotation": _enc_shift_rotation,
    "max_shifts_per_type": _enc_max_shifts_per_type,
    "max_total_minutes": _enc_max_total_minutes,
    "min_total_minutes": _enc_min_total_minutes,
    "max_consecutive_shifts": _enc_max_consecutive_shifts,
    "min_consecutive_shifts": _enc_min_consecutive_shifts,
    "min_consecutive_days_off": _enc_min_consecutive_days_off,
    "max_weekends": _enc_max_weekends,
    "days_off": _enc_days_off,
    "unavailability": _enc_unavailability,
    "skills": _enc_skills,
    "min_coverage": _enc_min_coverage,
}
