"""The repair operator of Section 4.3: the bounded neighbourhood (40)-(41) and
Algorithm 1 (escalation and fallback). Needs OR-Tools (the ``cp`` extra).

Given the roster x^(T) the policy constructed, which satisfies H^pre (Prop. 3)
but may violate H^ter (Lemma 1), the operator re-solves problem (20) exactly
over a neighbourhood of x^(T), subject to every hard constraint of the profile
(H^pre and H^ter), with every variable outside the neighbourhood fixed to its
value in x^(T):

    V_w = {(i, d, s) : d_xi <= d <= min(d_xi + w, m)}  ∪  V^∅  ∪  V^ter     (40)
    V^ter = {(i, d, s) : d >= d_xi, i violates some c in H^ter in x^(T)}   (41)

V^∅ collects the slots the policy left unfilled: for every pair (d, s) with
d >= d_xi whose staffing in x^(T) is below its slot count rho_bar_{ds} (the
empty action was taken there), the variables x_{i,d,s} of every employee i. An
employee already working another shift that day stays fixed to it, so V^∅ lets
an off-duty employee take the slot.

Algorithm 1: start at w = w0; solve the subproblem within the remaining budget;
return its solution if one is found (optimal within N_w when the solver proves
it); if the subproblem is proved infeasible and the window does not yet reach
the end of the horizon, double w and retry; otherwise stop and return x^(T)
with its violated constraints (the fallback). At the maximal window the
subproblem is the re-rostering problem itself, so a feasible roster is returned
whenever one exists and the budget suffices (Prop. 4).

Days are 0-indexed here: the window is d_xi..min(d_xi + w, m - 1) and is
maximal once d_xi + w >= m - 1 (the paper's d_xi + w >= m, 1-indexed).

Points the paper leaves open, and the choice made here:
* w0 = 0 would stay 0 under doubling; the window grows as max(1, 2w).
* If x^(T) is already feasible and the subproblem returns a worse roster (a
  time-limited, non-optimal solution), x^(T) is kept: the operator never makes
  a feasible roster worse. With an optimal subproblem this cannot happen,
  since x^(T) itself lies in N_w.
* Algorithm 1 returns at the first feasible window, so a larger tau_cp does not
  improve a roster once w0 yields one, whereas Section 4.4 describes tau_cp as
  the quality dial ("as tau_cp grows the operator reaches wider windows and the
  gap closes"). ``anytime=True`` implements that reading: keep doubling w after
  a success while budget remains. The default is Algorithm 1 as printed.
* Minimum coverage (4), a column constraint, is not covered by (41), which is
  stated for row constraints; when it is hard and violated, the variables of
  the violated pair (d, s) are freed for all employees. It is off for the
  Burke & Curtois family.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .cpsat import RosterModel
from .instance import OFF, Context, roster_to_binary
from .objective import evaluate, hard_violations


@dataclass
class RepairResult:
    roster: np.ndarray                 # the returned roster
    feasible: bool                     # under every hard constraint of the profile
    J: float                           # J of the returned roster under the profile
    violations: dict                   # hard constraints still violated (name -> total)
    J_before: float                    # J of x^(T)
    feasible_before: bool              # x^(T) feasible
    window: int | None                 # w at which a feasible roster was found, None if none
    windows: list = field(default_factory=list)       # w of every subproblem built
    statuses: list = field(default_factory=list)      # solver status of every subproblem
    n_free: list = field(default_factory=list)        # |V_w| of every subproblem
    n_ter_employees: int = 0           # employees whose row violates H^ter in x^(T)
    n_empty_pairs: int = 0             # pairs (d, s) in V^∅
    seconds: float = 0.0               # T_repair, wall clock, building included
    kept_input: bool = False           # x^(T) returned unchanged (fallback, or no improvement)
    changed_cells: int = 0             # cells (i, d) where the returned roster differs from x^(T)


def empty_pairs(roster: np.ndarray, ctx: Context, kappa: int = 1) -> np.ndarray:
    """(m, k) bool: pairs (d, s), d >= d_xi, whose staffing is below rho_bar = min(rho + kappa, n),
    i.e. where the policy left a slot unfilled."""
    inst = ctx.instance
    rho_bar = np.minimum(inst.rho + kappa, inst.n_emp)
    n = roster_to_binary(roster, inst.n_shifts).sum(axis=0)
    out = (n < rho_bar) & (rho_bar > 0)
    out[: ctx.d_xi] = False
    return out


def ter_employees(roster: np.ndarray, ctx: Context, profile) -> np.ndarray:
    """(n,) bool: employees whose row violates some row constraint of H^ter."""
    inst = ctx.instance
    bad = np.zeros(inst.n_emp, bool)
    for c in profile.ter:
        if c.scope == "row":
            bad |= c.violation(roster, ctx) > 0
    return bad


def violated_pairs(roster: np.ndarray, ctx: Context, profile) -> np.ndarray:
    """(m, k) bool: pairs violating a column constraint of H^ter (minimum coverage)."""
    inst = ctx.instance
    out = np.zeros((inst.n_days, inst.n_shifts), bool)
    for c in profile.ter:
        if c.scope == "column":
            out |= c.violation(roster, ctx) > 0
    out[: ctx.d_xi] = False
    return out


def neighbourhood(roster: np.ndarray, ctx: Context, profile, w: int, kappa: int = 1) -> np.ndarray:
    """(n, m, k) bool: the freed variables V_w of Eq. (40) around the roster x^(T)."""
    inst = ctx.instance
    I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
    if w < 0:
        raise ValueError("the window width w must be >= 0")
    free = np.zeros((I, D, S), bool)
    free[:, ctx.d_xi: min(ctx.d_xi + w, D - 1) + 1, :] = True             # the window
    free[:, empty_pairs(roster, ctx, kappa)] = True                         # V^∅
    free[ter_employees(roster, ctx, profile), ctx.d_xi:, :] = True          # V^ter, (41)
    free[:, violated_pairs(roster, ctx, profile)] = True                    # column constraints of H^ter
    return free


def _summary(roster, ctx, profile):
    ev = evaluate(roster, ctx, profile)
    return ev["J"], ev["feasible"], ev["violations"]


def repair(roster: np.ndarray, ctx: Context, profile, *, w0: int = 1, budget: float = 1.0,
           kappa: int = 1, workers: int = 8, seed: int = 0,
           deterministic_budget: float | None = None, anytime: bool = False) -> RepairResult:
    """Algorithm 1 on the roster x^(T) the policy constructed.

    ``anytime=False`` is Algorithm 1 as printed: stop at the first window that yields
    a feasible roster. ``anytime=True`` keeps widening after a success while budget
    remains (each wider subproblem hinted with the incumbent, the best roster kept),
    so that tau_cp acts as the quality dial Section 4.4 describes; with enough budget
    it reaches the maximal window and the exact optimum. N_w is always taken around
    x^(T), so the neighbourhoods stay nested (Prop. 4).

    ``profile``: the evaluation profile; all its hard constraints (H^pre and H^ter)
    are imposed and J is its objective. ``budget``: tau_cp, wall-clock seconds for
    the whole operator. ``deterministic_budget``: CP-SAT deterministic time for the
    whole operator instead (reproducible, for tests); if given, ``budget`` only caps
    wall-clock time. ``kappa``: as in the environment, to find the unfilled slots.
    """
    t0 = time.perf_counter()
    roster = np.asarray(roster, dtype=np.int8)
    inst = ctx.instance
    if roster.shape != (inst.n_emp, inst.n_days):
        raise ValueError(f"roster shape {roster.shape}, instance needs {(inst.n_emp, inst.n_days)}")
    if not np.array_equal(roster[:, : ctx.d_xi], ctx.published[:, : ctx.d_xi]):
        raise ValueError("the roster does not satisfy the history constraint (days before d_xi)")
    if w0 < 0:
        raise ValueError("w0 must be >= 0")

    J0, feas0, viol0 = _summary(roster, ctx, profile)
    res = RepairResult(roster=roster.copy(), feasible=feas0, J=J0, violations=viol0, J_before=J0,
                       feasible_before=feas0, window=None, kept_input=True,
                       n_ter_employees=int(ter_employees(roster, ctx, profile).sum()),
                       n_empty_pairs=int(empty_pairs(roster, ctx, kappa).sum()))
    w, det_used = int(w0), 0.0
    D = inst.n_days
    while True:
        remaining = budget - (time.perf_counter() - t0)
        det_left = None if deterministic_budget is None else deterministic_budget - det_used
        if remaining <= 0 or (det_left is not None and det_left <= 0):
            break
        free = neighbourhood(roster, ctx, profile, w, kappa)
        model = RosterModel(ctx, profile, free=free, base=roster)
        sol = model.solve(time_limit=remaining, workers=workers, seed=seed,
                          hint=res.roster if res.window is not None else roster,
                          deterministic_limit=det_left)
        det_used += sol.info.get("deterministic_time", 0.0)
        res.windows.append(w)
        res.statuses.append(sol.status)
        res.n_free.append(model.n_free)
        if sol.has_solution and sol.feasible:
            if not res.feasible or sol.J < res.J:          # better than the incumbent (or than a feasible x^(T))
                res.roster, res.J, res.feasible, res.violations = sol.roster, sol.J, True, {}
                res.kept_input = False
                res.window = w
            elif res.window is None:
                res.window = w                             # x^(T) was feasible and is kept
            if anytime and sol.status == "OPTIMAL" and ctx.d_xi + w < D - 1:
                w = max(1, 2 * w)                          # anytime: widen while budget remains
                continue
            break
        if sol.status == "INFEASIBLE" and ctx.d_xi + w < D - 1:
            w = max(1, 2 * w)                                  # widen and retry
            continue
        break                                                  # budget spent, or window already maximal
    res.seconds = time.perf_counter() - t0
    res.changed_cells = int((res.roster != roster).sum())
    return res


def violation_totals(roster: np.ndarray, ctx: Context, profile) -> dict:
    """{constraint: total violation} of the hard constraints that are violated."""
    return {n: float(v.sum()) for n, v in hard_violations(roster, ctx, profile).items() if v.sum()}


__all__ = ["RepairResult", "empty_pairs", "ter_employees", "violated_pairs", "neighbourhood", "repair",
           "violation_totals", "OFF"]
