"""The objective J(x; phi). One implementation, used by the reward, the CP subproblem and evaluation.

With x_{ids} = [roster_{id} = s] and n_{ds} = sum_i x_{ids}:

    requests   = sum_{i,d,s} p_{ids} x_{ids} + c^req,   p = w^- - w^+ (Eq. (11)), c^req = sum w^+
    cover      = sum_{d,s} C_{ds}(n_{ds}),
                 C_{ds}(n) = w^u_{ds} max(rho_{ds} - n, 0) + w^o_{ds} max(n - rho_{ds}, 0)   (Eq. (29))
    stability  = mu Delta(x, x°) = mu sum_{i, d >= d_xi, s} |x_{ids} - x°_{ids}|          (Eq. (16))
    penalties  = sum_c w_c v_c(x),   for the constraints the profile penalises

    J = f(x) + mu Delta(x, x°) + penalties,   f = requests + cover   (Eqs. (13), (20))

Hard constraints are not part of J; :func:`hard_violations` reports them and
:func:`evaluate` reports J together with feasibility.

Per-step decomposition (Prop. 2). Evaluate J on partial rosters, undecided
slots empty. Assigning (i, d, s) when f employees already work (d, s) changes J by

    c_t = lin[i, d, s] + C_{ds}(f + 1) - C_{ds}(f) + sum_c w_c delta_c(i),
    lin = p + mu (1 - 2 x°) [d >= d_xi],

where column constraints contribute per-fill increments instead of a delta,
and leaving a slot empty changes nothing. So c_t = J(x^(t+1)) - J(x^(t)), the
increments telescope from the initial roster x^(0) (frozen history, empty
suffix) whatever the order of the fills, and with

    c^con = J(x^(0))

charged once, the rewards -c_t sum to -J(x^(T)) exactly. c^con collects every
term no decision can change: c^req, the cost of the frozen history, the
shortfall C_{ds}(0) of every open pair, the instability mu sum_{d >= d_xi} x°
of the empty suffix, and the penalties of x^(0).
"""
from __future__ import annotations

import numpy as np

from .instance import Context, empty_roster, roster_to_binary


def cover_cost(n, rho, w_under, w_over):
    """C(n) = w^u max(rho - n, 0) + w^o max(n - rho, 0), elementwise."""
    return w_under * np.maximum(rho - n, 0) + w_over * np.maximum(n - rho, 0)


def J_terms(roster: np.ndarray, ctx: Context, profile) -> dict:
    """J split into its terms; penalties appear as 'penalty:<constraint name>'."""
    inst = ctx.instance
    S = inst.n_shifts
    x = roster_to_binary(roster, S)
    x0 = roster_to_binary(ctx.published, S)
    terms = {
        "requests": float((inst.on_req * ~x).sum() + (inst.off_req * x).sum()),
        "cover": float(cover_cost(x.sum(axis=0), inst.rho, inst.w_under, inst.w_over).sum()),
        "stability": ctx.mu * float((x[:, ctx.d_xi:] != x0[:, ctx.d_xi:]).sum()),
    }
    for c, w in profile.penalties:
        terms[f"penalty:{c.name}"] = w * float(c.violation(roster, ctx).sum())
    return terms


def J(roster: np.ndarray, ctx: Context, profile) -> float:
    return float(sum(J_terms(roster, ctx, profile).values()))


def hard_violations(roster: np.ndarray, ctx: Context, profile, which: str = "all") -> dict:
    """{constraint name: violation amounts} for the hard constraints.

    ``which``: "all", "pre" (H^pre, guaranteed by Prop. 3) or "ter" (H^ter, left to
    the repair operator). Row constraints give an (n,) array, column constraints (m, k).
    """
    groups = {"all": profile.pre + profile.ter, "pre": profile.pre, "ter": profile.ter}
    if which not in groups:
        raise ValueError(f"which must be one of {sorted(groups)}")
    return {c.name: c.violation(roster, ctx) for c in groups[which]}


def evaluate(roster: np.ndarray, ctx: Context, profile) -> dict:
    """What is reported for a final roster under the evaluation profile.

    J and its terms, whether every hard constraint holds, and the total violation
    of each hard constraint that does not.
    """
    viol = {n: float(v.sum()) for n, v in hard_violations(roster, ctx, profile).items()}
    terms = J_terms(roster, ctx, profile)
    return {"J": float(sum(terms.values())), "terms": terms,
            "feasible": not any(viol.values()),
            "violations": {n: v for n, v in viol.items() if v}}


def initial_roster(ctx: Context) -> np.ndarray:
    """x^(0): the frozen history d < d_xi of the published roster, nothing after it."""
    inst = ctx.instance
    roster = empty_roster(inst.n_emp, inst.n_days)
    roster[:, : ctx.d_xi] = ctx.published[:, : ctx.d_xi]
    return roster


def context_constant(ctx: Context, profile) -> float:
    """c^con = J(x^(0))."""
    return J(initial_roster(ctx), ctx, profile)


def linear_costs(ctx: Context) -> np.ndarray:
    """(n, m, k) per-assignment cost of preferences and instability: p + mu (1 - 2 x°) [d >= d_xi]."""
    inst = ctx.instance
    lin = inst.off_req - inst.on_req
    if ctx.mu:
        x0 = roster_to_binary(ctx.published, inst.n_shifts).astype(float)
        lin = lin.copy()
        lin[:, ctx.d_xi:] += ctx.mu * (1.0 - 2.0 * x0[:, ctx.d_xi:])
    return lin


def cover_increments(ctx: Context, d: int, s: int, n_max: int) -> np.ndarray:
    """(n_max,) C_{ds}(f + 1) - C_{ds}(f) for f = 0..n_max-1."""
    inst = ctx.instance
    c = cover_cost(np.arange(n_max + 1), inst.rho[d, s], inst.w_under[d, s], inst.w_over[d, s])
    return np.diff(c)


def fill_increments(ctx: Context, profile, d: int, s: int, n_max: int) -> np.ndarray:
    """(n_max,) change of J from the staffing of (d, s) when it goes from f to f + 1:
    the cover increment plus the increments of penalised column constraints."""
    inc = cover_increments(ctx, d, s, n_max)
    for c, w in profile.penalties:
        if c.scope == "column":
            inc = inc + w * c.increments(ctx, d, s, n_max)
    return inc
