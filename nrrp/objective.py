"""The objective J(x; phi). One implementation, used by the reward, the CP subproblem and evaluation.

With x_{ids} = [y_{id} = s], n_{ds} = sum_i x_{ids} and rho the context's
(post-disruption) cover requirement:

    requests   = sum_{i,d,s} on_{ids} (1 - x_{ids}) + off_{ids} x_{ids}         all days
    cover      = sum_{d,s} C_{ds}(n_{ds}),                                        all days
                 C_{ds}(n) = w^u_{ds} max(rho_{ds} - n, 0) + w^o_{ds} max(n - rho_{ds}, 0)
    stability  = mu sum_{i, d >= d_xi, s} |x_{ids} - x°_{ids}|
    penalties  = sum_c w_c sum_i v_c(row_i)       penalised constraints of the profile

    J = requests + cover + stability + penalties

Hard constraints are not part of J; :func:`hard_violations` reports them.

Decomposition used by the environment (Prop. 2, with fixes 1a and 1b of the
implementation brief). Evaluate J on partial rosters, undecided slots empty.
Assigning (i, d, s) when f employees already work (d, s) changes J by

    lin[i, d, s] + C_{ds}(f + 1) - C_{ds}(f) + sum_c w_c delta_c(i),
    lin = off - on + mu (1 - 2 x°) [d >= d_xi],

and leaving a slot empty changes nothing. The increments telescope from the
initial roster x^0 (frozen history, empty suffix), so they sum to
J(x) - J(x^0) whatever the order of the fills, and the constant

    K(phi) = J(x^0)

collects every term no decision can change: c^req = sum on, the cost of the
frozen history (requests, cover, penalties), the shortfall C_{ds}(0) of every
open pair, and the stability constant mu sum_{d >= d_xi} x°.
"""
from __future__ import annotations

import numpy as np

from .instance import Context, empty_roster, roster_to_binary


def cover_cost(n, rho, w_under, w_over):
    """C(n) = w^u max(rho - n, 0) + w^o max(n - rho, 0), elementwise."""
    return w_under * np.maximum(rho - n, 0) + w_over * np.maximum(n - rho, 0)


def J_terms(y: np.ndarray, ctx: Context, profile) -> dict:
    """J split into its terms; penalties appear as 'penalty:<constraint name>'."""
    inst = ctx.instance
    S = inst.n_shifts
    x = roster_to_binary(y, S)
    x0 = roster_to_binary(ctx.y0, S)
    terms = {
        "requests": float((inst.on_req * ~x).sum() + (inst.off_req * x).sum()),
        "cover": float(cover_cost(x.sum(axis=0), ctx.rho, inst.w_under, inst.w_over).sum()),
        "stability": ctx.mu * float((x[:, ctx.d_xi:] != x0[:, ctx.d_xi:]).sum()),
    }
    for c, w in profile.penalties:
        terms[f"penalty:{c.name}"] = w * float(c.violation(y, ctx).sum())
    return terms


def J(y: np.ndarray, ctx: Context, profile) -> float:
    return float(sum(J_terms(y, ctx, profile).values()))


def hard_violations(y: np.ndarray, ctx: Context, profile, which: str = "all") -> dict:
    """{constraint name: (I,) violation amounts} for the hard constraints.

    ``which``: "all", "masked" (hard and prefix-monotone, guaranteed by Prop. 3)
    or "ter" (H^ter, left to the repair step).
    """
    groups = {"all": profile.masked + profile.ter, "masked": profile.masked, "ter": profile.ter}
    if which not in groups:
        raise ValueError(f"which must be one of {sorted(groups)}")
    return {c.name: c.violation(y, ctx) for c in groups[which]}


def initial_roster(ctx: Context) -> np.ndarray:
    """x^0: the frozen history d < d_xi of the published roster, nothing after it."""
    inst = ctx.instance
    y = empty_roster(inst.n_emp, inst.n_days)
    y[:, : ctx.d_xi] = ctx.y0[:, : ctx.d_xi]
    return y


def context_constant(ctx: Context, profile) -> float:
    """K(phi) = J(x^0)."""
    return J(initial_roster(ctx), ctx, profile)


def linear_costs(ctx: Context) -> np.ndarray:
    """(I, D, S) per-assignment cost of requests and stability: off - on + mu (1 - 2 x°) [d >= d_xi]."""
    inst = ctx.instance
    lin = inst.off_req - inst.on_req
    if ctx.mu:
        x0 = roster_to_binary(ctx.y0, inst.n_shifts).astype(float)
        lin = lin.copy()
        lin[:, ctx.d_xi:] += ctx.mu * (1.0 - 2.0 * x0[:, ctx.d_xi:])
    return lin


def cover_increments(ctx: Context, d: int, s: int, n_max: int) -> np.ndarray:
    """(n_max,) C_{ds}(f + 1) - C_{ds}(f) for f = 0..n_max-1."""
    inst = ctx.instance
    c = cover_cost(np.arange(n_max + 1), ctx.rho[d, s], inst.w_under[d, s], inst.w_over[d, s])
    return np.diff(c)
