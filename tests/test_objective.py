from pathlib import Path

import numpy as np
import pytest

from nrrp.bc_format import read_bc
from nrrp.constraints import random_block_row
from nrrp.generator import synthetic_instance
from nrrp.instance import OFF, Context, roster_to_binary
from nrrp.objective import (J, J_terms, context_constant, cover_cost, cover_increments,
                            hard_violations, initial_roster, linear_costs)
from nrrp.profile import BURKE_CURTOIS

MINI = Path(__file__).parent / "data" / "mini_bc.txt"
_, E, L = OFF, 0, 1

# A: day off on day 0; C: days off 2, 3; requests A-1-E (2), B-4-E (3), off C-5-L (1)
ROSTER = np.array([
    [_, E, E, _, _, _, _],   # A
    [E, _, _, _, E, _, _],   # B
    [L, L, _, _, _, L, _],   # C
], np.int8)


def test_J_by_hand_on_the_mini_instance():
    ctx = Context(read_bc(MINI))
    t = J_terms(ROSTER, ctx, BURKE_CURTOIS)
    assert t["requests"] == 1                      # C works L on day 5 despite the off request
    # day 1: E 1 of 2 (100), L 1 over 0 (1); day 2: L 0 of 1 (100); day 3: E, L missing (200);
    # day 4: E 1 of 2, L 0 of 1 (200); day 5: E missing (100), L 1 over 0 (1);
    # day 6: E missing (100), L missing at weight 50
    assert t["cover"] == 101 + 100 + 200 + 200 + 101 + 150
    assert t["stability"] == 0
    assert J(ROSTER, ctx, BURKE_CURTOIS) == 853


def test_penalties_by_hand():
    ctx = Context(read_bc(MINI))
    p = BURKE_CURTOIS.with_penalised({"min_total_minutes": 0.5, "min_consecutive_shifts": 10})
    t = J_terms(ROSTER, ctx, p)
    assert t["penalty:min_total_minutes"] == 0.5 * (480 + 0 + 120)   # A 960 of 1440, C 1800 of 1920
    assert t["penalty:min_consecutive_shifts"] == 10 * 1             # C's single day 5 (min 2)
    assert J(ROSTER, ctx, p) == 853 + 300 + 10


def test_stability_counts_hamming_distance_from_d_xi():
    inst = read_bc(MINI)
    ctx = Context(inst, d_xi=3, y0=ROSTER, mu=2.0)
    y = ROSTER.copy()
    y[2, 5] = E      # L -> E on day 5: two binary variables change
    y[1, 4] = OFF    # E -> off on day 4: one
    y[0, 1] = OFF    # before d_xi: not counted
    assert J_terms(y, ctx, BURKE_CURTOIS)["stability"] == 2.0 * 3


def test_hard_violations_split():
    ctx = Context(read_bc(MINI))
    ter = hard_violations(ROSTER, ctx, BURKE_CURTOIS, "ter")
    assert set(ter) == {"min_total_minutes", "min_consecutive_shifts"}
    assert list(ter["min_total_minutes"]) == [480, 0, 120]
    masked = hard_violations(ROSTER, ctx, BURKE_CURTOIS, "masked")
    assert list(masked["max_weekends"]) == [0, 0, 1]     # C may work no weekend, works Saturday
    assert masked["shift_rotation"].sum() == 0           # L -> L is allowed
    assert list(masked["days_off"]) == [0, 0, 0]
    with pytest.raises(ValueError):
        hard_violations(ROSTER, ctx, BURKE_CURTOIS, "soft")


def test_cover_increments_telescope():
    inst = synthetic_instance(10, 7, 3, seed=1)
    ctx = Context(inst)
    for d, s in [(0, 0), (3, 1), (6, 2)]:
        inc = cover_increments(ctx, d, s, 10)
        for n in range(11):
            want = cover_cost(n, ctx.rho[d, s], inst.w_under[d, s], inst.w_over[d, s]) \
                - cover_cost(0, ctx.rho[d, s], inst.w_under[d, s], inst.w_over[d, s])
            assert inc[:n].sum() == want


@pytest.mark.parametrize("seed", range(5))
def test_linear_part_of_J(seed):
    """requests + stability = sum(on) + mu sum_{d>=d_xi} x° + sum lin * x, for any full roster."""
    rng = np.random.default_rng(seed)
    inst = synthetic_instance(8, 14, 3, seed=seed)
    y0 = np.array([random_block_row(14, 3, rng) for _ in range(8)])
    ctx = Context(inst, d_xi=int(rng.integers(1, 14)), y0=y0, mu=float(rng.uniform(0.5, 3)))
    y = np.array([random_block_row(14, 3, rng) for _ in range(8)])
    y[:, : ctx.d_xi] = y0[:, : ctx.d_xi]
    t = J_terms(y, ctx, BURKE_CURTOIS)
    x = roster_to_binary(y, 3)
    x0 = roster_to_binary(y0, 3)
    const = inst.on_req.sum() + ctx.mu * x0[:, ctx.d_xi:].sum()
    assert np.isclose(t["requests"] + t["stability"], const + (linear_costs(ctx) * x).sum(), rtol=0, atol=1e-9)


def test_context_constant_is_J_of_the_initial_roster():
    rng = np.random.default_rng(0)
    inst = synthetic_instance(6, 14, 2, seed=0)
    y0 = np.array([random_block_row(14, 2, rng) for _ in range(6)])
    ctx = Context(inst, d_xi=5, y0=y0, mu=1.0)
    x0 = initial_roster(ctx)
    assert (x0[:, :5] == y0[:, :5]).all() and (x0[:, 5:] == OFF).all()
    assert context_constant(ctx, BURKE_CURTOIS) == J(x0, ctx, BURKE_CURTOIS)
    # construction: K = c^req + sum of C(0) = sum(on) + sum(w_under * rho)
    c = Context(inst)
    assert context_constant(c, BURKE_CURTOIS) == inst.on_req.sum() + (inst.w_under * inst.rho).sum()
