"""The repair operator: neighbourhood (40)-(41), Algorithm 1, Proposition 4."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")

from nrrp.bc_format import parse_bc, read_bc  # noqa: E402
from nrrp.cpsat import solve  # noqa: E402
from nrrp.env import RosterEnv  # noqa: E402
from nrrp.generator import random_roster, synthetic_instance  # noqa: E402
from nrrp.instance import OFF, Context  # noqa: E402
from nrrp.objective import evaluate  # noqa: E402
from nrrp.profile import BURKE_CURTOIS  # noqa: E402
from nrrp.repair import empty_pairs, neighbourhood, repair, ter_employees  # noqa: E402

MINI = Path(__file__).parent / "data" / "mini_bc.txt"
FEASIBLE_TEXT = MINI.read_text().replace("C,E=7|L=3,2880,1920,", "C,E=7|L=3,2880,960,")
# the hand-checked feasible roster of mini_feasible (tests/test_roster_io.py), J = 250
FEASIBLE_ROSTER = np.array([[OFF, 0, 1, 1, 1, OFF, OFF],
                            [OFF, OFF, 0, 0, 0, 0, 0],
                            [0, 0, OFF, OFF, OFF, OFF, OFF]], np.int8)
E, L = 0, 1
QUICK = dict(workers=1, budget=60.0, deterministic_budget=60.0)


def mini():
    return parse_bc(FEASIBLE_TEXT, name="mini_feasible")


def test_neighbourhood_by_hand():
    """C loses day 1: C is short of 960 minutes (H^ter), and with kappa = 0 the pairs left
    below rho are (d1, E), (d4, E), (d0, L), (d6, L)."""
    inst = mini()
    ctx = Context(inst)
    x = FEASIBLE_ROSTER.copy()
    x[2, 1] = OFF
    np.testing.assert_array_equal(ter_employees(x, ctx, BURKE_CURTOIS), [False, False, True])
    want_empty = np.zeros((7, 2), bool)
    want_empty[[1, 4], E] = True
    want_empty[[0, 6], L] = True
    np.testing.assert_array_equal(empty_pairs(x, ctx, kappa=0), want_empty)
    free = neighbourhood(x, ctx, BURKE_CURTOIS, w=1, kappa=0)
    want = np.zeros((3, 7, 2), bool)
    want[:, 0:2, :] = True                 # window d_xi = 0 .. 0 + 1
    want[:, want_empty] = True             # V^∅
    want[2, :, :] = True                   # V^ter: C's whole horizon
    np.testing.assert_array_equal(free, want)


def test_window_starts_at_the_disruption_and_is_clipped():
    inst = mini()
    ctx = Context(inst, d_xi=3, published=FEASIBLE_ROSTER)
    free = neighbourhood(FEASIBLE_ROSTER, ctx, BURKE_CURTOIS, w=10, kappa=0)
    assert not free[:, :3].any()           # the frozen history is never freed
    assert free[:, 3:].all()               # the window reaches the last day


def random_masked_roster(ctx, seed):
    """x^(T) from a uniformly random masked rollout in the context (satisfies H^pre and the history)."""
    env = RosterEnv(ctx.instance, BURKE_CURTOIS, contexts=ctx)
    rng = np.random.default_rng(seed)
    env.reset()
    done = False
    while not done:
        allowed = np.flatnonzero(env.action_mask())
        _, _, done, _ = env.step(int(allowed[rng.integers(len(allowed))]))
    return env.roster.copy()


def test_neighbourhoods_are_nested():
    inst = synthetic_instance(6, 14, 3, seed=2)
    for seed in range(5):
        ctx = Context(inst)
        x = random_masked_roster(ctx, seed)
        prev = None
        for w in (0, 1, 2, 4, 8, 13):
            free = neighbourhood(x, ctx, BURKE_CURTOIS, w)
            if prev is not None:
                assert not (prev & ~free).any()
            prev = free
        assert free[:, ctx.d_xi:].all()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("w0", [0, 1])
def test_proposition_4_feasible_whenever_the_problem_is(seed, w0):
    inst = synthetic_instance(5, 7, 3, seed=seed)
    ctx = Context(inst)
    exact = solve(ctx, BURKE_CURTOIS, workers=1, deterministic_limit=30)
    x = random_masked_roster(ctx, seed)
    res = repair(x, ctx, BURKE_CURTOIS, w0=w0, **QUICK)
    if exact.has_solution:
        assert res.feasible and not res.violations
        assert evaluate(res.roster, ctx, BURKE_CURTOIS)["feasible"]
        assert res.J == evaluate(res.roster, ctx, BURKE_CURTOIS)["J"] >= exact.J - 1e-9
    else:
        assert exact.status == "INFEASIBLE"
        assert not res.feasible and res.kept_input


@pytest.mark.parametrize("seed", [10, 12, 14, 15])       # seeds with a proven optimum (11, 13 are infeasible)
def test_maximal_window_recovers_the_exact_optimum(seed):
    inst = synthetic_instance(5, 7, 3, seed=seed)
    ctx = Context(inst)
    exact = solve(ctx, BURKE_CURTOIS, workers=1, deterministic_limit=60)
    assert exact.status == "OPTIMAL"
    x = random_masked_roster(ctx, seed)
    res = repair(x, ctx, BURKE_CURTOIS, w0=inst.n_days, **QUICK)
    assert res.statuses[-1] == "OPTIMAL" and res.J == pytest.approx(exact.J)


def test_a_feasible_roster_is_never_made_worse():
    inst = mini()
    ctx = Context(inst)
    res = repair(FEASIBLE_ROSTER, ctx, BURKE_CURTOIS, w0=1, **QUICK)
    assert res.feasible_before and res.feasible
    assert res.J <= res.J_before == 250
    res = repair(FEASIBLE_ROSTER, ctx, BURKE_CURTOIS, w0=7, **QUICK)
    assert res.J == 250                            # 250 is the optimum (exhaustive search)


def test_fallback_on_an_infeasible_instance_escalates_then_reports():
    inst = read_bc(MINI)                           # no feasible roster (employee C)
    ctx = Context(inst)
    x = random_masked_roster(ctx, 0)
    res = repair(x, ctx, BURKE_CURTOIS, w0=0, **QUICK)
    assert res.windows == [0, 1, 2, 4, 8]          # doubling from 0 goes through 1; 0 + 8 >= m - 1
    assert res.statuses == ["INFEASIBLE"] * 5
    assert not res.feasible and res.kept_input and res.window is None
    np.testing.assert_array_equal(res.roster, x)
    assert res.violations == evaluate(x, ctx, BURKE_CURTOIS)["violations"] and res.violations


def test_rerostering_keeps_the_history_and_the_absences():
    inst = synthetic_instance(6, 14, 3, seed=1)
    base = solve(Context(inst), BURKE_CURTOIS, workers=1, deterministic_limit=30)
    assert base.has_solution
    pub = base.roster
    d_xi = 5
    absent = np.zeros((6, 14), bool)
    who = int(np.flatnonzero(pub[:, d_xi] != OFF)[0])
    absent[who, d_xi:d_xi + 2] = True
    ctx = Context(inst, d_xi=d_xi, published=pub, absent=absent, mu=1.0)
    x = random_masked_roster(ctx, 3)
    res = repair(x, ctx, BURKE_CURTOIS, w0=1, **QUICK)
    np.testing.assert_array_equal(res.roster[:, :d_xi], pub[:, :d_xi])
    assert (res.roster[absent] == OFF).all()
    exact = solve(ctx, BURKE_CURTOIS, workers=1, deterministic_limit=30)
    assert res.feasible == exact.has_solution


def test_no_budget_returns_the_input():
    inst = mini()
    ctx = Context(inst)
    x = FEASIBLE_ROSTER.copy()
    x[2, 1] = OFF
    res = repair(x, ctx, BURKE_CURTOIS, budget=0.0)
    assert res.windows == [] and res.kept_input and not res.feasible
    np.testing.assert_array_equal(res.roster, x)


def test_building_the_subproblem_counts_against_the_budget(monkeypatch):
    """Eq. (42), T_repair <= tau_cp: a budget spent while building the subproblem leaves
    nothing for CP-SAT, so no subproblem is solved and the input is returned."""
    import time

    import nrrp.repair as rep

    class SlowModel(rep.RosterModel):
        def __init__(self, *a, **kw):
            time.sleep(0.2)
            super().__init__(*a, **kw)

    monkeypatch.setattr(rep, "RosterModel", SlowModel)
    inst = mini()
    ctx = Context(inst)
    x = FEASIBLE_ROSTER.copy()
    x[2, 1] = OFF
    res = repair(x, ctx, BURKE_CURTOIS, budget=0.1, workers=1)
    assert res.windows == [] and res.statuses == [] and res.kept_input
    assert res.seconds < 0.5
    np.testing.assert_array_equal(res.roster, x)


def test_a_roster_breaking_the_history_is_refused():
    inst = mini()
    ctx = Context(inst, d_xi=3, published=FEASIBLE_ROSTER)
    x = FEASIBLE_ROSTER.copy()
    x[0, 0] = 0
    with pytest.raises(ValueError, match="history"):
        repair(x, ctx, BURKE_CURTOIS)


def test_construction_from_a_random_roster_on_instance1():
    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "Instance1.txt"
    if not raw.exists():
        pytest.skip("benchmark files not present")
    inst = read_bc(raw)
    ctx = Context(inst)
    x = random_roster(inst, BURKE_CURTOIS, np.random.default_rng(0))
    res = repair(x, ctx, BURKE_CURTOIS, w0=inst.n_days, **QUICK)
    assert res.feasible and res.J == 607            # the maximal window is the full problem


@pytest.mark.parametrize("seed", [10, 12, 14, 15])
def test_anytime_reaches_the_exact_optimum_and_is_never_worse(seed):
    inst = synthetic_instance(5, 7, 3, seed=seed)
    ctx = Context(inst)
    exact = solve(ctx, BURKE_CURTOIS, workers=1, deterministic_limit=60)
    x = random_masked_roster(ctx, seed)
    first = repair(x, ctx, BURKE_CURTOIS, w0=1, anytime=False, **QUICK)
    anytime = repair(x, ctx, BURKE_CURTOIS, w0=1, **QUICK)                 # the default
    assert anytime.feasible and anytime.J <= first.J + 1e-9
    assert anytime.J == pytest.approx(exact.J)
    assert anytime.windows == [1, 2, 4, 8]         # doubled until the window covers the horizon (m = 7)
    assert set(anytime.statuses) == {"OPTIMAL"}
    assert first.windows == [1]                    # the first-feasible form stops at the first success
