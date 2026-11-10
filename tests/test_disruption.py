import numpy as np
import pytest

pytest.importorskip("ortools")

from nrrp.cpsat import solve  # noqa: E402
from nrrp.disruption import DisruptionModel, changes, without_absences  # noqa: E402
from nrrp.env import RosterEnv  # noqa: E402
from nrrp.generator import synthetic_instance  # noqa: E402
from nrrp.instance import OFF, Context  # noqa: E402
from nrrp.objective import J_terms  # noqa: E402
from nrrp.profile import BURKE_CURTOIS, burke_curtois_training  # noqa: E402
from nrrp.repair import repair  # noqa: E402


@pytest.fixture(scope="module")
def published():
    inst = synthetic_instance(8, 14, 2, seed=3)
    res = solve(Context(inst), BURKE_CURTOIS, workers=1, deterministic_limit=30)
    assert res.has_solution and res.feasible
    return inst, res.roster


def test_samples_follow_the_model(published):
    inst, x0 = published
    P = DisruptionModel(inst, x0, n_absent=(1, 3), days=(2, 4), mu=2.5)
    rng = np.random.default_rng(0)
    for _ in range(200):
        ctx = P.sample(rng)
        assert 1 <= ctx.d_xi < inst.n_days and ctx.mu == 2.5
        np.testing.assert_array_equal(ctx.published, x0)
        who = np.flatnonzero(ctx.absent.any(axis=1))
        assert 1 <= len(who) <= 3
        for i in who:
            days = np.flatnonzero(ctx.absent[i])
            assert days[0] == ctx.d_xi                                   # reported on the day it starts
            assert np.array_equal(days, np.arange(days[0], days[-1] + 1))  # one stretch
            assert len(days) <= 4 and (len(days) >= 2 or days[-1] == inst.n_days - 1)
        if (x0[:, ctx.d_xi] != OFF).sum() >= 3:
            assert (x0[who, ctx.d_xi] != OFF).all()                      # absent among those working that day


def test_a_seed_gives_the_same_context(published):
    inst, x0 = published
    P = DisruptionModel(inst, x0)
    a, b = P.context(17), P.context(17)
    assert a.d_xi == b.d_xi and a.name == b.name
    np.testing.assert_array_equal(a.absent, b.absent)
    assert any(P.context(s).d_xi != a.d_xi or not np.array_equal(P.context(s).absent, a.absent)
               for s in range(18, 30))


def test_bad_parameters_are_refused(published):
    inst, x0 = published
    with pytest.raises(ValueError):
        DisruptionModel(inst, x0[:, :-1])
    with pytest.raises(ValueError):
        DisruptionModel(inst, x0, n_absent=(0, 1))
    with pytest.raises(ValueError):
        DisruptionModel(inst, x0, days=(3, 2))
    with pytest.raises(ValueError):
        DisruptionModel(inst, x0, n_absent=(1, inst.n_emp + 1))
    with pytest.raises(ValueError):
        DisruptionModel(inst, x0, mu=-1)


def test_the_environment_runs_on_fresh_contexts_and_keeps_the_history(published):
    inst, x0 = published
    P = DisruptionModel(inst, x0, n_absent=(1, 2), days=(1, 5))
    env = RosterEnv(inst, burke_curtois_training(w_min_minutes=0.2, w_min_consecutive=100.0), contexts=P, eval_profile=BURKE_CURTOIS,
                    features="extended", seed=0)
    rng = np.random.default_rng(1)
    for _ in range(5):
        env.reset()
        ctx, done, info = env.ctx, False, {}
        while not done:
            mask = env.action_mask()
            _, _, done, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        np.testing.assert_array_equal(env.roster[:, :ctx.d_xi], x0[:, :ctx.d_xi])
        assert (env.roster[ctx.absent] == OFF).all()                       # the mask excludes absences
        assert "evaluation" in info


def test_repairing_the_published_roster_without_the_absences(published):
    inst, x0 = published
    P = DisruptionModel(inst, x0, n_absent=(1, 1), days=(2, 3), mu=1.0)
    ctx = P.context(5)
    x = without_absences(ctx)
    np.testing.assert_array_equal(x[:, :ctx.d_xi], x0[:, :ctx.d_xi])
    assert (x[ctx.absent] == OFF).all()
    assert changes(x, ctx) == int((x0[ctx.absent] != OFF).sum())
    assert J_terms(x, ctx, BURKE_CURTOIS)["stability"] == pytest.approx(ctx.mu * changes(x, ctx))
    res = repair(x, ctx, BURKE_CURTOIS, workers=1, budget=60.0, deterministic_budget=60.0)
    exact = solve(ctx, BURKE_CURTOIS, workers=1, deterministic_limit=60)
    assert res.feasible == (exact.has_solution and exact.feasible)
    if res.feasible:
        assert (res.roster[ctx.absent] == OFF).all()
        assert res.J >= exact.J - 1e-6


def test_absences_are_credited_towards_the_minimum_minutes(published):
    """Decision of 10/11/2026: the shifts x° had on the absent days count towards H^min."""
    from nrrp.constraints import MinTotalMinutes
    inst, x0 = published
    i = int(np.flatnonzero((x0 != OFF).sum(axis=1) >= 3)[0])
    days = np.flatnonzero(x0[i] != OFF)[1:3]                    # two worked days, after day 0
    absent = np.zeros_like(x0, bool)
    absent[i, days] = True
    ctx = Context(inst, d_xi=int(days[0]), published=x0, absent=absent, mu=1.0)
    lost = int(inst.shift_len[x0[i, days]].sum())
    assert ctx.credit[i] == lost and ctx.credit.sum() == lost
    assert ctx.min_minutes[i] == max(int(inst.min_minutes[i]) - lost, 0)
    np.testing.assert_array_equal(np.delete(ctx.min_minutes, i), np.delete(inst.min_minutes, i))
    x = without_absences(ctx)                                    # x° less the two cancelled shifts
    assert MinTotalMinutes().violation_row(i, x[i], ctx) == 0    # the credit covers them
    assert (Context(inst).credit == 0).all()                     # construction: nothing credited


def test_a_work_stretch_next_to_an_absence_is_exempt():
    """An absence acts like an end of the horizon for the minimum stretch (9)."""
    from nrrp.constraints import MinConsecutiveShifts
    inst = synthetic_instance(3, 10, 1, seed=0)
    inst.min_consec[:] = 3
    c = MinConsecutiveShifts()
    row = np.full(10, OFF, np.int8)
    row[3:5] = 0                                                 # a 2-day stretch, days 3-4
    pub = np.full((3, 10), OFF, np.int8)

    def absent_on(day):
        absent = np.zeros((3, 10), bool)
        if day is not None:
            absent[0, day] = True
        return Context(inst, d_xi=2, published=pub, absent=absent)

    assert c.violation_row(0, row, absent_on(None)) == 1         # interior, one day short
    assert c.violation_row(0, row, absent_on(2)) == 0            # absence just before: exempt
    assert c.violation_row(0, row, absent_on(5)) == 0            # absence just after: exempt
    assert c.violation_row(0, row, absent_on(7)) == 1            # further away: no effect
