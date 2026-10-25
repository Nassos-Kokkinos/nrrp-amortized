import numpy as np
import pytest

from nrrp.constraints import (CONSTRAINTS, ROW_CONSTRAINTS, RosterState, find_truncation_counterexample,
                              random_block_row, roster_counts, trailing_run, true_runs)
from nrrp.generator import synthetic_instance
from nrrp.instance import OFF, Context, Instance
from nrrp.profile import BURKE_CURTOIS, ConstraintProfile, burke_curtois_training

_ = OFF


def one_employee(D=14, **limits):
    """1 employee, D days, 2 shifts (0 = E, 1 = L; E may not follow L), loose limits."""
    base = dict(max_shifts=[[D, D]], max_minutes=[10_000], min_minutes=[0], max_consec=[D],
                min_consec=[0], min_consec_off=[0], max_weekends=[D])
    base.update(limits)
    return Context(Instance(
        name="one", employee_ids=["A"], shift_ids=["E", "L"], n_days=D, shift_len=[480, 600],
        forbidden=[[False, False], [True, False]], day_off=np.zeros((1, D), bool),
        on_req=np.zeros((1, D, 2)), off_req=np.zeros((1, D, 2)), rho=np.zeros((D, 2)),
        w_under=np.zeros((D, 2)), w_over=np.zeros((D, 2)), **base))


def viol(name, ctx, row):
    return CONSTRAINTS[name].violation_row(0, np.array(row, np.int8), ctx)


def test_runs_helpers():
    a, b = true_runs([1, 1, 0, 1, 0, 0, 1])
    assert list(a) == [0, 3, 6] and list(b) == [2, 4, 7]
    w = np.array([[1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], bool)
    assert list(trailing_run(w, 5)) == [2, 0, 5]
    assert list(trailing_run(~w, 5)) == [0, 5, 0]
    assert list(trailing_run(w, 0)) == [0, 0, 0]


# --------------------------------------------------------------- reference definitions

def test_max_consecutive_shifts():
    ctx = one_employee(7, max_consec=[3])
    assert viol("max_consecutive_shifts", ctx, [0, 0, 0, 0, _, _, _]) == 1  # day -1 is off
    assert viol("max_consecutive_shifts", ctx, [_, 0, 0, 0, 0, 0, _]) == 2
    assert viol("max_consecutive_shifts", ctx, [_, _, _, _, 0, 0, 0]) == 0


def test_min_consecutive_shifts_exempts_runs_touching_either_end():
    ctx = one_employee(7, min_consec=[3])
    assert viol("min_consecutive_shifts", ctx, [0, _, _, _, _, _, _]) == 0  # touches the start
    assert viol("min_consecutive_shifts", ctx, [_, _, _, _, _, _, 0]) == 0  # touches the end
    assert viol("min_consecutive_shifts", ctx, [_, 0, _, _, _, _, _]) == 2
    assert viol("min_consecutive_shifts", ctx, [_, 0, 0, _, 0, _, 0]) == 1 + 2 + 0


def test_min_consecutive_days_off_exempts_runs_touching_either_end():
    ctx = one_employee(7, min_consec_off=[2])
    assert viol("min_consecutive_days_off", ctx, [_, 0, 0, 0, 0, 0, 0]) == 0
    assert viol("min_consecutive_days_off", ctx, [0, 0, 0, 0, 0, 0, _]) == 0
    assert viol("min_consecutive_days_off", ctx, [0, _, 0, 0, 0, 0, 0]) == 1
    assert viol("min_consecutive_days_off", ctx, [0, _, 0, _, _, 0, 0]) == 1


def test_shift_rotation_and_counts():
    ctx = one_employee(7, max_shifts=[[2, 7]], max_minutes=[2000], min_minutes=[1500])
    assert viol("shift_rotation", ctx, [1, 0, 1, 0, _, 1, 1]) == 2  # L->E twice; L->L allowed
    assert viol("shift_rotation", ctx, [1, _, 0, _, _, _, _]) == 0  # a day off in between
    assert viol("max_shifts_per_type", ctx, [0, 0, 0, _, 1, _, _]) == 1
    assert viol("max_total_minutes", ctx, [0, 0, 1, 1, _, _, _]) == 2160 - 2000
    assert viol("min_total_minutes", ctx, [0, 0, _, _, _, _, _]) == 1500 - 960
    assert viol("min_total_minutes", ctx, [_, _, _, _, _, _, _]) == 1500


def test_weekends():
    ctx = one_employee(14, max_weekends=[1])  # days 5, 6 and 12, 13 are weekends
    row = [_] * 14
    row[5] = row[6] = 0
    assert viol("max_weekends", ctx, row) == 0
    row[13] = 1
    assert viol("max_weekends", ctx, row) == 1


def test_days_off_and_unavailability():
    ctx = one_employee(7)
    ctx.instance.day_off[0, 2] = True
    ctx.absent[0, 4] = True
    assert viol("days_off", ctx, [_, _, 0, _, 0, _, _]) == 1
    assert viol("unavailability", ctx, [_, _, 0, _, 0, _, _]) == 1


# -------------------------------------------------------- Lemma 1, as an executable test

def binding_context(seed=0):
    """Synthetic instance with every row constraint able to bind, plus absences and skills."""
    inst = synthetic_instance(6, 21, 3, seed=seed, skills=True)
    inst.min_consec[:] = 3
    inst.min_consec_off[:] = 2
    rng = np.random.default_rng(seed)
    return Context(inst, absent=rng.random((6, 21)) < 0.1)


@pytest.mark.parametrize("name", list(ROW_CONSTRAINTS))
def test_prefix_monotone_flag_matches_a_search_for_counterexamples(name):
    c = ROW_CONSTRAINTS[name]
    n_ok, cex = find_truncation_counterexample(c, binding_context(), np.random.default_rng(1), 3000)
    if c.prefix_monotone:
        assert cex is None, f"{name} is flagged prefix-monotone but {cex} is a counterexample"
        assert n_ok > 50, "too few satisfying rows for the absence of a counterexample to mean much"
    else:
        assert cex is not None, f"{name} is flagged not prefix-monotone but no counterexample found"


def test_burke_curtois_profile_split():
    p = BURKE_CURTOIS
    assert {c.name for c in p.pre} == {
        "shift_rotation", "max_shifts_per_type", "max_total_minutes", "max_consecutive_shifts",
        "min_consecutive_days_off", "max_weekends", "days_off", "unavailability", "skills"}
    assert {c.name for c in p.ter} == {"min_total_minutes", "min_consecutive_shifts"}
    assert p.penalties == () and p.off == ("min_coverage",)


def test_training_profile_needs_explicit_weights():
    p = burke_curtois_training(w_min_minutes=0.2, w_min_consecutive=30)
    assert p.ter == () and {c.name for c in p.pre} == {c.name for c in BURKE_CURTOIS.pre}
    with pytest.raises(TypeError):
        burke_curtois_training()


def test_profile_with_penalised_and_validation():
    p = BURKE_CURTOIS.with_penalised({"min_total_minutes": 0.2, "min_consecutive_shifts": 30})
    assert p.ter == ()
    assert [(c.name, w) for c, w in p.penalties] == [("min_total_minutes", 0.2),
                                                      ("min_consecutive_shifts", 30.0)]
    with pytest.raises(ValueError, match="must be hard"):
        BURKE_CURTOIS.with_penalised({"unavailability": 1.0})
    with pytest.raises(ValueError, match="no role"):
        ConstraintProfile("x", hard=["unavailability"])
    with pytest.raises(ValueError, match="unknown"):
        BURKE_CURTOIS.with_penalised({"nonsense": 1.0})
    with pytest.raises(ValueError, match="positive"):
        BURKE_CURTOIS.with_penalised({"days_off": 0.0})
    with pytest.raises(ValueError, match="more than one role"):
        ConstraintProfile("x", hard=list(CONSTRAINTS), off=["days_off"])


# ---------------------------------------- incremental deltas against the reference rows

def random_partial_roster(ctx, d, rng):
    """Days < d decided at random, some employees already assigned on day d, the rest empty."""
    inst = ctx.instance
    y = np.full((inst.n_emp, inst.n_days), OFF, np.int8)
    for i in range(inst.n_emp):
        y[i, :d] = random_block_row(inst.n_days, inst.n_shifts, rng)[:d]
        if rng.random() < 0.3:
            y[i, d] = rng.integers(inst.n_shifts)
    return y


@pytest.mark.parametrize("seed", range(4))
def test_delta_equals_change_in_violation(seed):
    rng = np.random.default_rng(seed)
    ctx = binding_context(seed)
    inst = ctx.instance
    for _trial in range(25):
        d = int(rng.integers(inst.n_days))
        y = random_partial_roster(ctx, d, rng)
        st = RosterState(ctx, y, d)
        free = np.flatnonzero(y[:, d] == OFF)
        for s in range(inst.n_shifts):
            for name, c in ROW_CONSTRAINTS.items():
                got = c.delta(st, d, s)
                for i in free:
                    after = y[i].copy()
                    after[d] = s
                    want = c.violation_row(i, after, ctx) - c.violation_row(i, y[i], ctx)
                    assert got[i] == want, (name, d, s, i)


def test_delta_accepts_one_shift_per_employee():
    rng = np.random.default_rng(7)
    ctx = binding_context(7)
    inst = ctx.instance
    y = random_partial_roster(ctx, 9, rng)
    st = RosterState(ctx, y, 9)
    s_arr = rng.integers(inst.n_shifts, size=inst.n_emp)
    for c in ROW_CONSTRAINTS.values():
        vec = c.delta(st, 9, s_arr)
        for i in range(inst.n_emp):
            assert vec[i] == c.delta(st, 9, int(s_arr[i]))[i], c.name


def test_skills():
    ctx = one_employee(7, skills_held=[[True, False]], skill_required=[-1, 1])
    assert viol("skills", ctx, [0, 1, _, 1, _, 0, _]) == 2   # L needs the skill A lacks
    st = RosterState(ctx, np.full((1, 7), OFF, np.int8), 0)
    assert list(CONSTRAINTS["skills"].delta(st, 0, 1)) == [1.0]
    assert list(CONSTRAINTS["skills"].delta(st, 0, 0)) == [0.0]


def test_min_coverage_is_a_column_constraint_and_not_prefix_monotone():
    inst = synthetic_instance(5, 7, 2, seed=0, min_cover=True)
    ctx = Context(inst)
    c = CONSTRAINTS["min_coverage"]
    assert c.scope == "column" and not c.prefix_monotone
    full = np.zeros((5, 7), np.int8)                      # everybody on shift 0 every day
    v = c.violation(full, ctx)
    assert v.shape == (7, 2)
    np.testing.assert_array_equal(v[:, 0], np.maximum(inst.min_cover[:, 0] - 5, 0))
    empty = np.full((5, 7), OFF, np.int8)                  # its truncation to nothing
    assert c.violation(empty, ctx).sum() == inst.min_cover.sum() > 0
    for d, s in [(0, 0), (3, 1)]:
        inc = c.increments(ctx, d, s, 5)
        for n in range(6):
            assert inc[:n].sum() == max(inst.min_cover[d, s] - n, 0) - inst.min_cover[d, s]
    assert roster_counts(full, 2)[:, 0].tolist() == [5] * 7
