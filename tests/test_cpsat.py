"""The CP-SAT model against the reference definitions (constraints.py, objective.py),
against brute force, and on instances with known optima."""
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")

from nrrp import constraints as constraints_module  # noqa: E402
from nrrp.bc_format import parse_bc, read_bc  # noqa: E402
from nrrp.constraints import CONSTRAINTS, random_block_row  # noqa: E402
from nrrp.cpsat import RosterModel, solve  # noqa: E402
from nrrp.generator import synthetic_instance  # noqa: E402
from nrrp.instance import OFF, Context  # noqa: E402
from nrrp.objective import J, J_terms, hard_violations  # noqa: E402
from nrrp.profile import BURKE_CURTOIS, ConstraintProfile  # noqa: E402
from nrrp.validate import TEST_PROFILES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "data" / "mini_bc.txt"
RAW = ROOT / "data" / "raw"
QUICK = dict(workers=1, deterministic_limit=30.0)

# mini_bc with C's minimum lowered to 960 minutes (see test_roster_io): by exhaustive
# search over all feasible rows its optimum is 250, attained by this roster (checked by hand)
FEASIBLE_TEXT = MINI.read_text().replace("C,E=7|L=3,2880,1920,", "C,E=7|L=3,2880,960,")


def contexts(seed):
    """Construction and re-rostering contexts on instances with and without skills and cover."""
    rng = np.random.default_rng(seed)
    out = []
    for k, inst in enumerate([synthetic_instance(4, 7, 2, seed=1),
                              synthetic_instance(5, 10, 3, seed=3, skills=True, min_cover=True),
                              synthetic_instance(3, 15, 4, seed=4, skills=True)]):
        I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
        for rep in range(8):
            roster = (np.stack([random_block_row(D, S, rng, p_work=rng.uniform(0.3, 0.8)) for _ in range(I)])
                      if rep % 2 else rng.integers(OFF, S, size=(I, D)).astype(np.int8))
            if rep % 3 == 0:
                ctx = Context(inst)
            else:
                d_xi = int(rng.integers(1, D))
                absent = np.zeros((I, D), bool)
                absent[rng.integers(I), d_xi:d_xi + int(rng.integers(1, 4))] = True
                pub = rng.integers(OFF, S, size=(I, D)).astype(np.int8)
                ctx = Context(inst, d_xi=d_xi, published=pub, absent=absent, mu=float(rng.choice([0.5, 2.0])))
                roster[:, :d_xi] = pub[:, :d_xi]
            out.append((ctx, roster))
    return out


CASES = contexts(0)


def fixed(roster):
    return dict(free=np.zeros(roster.shape, bool), base=roster)


@pytest.mark.parametrize("name", list(CONSTRAINTS))
def test_each_constraint_alone_matches_its_reference_definition(name):
    c = CONSTRAINTS[name]
    others = tuple(n for n in CONSTRAINTS if n not in (name, "unavailability"))
    hard = ConstraintProfile("hard_only", hard=tuple({name, "unavailability"}), off=others)
    pen = (ConstraintProfile("pen_only", hard=("unavailability",), penalised={name: 1.0}, off=others)
           if name != "unavailability" else None)
    violated = 0
    for ctx, roster in CASES:
        v = float(c.violation(roster, ctx).sum())
        v_unav = float(CONSTRAINTS["unavailability"].violation(roster, ctx).sum())
        violated += v > 0
        status = RosterModel(ctx, hard, **fixed(roster)).solve(**QUICK).status
        assert (status == "OPTIMAL") == (v == 0 and v_unav == 0), (name, ctx.name, status, v)
        if pen is not None and not v_unav:
            res = RosterModel(ctx, pen, **fixed(roster)).solve(**QUICK)
            assert res.status == "OPTIMAL"
            assert res.info["terms"][f"penalty:{name}"] == pytest.approx(v, abs=1e-9), (name, ctx.name)
    assert violated >= 3, f"{name}: too few violated cases to test anything"


@pytest.mark.parametrize("pname", list(TEST_PROFILES))
def test_every_term_of_J_on_fixed_rosters(pname):
    prof = TEST_PROFILES[pname]
    n_feasible = 0
    for ctx, roster in CASES:
        res = RosterModel(ctx, prof, **fixed(roster)).solve(**QUICK)
        feasible = not any(v.sum() for v in hard_violations(roster, ctx, prof).values())
        assert (res.status == "OPTIMAL") == feasible
        if feasible:
            n_feasible += 1
            want = J_terms(roster, ctx, prof)
            assert set(res.info["terms"]) == set(want)
            for k, v in want.items():
                assert res.info["terms"][k] == pytest.approx(v, abs=1e-9), (pname, ctx.name, k)
            assert res.objective == pytest.approx(sum(want.values()), abs=1e-9)
            assert res.J == pytest.approx(sum(want.values()), abs=1e-9)
    if pname == "all_soft":            # only absences are hard there; random rosters are mostly feasible
        assert n_feasible >= len(CASES) // 2


def brute_force_optimum(ctx, prof):
    inst = ctx.instance
    I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
    days = list(range(ctx.d_xi, D))
    best = None
    for cells in itertools.product(range(OFF, S), repeat=I * len(days)):
        r = ctx.published.copy()
        r[:, days] = np.array(cells, np.int8).reshape(I, len(days))
        if any(v.sum() for v in hard_violations(r, ctx, prof).values()):
            continue
        j = J(r, ctx, prof)
        best = j if best is None or j < best else best
    return best


@pytest.mark.parametrize("seed, d_xi", [(0, 0), (1, 2), (2, 2), (3, 1)])
@pytest.mark.parametrize("pname", ["burke_curtois", "all_soft"])
def test_optimum_equals_brute_force(seed, d_xi, pname):
    inst = synthetic_instance(2, 4, 2, seed=seed, skills=seed == 2, min_cover=seed == 3)
    inst.min_minutes[:] = inst.min_minutes // 3
    rng = np.random.default_rng(seed)
    if d_xi:
        pub = rng.integers(OFF, 2, size=(2, 4)).astype(np.int8)
        absent = np.zeros((2, 4), bool)
        absent[seed % 2, d_xi] = True
        ctx = Context(inst, d_xi=d_xi, published=pub, absent=absent, mu=1.5)
    else:
        ctx = Context(inst)
    prof = TEST_PROFILES[pname]
    best = brute_force_optimum(ctx, prof)
    res = solve(ctx, prof, **QUICK)
    if best is None:
        assert res.status == "INFEASIBLE"
    else:
        assert res.status == "OPTIMAL" and res.feasible
        assert res.J == pytest.approx(best, abs=1e-9)
        assert res.objective == pytest.approx(best, abs=1e-9)


def test_hand_checked_optimum():
    inst = parse_bc(FEASIBLE_TEXT, name="mini_feasible")
    res = solve(Context(inst), BURKE_CURTOIS, **QUICK)
    assert res.status == "OPTIMAL" and res.feasible
    assert res.J == 250 and res.objective == 250 and res.bound == 250


def test_mini_bc_has_no_feasible_roster():
    res = solve(Context(read_bc(MINI)), BURKE_CURTOIS, **QUICK)
    assert res.status == "INFEASIBLE" and not res.has_solution


@pytest.mark.skipif(not (RAW / "Instance1.txt").exists(), reason="benchmark files not present")
def test_instance1_published_optimum():
    res = solve(Context(read_bc(RAW / "Instance1.txt")), BURKE_CURTOIS, **QUICK)
    assert res.status == "OPTIMAL" and res.feasible and res.J == 607


def test_free_cells_and_the_rest_fixed():
    inst = synthetic_instance(5, 10, 3, seed=6)
    prof = TEST_PROFILES["all_soft"]
    base = np.random.default_rng(2).integers(OFF, 3, size=(5, 10)).astype(np.int8)
    free = np.zeros((5, 10), bool)
    free[1, 3:8] = True
    free[3, :] = True
    res = solve(Context(inst), prof, free=free, base=base, **QUICK)
    assert res.status == "OPTIMAL"
    np.testing.assert_array_equal(res.roster[~free], base[~free])
    assert res.J <= J(base, Context(inst), prof) + 1e-9      # the base itself is a candidate


def test_frozen_history_is_kept():
    inst = synthetic_instance(4, 10, 2, seed=8)
    rng = np.random.default_rng(3)
    pub = rng.integers(OFF, 2, size=(4, 10)).astype(np.int8)
    ctx = Context(inst, d_xi=4, published=pub, mu=1.0)
    res = solve(ctx, TEST_PROFILES["all_soft"], **QUICK)
    assert res.status == "OPTIMAL"
    np.testing.assert_array_equal(res.roster[:, :4], pub[:, :4])


def test_non_integer_weights_are_scaled():
    inst = synthetic_instance(4, 7, 2, seed=1)
    prof = BURKE_CURTOIS.with_penalised({"min_total_minutes": 0.125, "min_consecutive_shifts": 2.5})
    m = RosterModel(Context(inst), prof)
    assert m.scale == 1000
    res = m.solve(**QUICK)
    assert res.status == "OPTIMAL"
    assert res.objective == pytest.approx(res.J, abs=1e-9)


def test_weights_finer_than_the_scale_limit_are_refused():
    inst = synthetic_instance(3, 7, 1, seed=1)
    prof = BURKE_CURTOIS.with_penalised({"min_total_minutes": 1 / 3, "min_consecutive_shifts": 1.0})
    with pytest.raises(ValueError, match="CP-SAT needs integers"):
        RosterModel(Context(inst), prof)


def test_a_constraint_without_an_encoding_is_refused(monkeypatch):
    class Fake(constraints_module.DaysOff):
        name = "fake_constraint"
    patched = dict(CONSTRAINTS, fake_constraint=Fake())
    monkeypatch.setattr("nrrp.cpsat.CONSTRAINTS", patched)
    with pytest.raises(NotImplementedError, match="fake_constraint"):
        RosterModel(Context(synthetic_instance(3, 7, 1, seed=1)), BURKE_CURTOIS)


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_solve_and_plot_scripts(tmp_path):
    inst_file = tmp_path / "mini_feasible.txt"
    inst_file.write_text(FEASIBLE_TEXT)
    out = run("scripts/solve_cp.py", str(inst_file), "--root", str(tmp_path / "res"), "--run", "t",
              "--time-limit", "30", "--workers", "1")
    assert out.returncode == 0, out.stdout + out.stderr
    run_dir = tmp_path / "res" / "t"
    lines = (run_dir / "results.csv").read_text().splitlines()
    assert len(lines) == 2 and lines[1].startswith("mini_feasible,3,7,2,OPTIMAL,250.0,250.0")
    assert (run_dir / "run.json").exists() and (run_dir / "solutions" / "mini_feasible.cp.roster").exists()
    sol = read_bc(inst_file)
    from nrrp.bc_format import read_roster
    r = read_roster(run_dir / "solutions" / "mini_feasible.cp.roster", sol)
    assert J(r, Context(sol), BURKE_CURTOIS) == 250
    pytest.importorskip("matplotlib")
    out = run("scripts/plot_results.py", "cp", str(run_dir))
    assert out.returncode == 0, out.stdout + out.stderr
    for stem in ("gap_to_best_known", "solve_time"):
        assert (run_dir / "figures" / f"{stem}.pdf").stat().st_size > 1000
