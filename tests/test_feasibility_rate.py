import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")

import nrrp.cpsat as cpsat  # noqa: E402
from nrrp.cpsat import solve  # noqa: E402
from nrrp.generator import synthetic_instance  # noqa: E402
from nrrp.instance import OFF, Context  # noqa: E402
from nrrp.profile import BURKE_CURTOIS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("feasibility_rate", ROOT / "scripts" / "feasibility_rate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_uncredited_reading_holds_only_inside_the_block():
    inst = synthetic_instance(8, 14, 2, seed=3)
    x0 = solve(Context(inst), BURKE_CURTOIS, workers=1, deterministic_limit=30).roster
    i = int(np.flatnonzero((x0 != OFF).sum(axis=1) >= 3)[0])
    days = np.flatnonzero(x0[i] != OFF)[1:3]
    absent = np.zeros_like(x0, bool)
    absent[i, days] = True
    ctx = Context(inst, d_xi=int(days[0]), published=x0, absent=absent, mu=1.0)
    credited, encoder = ctx.min_minutes.copy(), cpsat._interior_run_patterns
    assert credited[i] < inst.min_minutes[i] or inst.min_minutes[i] == 0
    with load_script().uncredited(ctx):
        np.testing.assert_array_equal(ctx.min_minutes, inst.min_minutes)       # no credit
        assert cpsat._interior_run_patterns is not encoder                     # no exemption from (9)
    np.testing.assert_array_equal(ctx.min_minutes, credited)
    assert cpsat._interior_run_patterns is encoder


def test_the_script_end_to_end(tmp_path):
    from nrrp.bc_format import write_bc, write_roster
    inst = synthetic_instance(6, 14, 2, seed=2, name="MiniBC")
    res = solve(Context(inst), BURKE_CURTOIS, workers=1, deterministic_limit=30)
    assert res.has_solution and res.feasible
    (tmp_path / "solutions").mkdir()
    write_bc(inst, tmp_path / "MiniBC.txt")
    write_roster(res.roster, inst, tmp_path / "solutions" / f"MiniBC.Solution.{res.J:g}.roster", penalty=res.J)
    out = subprocess.run([sys.executable, "scripts/feasibility_rate.py", "--instance", str(tmp_path / "MiniBC.txt"),
                          "--n-absent", "1", "2", "--absence-days", "2", "5", "--n", "6", "--time-limit", "5",
                          "--workers", "1", "--results", str(tmp_path / "results")],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    folder = tmp_path / "results" / "MiniBC-a1-2-l2-5"
    summary = json.loads((folder / "run.json").read_text())["summary"]
    assert all(sum(summary[r][k] for k in ("feasible", "infeasible", "undecided")) == 6
               for r in ("credited", "uncredited"))
    status = {}
    for row in csv.DictReader((folder / "results.csv").open()):
        status.setdefault(row["seed"], {})[row["reading"]] = row["status"]
    assert len(status) == 6
    for s in status.values():                    # the credit only relaxes: X(uncredited) is in X(credited)
        if s["uncredited"] == "feasible":
            assert s["credited"] != "infeasible"
