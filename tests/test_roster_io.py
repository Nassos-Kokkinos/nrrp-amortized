"""The .roster solution format: reader, writer, and the check against published values."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nrrp.bc_format import (FormatError, format_roster, parse_bc, parse_roster, read_bc, read_roster,
                            roster_penalty, solution_name)
from nrrp.generator import synthetic_instance
from nrrp.instance import OFF, Context
from nrrp.objective import J_terms, evaluate
from nrrp.profile import BURKE_CURTOIS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
MINI = DATA / "mini_bc.txt"
MINI_SOL = DATA / "mini_bc.Solution.855.roster"

# mini_bc with employee C's minimum lowered from 1920 to 960 minutes, and a feasible
# roster of it, checked by hand: J = cover only, 100 (d0 L) + 100 (d4 E) + 50 (d6 L) = 250.
FEASIBLE_TEXT = MINI.read_text().replace("C,E=7|L=3,2880,1920,", "C,E=7|L=3,2880,960,")
FEASIBLE_ROSTER = np.array([[OFF, 0, 1, 1, 1, OFF, OFF],
                            [OFF, OFF, 0, 0, 0, 0, 0],
                            [0, 0, OFF, OFF, OFF, OFF, OFF]], np.int8)


def roster_file(inst, body):
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<Roster>\n'
            '<SchedulingPeriodFile>x.ros</SchedulingPeriodFile>\n' + body + "\n</Roster>\n")


def test_hand_written_solution():
    inst = read_bc(MINI)
    r = read_roster(MINI_SOL, inst)
    want = np.full((3, 7), OFF, np.int8)
    want[0, 1] = want[0, 2] = 0
    want[1, 0] = want[1, 4] = 1
    want[2, 0], want[2, 5] = 0, 1
    np.testing.assert_array_equal(r, want)   # the <Employee>s inside <Violations> are not assignments
    assert r.dtype == np.int8
    assert roster_penalty(MINI_SOL.read_bytes()) == 855
    assert solution_name(MINI_SOL) == ("mini_bc", 855.0)


def test_J_of_the_hand_written_solution_is_the_hand_computed_value():
    inst = read_bc(MINI)
    terms = J_terms(read_roster(MINI_SOL, inst), Context(inst), BURKE_CURTOIS)
    assert terms == {"requests": 4.0, "cover": 851.0, "stability": 0.0}


def test_feasible_roster_by_hand():
    inst = parse_bc(FEASIBLE_TEXT, name="mini_feasible")
    ev = evaluate(FEASIBLE_ROSTER, Context(inst), BURKE_CURTOIS)
    assert ev["feasible"] and ev["J"] == 250


@pytest.mark.parametrize("size", [(1, 7, 1), (5, 7, 3), (12, 14, 3), (20, 28, 6)])
def test_round_trip(size):
    inst = synthetic_instance(*size, seed=5)
    r = np.random.default_rng(1).integers(OFF, inst.n_shifts, size=(inst.n_emp, inst.n_days)).astype(np.int8)
    text = format_roster(r, inst, penalty=12.5)
    np.testing.assert_array_equal(parse_roster(text, inst), r)
    np.testing.assert_array_equal(parse_roster(text.encode(), inst), r)
    assert roster_penalty(text) == 12.5
    assert roster_penalty(format_roster(r, inst)) is None


def test_hand_written_file_round_trip():
    inst = read_bc(MINI)
    r = read_roster(MINI_SOL, inst)
    np.testing.assert_array_equal(parse_roster(format_roster(r, inst), inst), r)


def test_solution_names():
    assert solution_name("a/b/Instance12.Solution.4040.roster") == ("Instance12", 4040.0)
    assert solution_name("Instance3.Solution.1001.5.roster") == ("Instance3", 1001.5)
    assert solution_name("Instance21_22215.txt") is None
    assert solution_name("Instance1.roster") is None


@pytest.mark.parametrize("body, message", [
    ('<Employee ID="Z"></Employee>', "unknown employee id 'Z'"),
    ('<Employee ID="A"><Assign><Day>1</Day><Shift>Q</Shift></Assign></Employee>', "unknown shift id 'Q'"),
    ('<Employee ID="A"><Assign><Day>7</Day><Shift>E</Shift></Assign></Employee>', "day 7 outside 0..6"),
    ('<Employee ID="A"><Assign><Day>x</Day><Shift>E</Shift></Assign></Employee>', "day must be an integer"),
    ('<Employee ID="A"><Assign><Day>1</Day><Shift>E</Shift></Assign>'
     '<Assign><Day>1</Day><Shift>L</Shift></Assign></Employee>', "two shifts on day 1"),
    ('<Employee ID="A"></Employee><Employee ID="A"></Employee>', "listed twice"),
    ('<Penalty>abc</Penalty>', None),
])
def test_errors(body, message):
    inst = read_bc(MINI)
    text = roster_file(inst, body)
    if message is None:                                        # a bad <Penalty> only matters to roster_penalty
        parse_roster(text, inst)
        with pytest.raises(FormatError, match="<Penalty> must be a number"):
            roster_penalty(text)
        return
    with pytest.raises(FormatError, match=message):
        parse_roster(text, inst, name="bad.roster")


def test_malformed_xml_and_wrong_root():
    inst = read_bc(MINI)
    with pytest.raises(FormatError, match="not well-formed"):
        parse_roster("<Roster><Employee ID='A'></Roster>", inst)
    with pytest.raises(FormatError, match="expected <Roster>"):
        parse_roster("<SchedulingPeriod/>", inst)


def test_writer_refuses_a_wrong_shape():
    inst = read_bc(MINI)
    with pytest.raises(ValueError, match="roster shape"):
        format_roster(np.full((2, 7), OFF, np.int8), inst)


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_check_script_passes_on_a_correct_solution(tmp_path):
    inst = parse_bc(FEASIBLE_TEXT, name="mini_feasible")
    (tmp_path / "mini_feasible.txt").write_text(FEASIBLE_TEXT)
    sol = tmp_path / "mini_feasible.Solution.250.roster"
    sol.write_text(format_roster(FEASIBLE_ROSTER, inst, penalty=250))
    out = run("scripts/check_solutions.py", str(sol), "--instances", str(tmp_path))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "1/1 solutions" in out.stdout and "ALL CHECKS PASS" in out.stdout


def test_check_script_reports_wrong_values_and_infeasibility(tmp_path):
    inst = parse_bc(FEASIBLE_TEXT, name="mini_feasible")
    (tmp_path / "mini_feasible.txt").write_text(FEASIBLE_TEXT)
    wrong = tmp_path / "mini_feasible.Solution.251.roster"
    wrong.write_text(format_roster(FEASIBLE_ROSTER, inst))
    out = run("scripts/check_solutions.py", str(wrong), str(MINI_SOL), "--instances", str(tmp_path))
    assert out.returncode == 1
    assert "J differs from the file name by -1" in out.stdout
    assert "cannot be read" in out.stdout                       # mini_bc.txt is not in tmp_path
    out = run("scripts/check_solutions.py", str(MINI_SOL), "--instances", str(DATA))
    assert out.returncode == 1
    assert "hard constraints violated" in out.stdout            # J is right (855), feasibility is not
    assert "J differs" not in out.stdout
    assert "requests term" not in out.stdout and "cover term" not in out.stdout   # the listing matches J
