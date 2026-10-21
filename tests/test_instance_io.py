from pathlib import Path

import numpy as np
import pytest

from nrrp.bc_format import FormatError, format_bc, parse_bc, read_bc
from nrrp.generator import synthetic_instance
from nrrp.instance import OFF, Context, binary_to_roster, roster_to_binary

MINI = Path(__file__).parent / "data" / "mini_bc.txt"

ARRAYS = ("shift_len", "forbidden", "max_shifts", "max_minutes", "min_minutes", "max_consec",
          "min_consec", "min_consec_off", "max_weekends", "day_off", "on_req", "off_req",
          "rho", "w_under", "w_over")


def same_instance(a, b):
    assert a.employee_ids == b.employee_ids
    assert a.shift_ids == b.shift_ids
    assert a.n_days == b.n_days
    for f in ARRAYS:
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)


def test_parse_hand_written_file():
    inst = read_bc(MINI)
    assert inst.name == "mini_bc"
    assert (inst.n_emp, inst.n_days, inst.n_shifts) == (3, 7, 2)
    assert inst.employee_ids == ("A", "B", "C") and inst.shift_ids == ("E", "L")
    np.testing.assert_array_equal(inst.shift_len, [480, 600])
    np.testing.assert_array_equal(inst.forbidden, [[False, False], [True, False]])  # E may not follow L
    np.testing.assert_array_equal(inst.max_shifts, [[7, 7], [5, 0], [7, 3]])
    np.testing.assert_array_equal(inst.max_minutes, [2400, 2400, 2880])
    np.testing.assert_array_equal(inst.min_minutes, [1440, 960, 1920])
    np.testing.assert_array_equal(inst.max_consec, [4, 5, 3])
    np.testing.assert_array_equal(inst.min_consec, [2, 1, 2])
    np.testing.assert_array_equal(inst.min_consec_off, [2, 1, 1])
    np.testing.assert_array_equal(inst.max_weekends, [1, 1, 0])
    assert set(zip(*np.nonzero(inst.day_off))) == {(0, 0), (2, 2), (2, 3)}
    assert inst.on_req[0, 1, 0] == 2 and inst.on_req[1, 4, 0] == 3 and inst.on_req.sum() == 5
    assert inst.off_req[2, 5, 1] == 1 and inst.off_req.sum() == 1
    np.testing.assert_array_equal(inst.rho[:, 0], [1, 2, 1, 1, 2, 1, 1])
    np.testing.assert_array_equal(inst.rho[:, 1], [1, 0, 1, 1, 1, 0, 1])
    assert inst.w_under[6, 1] == 50 and inst.w_over[6, 1] == 2 and inst.w_under[0, 0] == 100
    np.testing.assert_array_equal(inst.weekend_of_day, [-1, -1, -1, -1, -1, 0, 0])


@pytest.mark.parametrize("size", [(1, 7, 1), (5, 7, 3), (12, 14, 3), (20, 28, 6)])
def test_round_trip(size):
    inst = synthetic_instance(*size, seed=3)
    same_instance(inst, parse_bc(format_bc(inst), name=inst.name))


def test_mini_file_round_trip():
    inst = read_bc(MINI)
    same_instance(inst, parse_bc(format_bc(inst), name="again"))


def test_missing_cover_pair_has_no_cover_term():
    text = MINI.read_text().replace("6,L,1,50,2\n", "")
    inst = parse_bc(text)
    assert inst.rho[6, 1] == 0 and inst.w_under[6, 1] == 0 and inst.w_over[6, 1] == 0


@pytest.mark.parametrize("old, new, message", [
    ("0,E,1,100,1", "0,X,1,100,1", "unknown shift id 'X'"),
    ("A,1,E,2", "A,7,E,2", "day 7 outside 0..6"),
    ("A,1,E,2", "Z,1,E,2", "unknown employee id 'Z'"),
    ("A,E=7|L=7,", "A,E=7,", "MaxShifts gives no limit"),
    ("C,5,L,1", "C,5,L", "expects 4 fields"),
    ("0,L,1,100,1", "0,E,1,100,1", "second cover line"),
    ("SECTION_COVER", "SECTION_COVERS", "unknown section"),
    ("A,E=7|L=7,2400,1440,4,2,2,1", "A,E=7|L=7,1440,2400,4,2,2,1", "min_minutes > max_minutes"),
])
def test_format_errors_name_the_problem(old, new, message):
    text = MINI.read_text()
    assert old in text
    with pytest.raises(FormatError, match=message):
        parse_bc(text.replace(old, new, 1), name="bad")


def test_format_errors_give_the_line():
    text = MINI.read_text().replace("0,E,1,100,1", "0,X,1,100,1")
    with pytest.raises(FormatError, match=r"bad, line 34: unknown shift"):  # first cover line
        parse_bc(text, name="bad")


def test_binary_roundtrip():
    y = np.array([[0, OFF, 1], [OFF, OFF, 0]], np.int8)
    x = roster_to_binary(y, 2)
    assert x.shape == (2, 3, 2) and x.sum() == 3
    np.testing.assert_array_equal(binary_to_roster(x), y)


def test_context_validation():
    inst = synthetic_instance(4, 7, 2, seed=0)
    ctx = Context(inst)
    assert ctx.is_construction and (ctx.y0 == OFF).all()
    with pytest.raises(ValueError, match="needs a published roster"):
        Context(inst, d_xi=2)
    absent = np.zeros((4, 7), bool)
    absent[0, 1] = True
    with pytest.raises(ValueError, match="before d_xi"):
        Context(inst, d_xi=3, y0=np.full((4, 7), OFF), absent=absent)
    rho = inst.rho.copy()
    rho[0, 0] += 1
    with pytest.raises(ValueError, match="cover before d_xi"):
        Context(inst, d_xi=2, y0=np.full((4, 7), OFF), rho=rho)
