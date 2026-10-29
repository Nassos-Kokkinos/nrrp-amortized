"""Reader and writer for the Burke & Curtois text format.

Source of the format: https://www.schedulingbenchmarks.org/nrp/ (Instance1-24),
Burke & Curtois, EJOR 237(1) (2014) 71-81. Sections and columns, as in the
comment lines of the files:

==========================  =====================================================
SECTION_HORIZON             horizon length in days; all instances start on a Monday
SECTION_SHIFTS              ShiftID, Length in mins, Shifts which cannot follow
                            this shift (| separated, may be empty)
SECTION_STAFF               ID, MaxShifts (ShiftID=max, | separated),
                            MaxTotalMinutes, MinTotalMinutes, MaxConsecutiveShifts,
                            MinConsecutiveShifts, MinConsecutiveDaysOff, MaxWeekends
SECTION_DAYS_OFF            EmployeeID, DayIndexes (start at zero)
SECTION_SHIFT_ON_REQUESTS   EmployeeID, Day, ShiftID, Weight
SECTION_SHIFT_OFF_REQUESTS  EmployeeID, Day, ShiftID, Weight
SECTION_COVER               Day, ShiftID, Requirement, Weight for under,
                            Weight for over
==========================  =====================================================

A (day, shift) pair absent from SECTION_COVER has no cover term (requirement and
both weights 0). Errors name the file and line.

Solutions. The same site publishes solutions as XML rosters, one file per
solution named ``InstanceN.Solution.VALUE.roster`` with VALUE the objective
value; the format is RosterBooster's (``Roster.xsd``)::

    <Roster>
      <SchedulingPeriodFile>../../Instance1.ros</SchedulingPeriodFile>
      <Penalty>607</Penalty>                                   (optional)
      <Employee ID="A"><Assign><Day>1</Day><Shift>D</Shift></Assign> ...</Employee>
      ...
    </Roster>

Days start at zero, an employee-day with no ``Assign`` is a day off, and
employee and shift IDs are those of the instance. Other elements (solver
details, violation listings) are ignored. :func:`read_roster` turns a file into
a roster array for a given instance; :func:`format_roster` writes one.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, quoteattr
from pathlib import Path

import numpy as np

from .instance import OFF, Instance, empty_roster

HORIZON, SHIFTS, STAFF, DAYS_OFF, ON_REQ, OFF_REQ, COVER = (
    "SECTION_HORIZON", "SECTION_SHIFTS", "SECTION_STAFF", "SECTION_DAYS_OFF",
    "SECTION_SHIFT_ON_REQUESTS", "SECTION_SHIFT_OFF_REQUESTS", "SECTION_COVER")
SECTIONS = (HORIZON, SHIFTS, STAFF, DAYS_OFF, ON_REQ, OFF_REQ, COVER)
REQUIRED = (HORIZON, SHIFTS, STAFF, COVER)

HEADERS = {
    HORIZON: "# All instances start on a Monday\n# The horizon length in days:",
    SHIFTS: "# ShiftID, Length in mins, Shifts which cannot follow this shift | separated",
    STAFF: "# ID, MaxShifts, MaxTotalMinutes, MinTotalMinutes, MaxConsecutiveShifts, "
           "MinConsecutiveShifts, MinConsecutiveDaysOff, MaxWeekends",
    DAYS_OFF: "# EmployeeID, DayIndexes (start at zero)",
    ON_REQ: "# EmployeeID, Day, ShiftID, Weight",
    OFF_REQ: "# EmployeeID, Day, ShiftID, Weight",
    COVER: "# Day, ShiftID, Requirement, Weight for under, Weight for over",
}


# Published table of schedulingbenchmarks.org/nrp (retrieved 2026-10-11):
# name -> (weeks, employees, shift types, best known objective value).
PUBLISHED = {
    "Instance1": (2, 8, 1, 607), "Instance2": (2, 14, 2, 828), "Instance3": (2, 20, 3, 1001),
    "Instance4": (4, 10, 2, 1716), "Instance5": (4, 16, 2, 1143), "Instance6": (4, 18, 3, 1950),
    "Instance7": (4, 20, 3, 1056), "Instance8": (4, 30, 4, 1300), "Instance9": (4, 36, 4, 439),
    "Instance10": (4, 40, 5, 4631), "Instance11": (4, 50, 6, 3443), "Instance12": (4, 60, 10, 4040),
    "Instance13": (4, 120, 18, 1348), "Instance14": (6, 32, 4, 1278), "Instance15": (6, 45, 6, 3828),
    "Instance16": (8, 20, 3, 3225), "Instance17": (8, 32, 4, 5746), "Instance18": (12, 22, 3, 4459),
    "Instance19": (12, 40, 5, 3149), "Instance20": (26, 50, 6, 4769), "Instance21": (26, 100, 8, 21133),
    "Instance22": (52, 50, 10, 30241), "Instance23": (52, 100, 16, 17285),
    "Instance24": (52, 150, 32, 40439),
}


def published_mismatch(inst: Instance) -> str | None:
    """None if the instance has the size the published table gives for its name
    (or the name is not in the table); otherwise a description of the mismatch."""
    if inst.name not in PUBLISHED:
        return None
    weeks, n_emp, n_shifts, _ = PUBLISHED[inst.name]
    got, want = (inst.n_days, inst.n_emp, inst.n_shifts), (7 * weeks, n_emp, n_shifts)
    if got == want:
        return None
    return f"{inst.name}: parsed {got} (days, employees, shifts), published {want}"


class FormatError(ValueError):
    pass


def read_bc(path) -> Instance:
    path = Path(path)
    return parse_bc(path.read_text(encoding="utf-8-sig"), name=path.stem)


def parse_bc(text: str, name: str = "instance") -> Instance:
    rows = _split_sections(text, name)

    def err(lineno, msg):
        return FormatError(f"{name}, line {lineno}: {msg}")

    def fields(section, n_min, n_max=None):
        for lineno, f in rows.get(section, []):
            if len(f) < n_min or (n_max is not None and len(f) > n_max):
                want = n_min if n_max == n_min else f"{n_min}..{n_max if n_max else 'n'}"
                raise err(lineno, f"{section} expects {want} fields, got {len(f)}")
            yield lineno, f

    def integer(lineno, v, what):
        try:
            return int(v)
        except ValueError:
            raise err(lineno, f"{what} must be an integer, got {v!r}") from None

    def number(lineno, v, what):
        try:
            x = float(v)
        except ValueError:
            raise err(lineno, f"{what} must be a number, got {v!r}") from None
        if x < 0:
            raise err(lineno, f"{what} must be non-negative, got {v!r}")
        return x

    # horizon
    horizon = list(fields(HORIZON, 1, 1))
    if not horizon:
        raise FormatError(f"{name}: SECTION_HORIZON is empty")
    if len(horizon) > 1:
        raise err(horizon[1][0], "SECTION_HORIZON must hold one number")
    D = integer(horizon[0][0], horizon[0][1][0], "horizon")
    if D <= 0:
        raise err(horizon[0][0], "horizon must be positive")

    def day(lineno, v):
        d = integer(lineno, v, "day")
        if not 0 <= d < D:
            raise err(lineno, f"day {d} outside 0..{D - 1}")
        return d

    # shifts
    shift_rows = list(fields(SHIFTS, 2, 3))
    shift_ids = [f[0] for _, f in shift_rows]
    sidx = {}
    for lineno, f in shift_rows:
        if f[0] in sidx:
            raise err(lineno, f"duplicate shift id {f[0]!r}")
        sidx[f[0]] = len(sidx)
    S = len(sidx)

    def shift(lineno, v):
        if v not in sidx:
            raise err(lineno, f"unknown shift id {v!r}")
        return sidx[v]

    shift_len = np.zeros(S, np.int64)
    forbidden = np.zeros((S, S), bool)
    for lineno, f in shift_rows:
        a = sidx[f[0]]
        shift_len[a] = integer(lineno, f[1], "shift length")
        follow = f[2] if len(f) > 2 else ""
        for b in filter(None, (t.strip() for t in follow.split("|"))):
            forbidden[a, shift(lineno, b)] = True

    # staff
    staff_rows = list(fields(STAFF, 8, 8))
    emp_ids = [f[0] for _, f in staff_rows]
    eidx = {}
    for lineno, f in staff_rows:
        if f[0] in eidx:
            raise err(lineno, f"duplicate employee id {f[0]!r}")
        eidx[f[0]] = len(eidx)
    I = len(eidx)

    def employee(lineno, v):
        if v not in eidx:
            raise err(lineno, f"unknown employee id {v!r}")
        return eidx[v]

    max_shifts = np.zeros((I, S), np.int64)
    limits = np.zeros((I, 6), np.int64)
    for lineno, f in staff_rows:
        i = eidx[f[0]]
        seen = set()
        for item in filter(None, (t.strip() for t in f[1].split("|"))):
            if "=" not in item:
                raise err(lineno, f"MaxShifts entry {item!r} is not ShiftID=max")
            sid, v = (t.strip() for t in item.split("=", 1))
            s = shift(lineno, sid)
            max_shifts[i, s] = integer(lineno, v, "MaxShifts")
            seen.add(s)
        missing = [shift_ids[s] for s in range(S) if s not in seen]
        if missing:
            raise err(lineno, f"MaxShifts gives no limit for shift(s) {missing}")
        limits[i] = [integer(lineno, v, "staff limit") for v in f[2:8]]
    max_minutes, min_minutes, max_consec, min_consec, min_consec_off, max_weekends = limits.T

    # days off
    day_off = np.zeros((I, D), bool)
    for lineno, f in fields(DAYS_OFF, 1):
        i = employee(lineno, f[0])
        for v in filter(None, f[1:]):
            day_off[i, day(lineno, v)] = True

    # requests (duplicates add up)
    on_req = np.zeros((I, D, S))
    off_req = np.zeros((I, D, S))
    for section, target in ((ON_REQ, on_req), (OFF_REQ, off_req)):
        for lineno, f in fields(section, 4, 4):
            target[employee(lineno, f[0]), day(lineno, f[1]), shift(lineno, f[2])] += \
                number(lineno, f[3], "weight")

    # cover
    rho = np.zeros((D, S), np.int64)
    w_under = np.zeros((D, S))
    w_over = np.zeros((D, S))
    seen = set()
    for lineno, f in fields(COVER, 5, 5):
        d, s = day(lineno, f[0]), shift(lineno, f[1])
        if (d, s) in seen:
            raise err(lineno, f"second cover line for day {d}, shift {f[1]!r}")
        seen.add((d, s))
        rho[d, s] = integer(lineno, f[2], "requirement")
        w_under[d, s] = number(lineno, f[3], "under-cover weight")
        w_over[d, s] = number(lineno, f[4], "over-cover weight")

    try:
        return Instance(
            name=name, employee_ids=emp_ids, shift_ids=shift_ids, n_days=D,
            shift_len=shift_len, forbidden=forbidden, max_shifts=max_shifts,
            max_minutes=max_minutes, min_minutes=min_minutes, max_consec=max_consec,
            min_consec=min_consec, min_consec_off=min_consec_off, max_weekends=max_weekends,
            day_off=day_off, on_req=on_req, off_req=off_req, rho=rho,
            w_under=w_under, w_over=w_over, first_weekday=0)
    except ValueError as e:
        raise FormatError(f"{name}: {e}") from None


def _split_sections(text: str, name: str) -> dict:
    """{section: [(line number, [field, ...]), ...]} with comments and blank lines removed."""
    rows, section = {}, None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("SECTION_"):
            if line not in SECTIONS:
                raise FormatError(f"{name}, line {lineno}: unknown section {line!r}")
            if line in rows:
                raise FormatError(f"{name}, line {lineno}: {line} appears twice")
            section = line
            rows[section] = []
            continue
        if section is None:
            raise FormatError(f"{name}, line {lineno}: data before the first SECTION_ header")
        rows[section].append((lineno, [t.strip() for t in line.split(",")]))
    missing = [s for s in REQUIRED if s not in rows]
    if missing:
        raise FormatError(f"{name}: missing section(s) {', '.join(missing)}")
    return rows


def _num(x) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(x)


def format_bc(inst: Instance) -> str:
    """Text in the Burke & Curtois format; ``parse_bc(format_bc(inst))`` reproduces inst."""
    if inst.first_weekday != 0:
        raise ValueError("the format assumes the horizon starts on a Monday")
    if inst.min_cover.any() or inst.n_skills:
        raise ValueError("the format has no minimum cover and no skills")
    E, Sh = inst.employee_ids, inst.shift_ids
    out = [f"# {inst.name}, written by nrrp.bc_format", ""]

    def section(title, lines):
        out.extend([title, HEADERS[title], *lines, ""])

    section(HORIZON, [str(inst.n_days)])
    section(SHIFTS, [f"{Sh[a]},{inst.shift_len[a]},"
                     + "|".join(Sh[b] for b in np.flatnonzero(inst.forbidden[a]))
                     for a in range(inst.n_shifts)])
    section(STAFF, [",".join([E[i], "|".join(f"{Sh[s]}={inst.max_shifts[i, s]}" for s in range(inst.n_shifts)),
                              str(inst.max_minutes[i]), str(inst.min_minutes[i]), str(inst.max_consec[i]),
                              str(inst.min_consec[i]), str(inst.min_consec_off[i]), str(inst.max_weekends[i])])
                    for i in range(inst.n_emp)])
    section(DAYS_OFF, [",".join([E[i], *map(str, np.flatnonzero(inst.day_off[i]))])
                       for i in range(inst.n_emp) if inst.day_off[i].any()])
    for title, w in ((ON_REQ, inst.on_req), (OFF_REQ, inst.off_req)):
        section(title, [f"{E[i]},{d},{Sh[s]},{_num(w[i, d, s])}" for i, d, s in zip(*np.nonzero(w))])
    section(COVER, [f"{d},{Sh[s]},{inst.rho[d, s]},{_num(inst.w_under[d, s])},{_num(inst.w_over[d, s])}"
                    for d in range(inst.n_days) for s in range(inst.n_shifts)])
    return "\n".join(out)


def write_bc(inst: Instance, path) -> None:
    Path(path).write_text(format_bc(inst), encoding="utf-8")


# --- solutions: the XML roster format (.roster) ---------------------------------

SOLUTION_NAME = re.compile(r"^(?P<instance>.+)\.Solution\.(?P<value>\d+(?:\.\d+)?)\.roster$")


def solution_name(path) -> tuple[str, float] | None:
    """(instance name, objective value) for a file named ``InstanceN.Solution.VALUE.roster``,
    None for any other name."""
    m = SOLUTION_NAME.match(Path(path).name)
    return (m["instance"], float(m["value"])) if m else None


def read_roster(path, inst: Instance) -> np.ndarray:
    """The roster array of a solution file, for the instance it solves."""
    path = Path(path)
    return parse_roster(path.read_bytes(), inst, name=path.name)


def _roster_root(text, name: str) -> ET.Element:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise FormatError(f"{name}: not well-formed XML ({e})") from None
    if root.tag != "Roster":
        raise FormatError(f"{name}: root element is <{root.tag}>, expected <Roster>")
    return root


def parse_roster(text, inst: Instance, name: str = "roster") -> np.ndarray:
    """(n, m) int8 roster array from the text (str or bytes) of a ``.roster`` file.

    Refuses unknown employee or shift IDs, days outside the horizon, an employee
    listed twice and two shifts for one employee on one day.
    """
    root = _roster_root(text, name)
    eidx = {e: i for i, e in enumerate(inst.employee_ids)}
    sidx = {s: k for k, s in enumerate(inst.shift_ids)}
    roster = empty_roster(inst.n_emp, inst.n_days)
    seen = set()
    for emp in root.findall("Employee"):  # direct children; <Violations> nests other <Employee>s
        eid = (emp.get("ID") or "").strip()
        if eid not in eidx:
            raise FormatError(f"{name}: unknown employee id {eid!r}")
        if eid in seen:
            raise FormatError(f"{name}: employee {eid!r} listed twice")
        seen.add(eid)
        i = eidx[eid]
        for a in emp.findall("Assign"):
            day_text, shift_id = (a.findtext("Day") or "").strip(), (a.findtext("Shift") or "").strip()
            try:
                d = int(day_text)
            except ValueError:
                raise FormatError(f"{name}: employee {eid!r}: day must be an integer, got {day_text!r}") from None
            if not 0 <= d < inst.n_days:
                raise FormatError(f"{name}: employee {eid!r}: day {d} outside 0..{inst.n_days - 1}")
            if shift_id not in sidx:
                raise FormatError(f"{name}: employee {eid!r}, day {d}: unknown shift id {shift_id!r}")
            if roster[i, d] != OFF:
                raise FormatError(f"{name}: employee {eid!r} has two shifts on day {d}")
            roster[i, d] = sidx[shift_id]
    return roster


def roster_penalty(text, name: str = "roster") -> float | None:
    """The ``<Penalty>`` element of a ``.roster`` file (str or bytes), None if absent."""
    v = _roster_root(text, name).findtext("Penalty")
    if v is None or not v.strip():
        return None
    try:
        return float(v)
    except ValueError:
        raise FormatError(f"{name}: <Penalty> must be a number, got {v.strip()!r}") from None


def format_roster(roster: np.ndarray, inst: Instance, penalty=None) -> str:
    """A ``.roster`` file for the roster; ``parse_roster(format_roster(r, inst), inst)`` gives r back."""
    roster = np.asarray(roster)
    if roster.shape != (inst.n_emp, inst.n_days):
        raise ValueError(f"roster shape {roster.shape}, instance needs {(inst.n_emp, inst.n_days)}")
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<Roster xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
           'xsi:noNamespaceSchemaLocation="Roster.xsd">',
           f" <SchedulingPeriodFile>{escape(inst.name)}.ros</SchedulingPeriodFile>"]
    if penalty is not None:
        out.append(f" <Penalty>{_num(penalty)}</Penalty>")
    for i, eid in enumerate(inst.employee_ids):
        days = np.flatnonzero(roster[i] != OFF)
        out.append(f" <Employee ID={quoteattr(eid)}>" + "".join(
            f"<Assign><Day>{d}</Day><Shift>{escape(inst.shift_ids[roster[i, d]])}</Shift></Assign>" for d in days)
            + "</Employee>")
    out.append("</Roster>")
    return "\n".join(out) + "\n"


def write_roster(roster: np.ndarray, inst: Instance, path, penalty=None) -> None:
    Path(path).write_text(format_roster(roster, inst, penalty=penalty), encoding="utf-8")
