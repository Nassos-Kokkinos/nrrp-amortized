#!/usr/bin/env python3
"""Check J against the published solutions of the Burke & Curtois instances.

  python scripts/check_solutions.py data/raw/solutions/*.roster
  python scripts/check_solutions.py SOLUTIONS... --instances data/raw

Each file InstanceN.Solution.VALUE.roster is read for the instance
InstanceN.txt in the instances folder (default: data/raw). Its roster is
evaluated under the evaluation profile of the family (BURKE_CURTOIS), as a
construction context, and the check passes iff J equals VALUE (and the file's
<Penalty>, when it has one) and every hard constraint holds. Files that list
their violations (<Violations>) are also checked term by term: the request
penalties must equal the requests term of J and the cover penalties its cover
term. Exit code 0 iff every file passes.

This validates the objective (20) and the hard constraints against the
benchmark's own evaluator, independently of any solver: the published values
were computed by the authors of the benchmark.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from nrrp.bc_format import FormatError, read_bc, read_roster, roster_penalty, solution_name
from nrrp.instance import Context
from nrrp.objective import J_terms, evaluate
from nrrp.profile import BURKE_CURTOIS

TOL = 1e-9


def natural(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", Path(path).name)]


def listed_terms(text) -> dict | None:
    """{'requests': ..., 'cover': ...} summed from the file's <Violations> listing, None if it has none."""
    v = ET.fromstring(text).find("Violations")
    if v is None:
        return None
    total = lambda q: sum(float(p.text) for p in v.findall(q))  # noqa: E731
    return {"requests": total("EmployeeViolations//Penalty"), "cover": total("CoverViolations//Penalty")}


def check(path, instances: Path, cache: dict) -> tuple[bool, str]:
    path = Path(path)
    parsed = solution_name(path)
    if parsed is None:
        return False, "name is not InstanceN.Solution.VALUE.roster"
    name, value = parsed
    if name not in cache:
        cache[name] = read_bc(instances / f"{name}.txt")
    inst = cache[name]
    text = path.read_bytes()
    penalty = roster_penalty(text, path.name)
    roster, ctx = read_roster(path, inst), Context(inst)
    ev = evaluate(roster, ctx, BURKE_CURTOIS)
    problems = []
    if abs(ev["J"] - value) > TOL:
        problems.append(f"J differs from the file name by {ev['J'] - value:+g}")
    if penalty is not None and abs(ev["J"] - penalty) > TOL:
        problems.append(f"J differs from <Penalty> by {ev['J'] - penalty:+g}")
    listed = listed_terms(text)
    if listed is not None:
        terms = J_terms(roster, ctx, BURKE_CURTOIS)
        for t in ("requests", "cover"):
            if abs(terms[t] - listed[t]) > TOL:
                problems.append(f"{t} term {terms[t]:g}, the file lists {listed[t]:g}")
    if not ev["feasible"]:
        problems.append(f"hard constraints violated: {ev['violations']}")
    pen = "-" if penalty is None else f"{penalty:g}"
    by_term = "-" if listed is None else "checked"
    line = (f"{path.name:36s} {inst.size:32s} {value:>9g} {pen:>9s} {ev['J']:>11g}  "
            f"{'yes' if ev['feasible'] else 'NO':>8s}  {by_term:>7s}")
    return not problems, line + ("" if not problems else "  <- " + "; ".join(problems))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="solution files InstanceN.Solution.VALUE.roster")
    ap.add_argument("--instances", type=Path, default=Path("data/raw"),
                    help="folder holding InstanceN.txt (default: data/raw)")
    args = ap.parse_args()

    print(f"      {'solution':36s} {'instance size':32s} {'published':>9s} {'Penalty':>9s} {'our J':>11s}  "
          f"{'feasible':>8s}  {'terms':>7s}")
    cache, n_ok, files = {}, 0, sorted(args.files, key=natural)
    for f in files:
        try:
            ok, line = check(f, args.instances, cache)
        except (OSError, FormatError) as e:
            ok, line = False, f"{Path(f).name:36s} cannot be read: {e}"
        print(("ok    " if ok else "FAIL  ") + line)
        n_ok += ok
    print(f"\n{n_ok}/{len(files)} solutions: J equals the published value and every hard constraint holds"
          f" (profile {BURKE_CURTOIS.name})")
    print("ALL CHECKS PASS" if n_ok == len(files) else "SOME CHECKS FAILED")
    sys.exit(0 if n_ok == len(files) else 1)


if __name__ == "__main__":
    main()
