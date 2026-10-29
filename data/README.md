# Data

Benchmark files are not stored in the repository. Put them in `data/raw/`
(ignored by git) and pass their paths to the scripts:

```
data/raw/Instance1.txt ... Instance24.txt                 instances, text format
data/raw/solutions/InstanceN.Solution.VALUE.roster        published solutions (50 files)
```

## Burke & Curtois instances (Instance1–Instance24)

Source: https://www.schedulingbenchmarks.org/nrp/ (text format, `InstanceN.txt`).
From the external drive, or downloaded:

```bash
curl -O https://www.schedulingbenchmarks.org/nrp/data/instances1_24.zip
unzip -j instances1_24.zip 'instances1_24/*.txt' -d data/raw
```

Check that every file parses and has the published size, and run the smoke test
on it:

```bash
python scripts/smoke.py data/raw/Instance*.txt
```

Reference: E.K. Burke and T. Curtois, New approaches to nurse rostering
benchmark instances, European Journal of Operational Research 237(1) (2014)
71–81.

## Burke & Curtois solutions

The same page offers the solutions as `instances1_24solutions.zip` (also on the
external drive). Its folder `Solutions/XML/` holds 50 files
`InstanceN.Solution.VALUE.roster`, at least two for most instances, where VALUE
is the solution's objective value as computed by the authors of the benchmark.
Copy them to `data/raw/solutions/`:

```bash
mkdir -p data/raw/solutions
unzip -j instances1_24solutions.zip '*/Solutions/XML/*.roster' -d data/raw/solutions
python scripts/check_solutions.py data/raw/solutions/*.roster
```

The check reads every solution for its instance and passes iff our J equals
VALUE (and the file's `<Penalty>`), every hard constraint of the published
profile holds, and, for the 33 files that list their violations, the request
and cover terms match the listing separately. On 29/10/2026 all 50 passed.

These solutions are the published rosters x° of the re-rostering contexts on
this family (Remark 5): a context is an instance, one of its solution files
as x°, and a disruption. They are not all the current best known values: the
table in `nrrp/bc_format.py` (`PUBLISHED`) has the best known values, which are
lower for some instances (e.g. Instance8: best known 1300, files 1308, 1314,
1315). The zip also holds `.ros` files, the instances in XML; we read the
`.txt` files instead.

## Not yet supported

Veenaas & Buchwitz NRRP dataset (HICSS 2026: INRC-II-based format, 14- and
28-day rosters with disruption scenarios), NSPLib, INRC-II. Disruptions ξ are
generated for every family (`nrrp/generator.py`, provisional).
