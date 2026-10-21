# Data

Instance files are not stored in the repository. Put them in `data/raw/`
(ignored by git) and pass their paths to the scripts.

## Burke & Curtois instances (Instance1–Instance24)

Source: https://www.schedulingbenchmarks.org/nrp/ (text format, `InstanceN.txt`).
From the external drive, or downloaded:

```bash
curl -O https://www.schedulingbenchmarks.org/nrp/data/instances1_24.zip
unzip instances1_24.zip -d data/raw
```

Check that every file parses and has the published size, and run the smoke test
on it:

```bash
python scripts/smoke.py data/raw/Instance*.txt
```

Reference: E.K. Burke and T. Curtois, New approaches to nurse rostering
benchmark instances, European Journal of Operational Research 237(1) (2014)
71–81.

## Not yet supported

Veenaas & Buchwitz NRRP dataset, NSPLib, INRC-II. None of the families above
supplies published rosters x° or disruptions ξ; both are generated
(`nrrp/generator.py`).
