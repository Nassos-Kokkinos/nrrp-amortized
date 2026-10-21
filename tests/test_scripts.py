import subprocess
import sys
from pathlib import Path

from nrrp.bc_format import published_mismatch
from nrrp.generator import synthetic_instance

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "data" / "mini_bc.txt"


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_smoke_script_on_a_file():
    out = run("scripts/smoke.py", str(MINI), "--episodes", "2")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ALL CHECKS PASS" in out.stdout and "FAIL" not in out.stdout


def test_smoke_script_reports_a_bad_file(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text(MINI.read_text().replace("0,E,1,100,1", "0,X,1,100,1"))
    out = run("scripts/smoke.py", str(bad), "--episodes", "1")
    assert out.returncode == 1
    assert "unknown shift id 'X'" in out.stdout


def test_bench_script():
    out = run("scripts/bench_env.py", str(MINI), "--seconds", "0.2")
    assert out.returncode == 0, out.stderr
    assert "mini_bc" in out.stdout


def test_published_size_check():
    assert published_mismatch(synthetic_instance(8, 14, 1, name="Instance1")) is None
    assert "published (14, 14, 2)" in published_mismatch(synthetic_instance(8, 14, 1, name="Instance2"))
    assert published_mismatch(synthetic_instance(8, 14, 1, name="anything")) is None
