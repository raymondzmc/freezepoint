"""Tests for `scripts/run_cell.sh`: a tiny cell end to end with the fake generator (needs data), and a failing run
(needs none)."""
import os
import pathlib
import subprocess
import time

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
QLIB = pathlib.Path.home() / ".qlib" / "qlib_data"
ARMS = ("frozen", "refine", "bestofn")


def tiny_configs(tmp_path):
    paths = []
    for arm in ARMS:
        d = yaml.safe_load((ROOT / "configs" / f"cn_{arm}.yaml").read_text())
        d["budget"].update(max_steps=1, n_candidates=2)
        d["transfer_to"] = ["csi500"]
        paths.append(tmp_path / f"cn_{arm}.yaml")
        paths[-1].write_text(yaml.safe_dump(d))
    return paths


def run_script(tmp_path, cfgs, **env):
    e = {**os.environ, "GEN": "fake", "RUNS": str(tmp_path / "runs"), "OUT": str(tmp_path / "out"), **env}
    return subprocess.run(["bash", "scripts/run_cell.sh", *map(str, cfgs)], cwd=ROOT, env=e, capture_output=True,
                          text=True, timeout=1500, check=False)


def test_a_failing_run_fails_the_script_and_writes_no_report(tmp_path):
    cfgs = tiny_configs(tmp_path)[:1]
    p = run_script(tmp_path, cfgs, SEEDS="9", JOBS="2")       # seed 9 is not one of the config's seeds
    assert p.returncode != 0
    assert not (tmp_path / "out" / "report.md").exists()
    assert "cn_frozen.yaml" in p.stderr and "seed 9" in p.stderr and str(tmp_path / "out" / "logs") in p.stderr
    (log,) = (tmp_path / "out" / "logs").glob("*.log")
    assert "seed 9 is not one of the config's seeds" in log.read_text()


@pytest.mark.data
def test_tiny_cell_end_to_end(tmp_path):
    if not ((QLIB / "cn_data").exists() and (QLIB / "us_data").exists()): pytest.skip("no Qlib data")
    t0 = time.monotonic()
    p = run_script(tmp_path, tiny_configs(tmp_path), SEEDS="0 1", JOBS="2")
    print(f"tiny cell wall-clock: {time.monotonic() - t0:.1f} s")
    assert p.returncode == 0, p.stderr[-4000:]
    done = [d for d in (tmp_path / "runs").iterdir() if (d / "done").is_file()]
    assert len(done) == 6 and len(list((tmp_path / "runs").iterdir())) == 6
    out = tmp_path / "out"
    assert all((out / f).is_file() for f in ("report.md", "curve.csv", "selection.csv", "cost.md"))
    assert "fake generator" in (out / "report.md").read_text()
    assert p.stdout.splitlines()[-1] == str(out / "report.md")
    assert "Wall-clock time of the cell:" in (out / "cost.md").read_text()
    assert len(list((out / "logs").glob("*.log"))) == 6
