"""Tests for `loop.provenance`: what a run records about the code, prompts, data pins and versions that produced it."""
import importlib.metadata
import json
import pathlib
import shutil
import subprocess
import sys

from harness import splits
from loop import config, provenance

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = config.load(ROOT / "configs" / "cn_refine.yaml")


def test_contents():
    p = provenance.compute(CFG)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert p["git_commit"] == head.strip() and isinstance(p["git_dirty"], bool)
    assert set(p) == {"git_commit", "git_dirty", "prompts_sha256", "source_sha256", "lock_files", "versions", "split"}
    assert len(p["prompts_sha256"]) == 64 and len(p["source_sha256"]) == 64
    assert "uv.lock" in p["lock_files"] and "data/qlib_cn.lock.json" in p["lock_files"]
    assert all(k == "uv.lock" or (k.startswith("data/") and k.endswith(".lock.json")) for k in p["lock_files"])
    assert p["versions"] == {"python": sys.version.split()[0], "pyqlib": importlib.metadata.version("pyqlib"),
                             "anthropic": importlib.metadata.version("anthropic")}
    s = splits.get(CFG.split)
    assert p["split"] == {"name": s.name, "feedback": [s.feedback.start, s.feedback.end],
                          "pilot": [s.pilot.start, s.pilot.end], "heldout": [s.heldout.start, s.heldout.end],
                          "forward": None, "embargo_days": s.embargo_days}
    assert set(provenance.COMPARED) == {"prompts_sha256", "source_sha256", "lock_files", "split"}


def test_write_puts_it_in_the_run_directory(tmp_path):
    path = provenance.write(CFG, tmp_path)
    assert path == tmp_path / "provenance.json"
    got = json.loads(path.read_text())
    assert got["source_sha256"] == provenance.compute(CFG)["source_sha256"]


def _copy(tmp_path):
    root = tmp_path / "repo"
    for d in ("harness", "loop", "analysis"):
        shutil.copytree(ROOT / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
    (root / "data").mkdir()
    for f in [*ROOT.glob("data/*.lock.json"), ROOT / "uv.lock"]:
        shutil.copy(f, root / f.relative_to(ROOT))
    return root


def test_each_hash_follows_its_own_files(tmp_path):
    root = _copy(tmp_path)
    base = provenance.compute(CFG, root)
    assert {k: base[k] for k in provenance.COMPARED} == {k: provenance.compute(CFG)[k] for k in provenance.COMPARED}
    (root / "loop" / "prompts" / "refine.md").write_text("changed")
    a = provenance.compute(CFG, root)
    assert a["prompts_sha256"] != base["prompts_sha256"] and a["source_sha256"] == base["source_sha256"]
    (root / "analysis" / "new.py").write_text("x = 1\n")
    b = provenance.compute(CFG, root)
    assert b["source_sha256"] != a["source_sha256"]
    (root / "data" / "qlib_cn.lock.json").write_text("{}")
    c = provenance.compute(CFG, root)
    assert c["lock_files"]["data/qlib_cn.lock.json"] != base["lock_files"]["data/qlib_cn.lock.json"]
    assert c["git_commit"] is None and c["git_dirty"] is None          # not a git work tree: recorded as unknown
