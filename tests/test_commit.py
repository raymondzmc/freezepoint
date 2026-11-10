"""commit/commit.py: `from-runs` builds the commitment from finished run directories and `verify` fails loudly on a
mismatch. Data-free; `ots` is never reachable (subprocess PATH without it, or `shutil.which` patched)."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from loop import config as lcfg

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "commit" / "commit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("commit_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cli(*args, cwd=ROOT):
    """Run the script as a subprocess whose PATH holds only the interpreter's directory, so `ots` cannot be found."""
    env = {"PATH": str(Path(sys.executable).parent), "HOME": os.environ.get("HOME", "")}
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          check=False, capture_output=True, text=True, cwd=cwd, env=env)


def make_run(root, seed, *, arm="refine", artifact="Mean($close, 5)/$close", done=True, chosen=True, nothing=False):
    cfg = lcfg.load(ROOT / "configs" / f"cn_{arm}.yaml")
    cfg = dataclasses.replace(cfg, seeds=(seed,))
    d = lcfg.run_dir(cfg, seed, root=root)
    if chosen:
        out = {"id": None if nothing else f"t{seed}", "artifact": None if nothing else artifact, "arm": arm,
               "run": d.name, "seed": seed, "step": None if nothing else 1}
        (d / "chosen.json").write_text(json.dumps(out, indent=2) + "\n")
    (d / "ledger.jsonl").write_text("")
    if done:
        (d / "done").write_text("2026-11-09T00:00:00+00:00\n")
    return d


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    return root, [make_run(root, s, artifact=f"Std($close, {s + 2})") for s in (3, 1, 2)]


def test_from_runs_collects_chosen_artifacts(tmp_path, runs):
    root, dirs = runs
    out = tmp_path / "out"
    r = cli("from-runs", root / "*", out)
    assert r.returncode == 0, r.stderr
    body = json.loads((out / "commitment.json").read_text())
    assert [a["run"] for a in body["artifacts"]] == sorted(d.name for d in dirs)
    assert {a["artifact"] for a in body["artifacts"]} == {"Std($close, 3)", "Std($close, 4)", "Std($close, 5)"}
    assert set(body["configs"]) == {d.name for d in dirs}
    for d in dirs:
        assert body["configs"][d.name] == yaml.safe_load((d / "config.yaml").read_text())
    assert body["made"]
    assert r.stdout.splitlines()[0].startswith("commitment ")
    assert len(r.stdout.splitlines()) == 2


def test_verify_detects_tampering(tmp_path, runs):
    root, _ = runs
    out = tmp_path / "out"
    assert cli("from-runs", root / "*", out).returncode == 0
    ok = cli("verify", out)
    assert ok.returncode == 0 and "hash matches: True" in ok.stdout
    p = out / "commitment.json"
    b = bytearray(p.read_bytes())
    i = b.index(b"Std")
    b[i] ^= 0x01
    p.write_bytes(bytes(b))
    bad = cli("verify", out)
    assert bad.returncode == 1
    assert "False" in bad.stdout


def test_run_without_chosen_artifact_is_included(tmp_path):
    root = tmp_path / "runs"
    d = make_run(root, 0, nothing=True)
    out = tmp_path / "out"
    assert cli("from-runs", d, out).returncode == 0
    (a,) = json.loads((out / "commitment.json").read_text())["artifacts"]
    assert a["id"] is None and a["artifact"] is None


def test_unfinished_run_is_refused_and_nothing_written(tmp_path):
    root = tmp_path / "runs"
    make_run(root, 0)
    bad = make_run(root, 1, done=False)
    out = tmp_path / "out"
    r = cli("from-runs", root / "*", out)
    assert r.returncode != 0 and bad.name in (r.stderr + r.stdout)
    assert not out.exists() or not any(out.iterdir())


def test_aborted_and_non_run_directories_are_ignored(tmp_path):
    root = tmp_path / "runs"
    good = make_run(root, 0)
    aborted = make_run(root, 1, done=False)
    aborted = aborted.rename(aborted.with_name(aborted.name + ".aborted-20261109T000000Z"))
    (root / "report").mkdir()
    (root / "report" / "notes.txt").write_text("x")
    out = tmp_path / "out"
    r = cli("from-runs", root / "*", out)
    assert r.returncode == 0, r.stderr
    body = json.loads((out / "commitment.json").read_text())
    assert [a["run"] for a in body["artifacts"]] == [good.name]
    assert list(body["configs"]) == [good.name]


def test_no_run_matched_is_an_error(tmp_path):
    (tmp_path / "runs" / "report").mkdir(parents=True)
    out = tmp_path / "out"
    for pattern in (tmp_path / "runs" / "*", tmp_path / "nothing-*"):
        r = cli("from-runs", pattern, out)
        assert r.returncode != 0 and "no run" in (r.stderr + r.stdout).lower()
    assert not out.exists()


def test_glob_and_expanded_dirs_agree(tmp_path, runs):
    root, dirs = runs
    a, b = tmp_path / "a", tmp_path / "b"
    assert cli("from-runs", root / "*", a).returncode == 0
    assert cli("from-runs", *dirs, b).returncode == 0
    ja, jb = (json.loads((x / "commitment.json").read_text()) for x in (a, b))
    assert ja["artifacts"] == jb["artifacts"] and ja["configs"] == jb["configs"]


def test_from_runs_never_anchors(tmp_path, runs, monkeypatch):
    root, _ = runs
    mod = _load_module()
    def boom(*a, **k): raise AssertionError("subprocess.run called")
    monkeypatch.setattr(mod.subprocess, "run", boom)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "argv", ["commit.py", "from-runs", str(root / "*"), str(tmp_path / "out")])
    mod.main()
    assert not list((tmp_path / "out").glob("*.ots"))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["commitment.json", "commitment.sha256"]


def test_verify_skips_ots_without_receipt(tmp_path, runs, monkeypatch):
    root, _ = runs
    mod = _load_module()
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/fake/ots")
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: calls.append(a))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["commit.py", "from-runs", str(root / "*"), str(out)])
    mod.main()
    mod.verify(out)
    assert calls == []
