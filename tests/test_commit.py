"""commit/commit.py: `from-runs` builds the commitment from finished run directories and `verify` fails loudly on a
mismatch. Data-free; `ots` is never reachable (subprocess PATH without it, or `shutil.which` patched)."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from loop import config as lcfg
from loop import provenance

PROVENANCE = provenance.compute(lcfg.load(Path(__file__).resolve().parent.parent / "configs" / "cn_refine.yaml"))

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


def make_run(root, seed, *, arm="refine", artifact="Mean($close, 5)/$close", done=True, chosen=True, nothing=False,
             fake=False, prov=True):
    cfg = lcfg.load(ROOT / "configs" / f"cn_{arm}.yaml")
    if fake:
        g = dataclasses.replace(cfg.generator, provider="fake", model="fake", revision="fake", exprs=("$close",))
        cfg = dataclasses.replace(cfg, generator=g)
    cfg = dataclasses.replace(cfg, seeds=(seed,))
    d = lcfg.run_dir(cfg, seed, root=root)
    if prov:
        (d / "provenance.json").write_text(json.dumps({**PROVENANCE, "git_commit": f"commit-{seed}"}) + "\n")
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
        assert body["provenance"][d.name] == json.loads((d / "provenance.json").read_text())
    assert set(body) == {"artifacts", "configs", "provenance", "made"}
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


def test_fake_generator_run_is_refused_and_nothing_written(tmp_path):
    root = tmp_path / "runs"
    make_run(root, 0)
    fake = make_run(root, 1, fake=True)
    out = tmp_path / "out"
    r = cli("from-runs", root / "*", out)
    assert r.returncode != 0 and fake.name in r.stderr and "fake" in r.stderr and "never committed" in r.stderr
    assert not out.exists() or not any(out.iterdir())
    ok = cli("from-runs", make_run(root, 2), tmp_path / "out2")    # real runs still work
    assert ok.returncode == 0 and (tmp_path / "out2" / "commitment.json").exists()


def test_run_without_provenance_is_refused_and_nothing_written(tmp_path):
    root = tmp_path / "runs"
    make_run(root, 0)
    bad = make_run(root, 1, prov=False)
    out = tmp_path / "out"
    r = cli("from-runs", root / "*", out)
    assert r.returncode != 0 and bad.name in r.stderr and "provenance.json" in r.stderr
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


@pytest.mark.parametrize("which", ["chosen.json", "config.yaml"])
def test_output_directory_that_is_a_run_directory_is_refused(tmp_path, runs, which):
    """The output argument forgotten: the last run directory would become the output."""
    _, dirs = runs
    target = dirs[-1]
    if which == "config.yaml": (target / "chosen.json").unlink()
    before = _snapshot(target)
    r = cli("from-runs", *dirs[:-1], target)
    assert r.returncode != 0 and "run directory" in r.stderr and str(target) in r.stderr
    assert _snapshot(target) == before


def test_literal_path_that_matches_nothing_is_an_error(tmp_path, runs):
    root, dirs = runs
    out = tmp_path / "out"
    for bad in (root / "no-such-run", root / "report"):
        if bad.name == "report": (bad / "notes").mkdir(parents=True)
        r = cli("from-runs", dirs[0], bad, out)
        assert r.returncode != 0 and str(bad) in r.stderr and "not a run directory" in r.stderr
    assert not out.exists()


def test_two_directories_claiming_one_run_id_are_refused(tmp_path, runs):
    _, dirs = runs
    copy = tmp_path / "elsewhere" / dirs[0].name
    shutil.copytree(dirs[0], copy)
    out = tmp_path / "out"
    r = cli("from-runs", *dirs, copy, out)
    assert r.returncode != 0 and dirs[0].name in r.stderr and str(copy) in r.stderr and "same run id" in r.stderr
    assert not out.exists()


def test_verify_says_the_timestamp_was_not_checked_without_ots(tmp_path, runs, monkeypatch, capsys):
    root, _ = runs
    mod = _load_module()
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["commit.py", "from-runs", str(root / "*"), str(out)])
    mod.main()
    (out / "commitment.sha256.ots").write_bytes(b"receipt")
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: pytest.fail("ots must not be run"))
    capsys.readouterr()
    mod.verify(out)
    text = capsys.readouterr().out
    assert "hash matches: True" in text and "timestamp not checked" in text and "uv tool install" in text


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


def _snapshot(d): return {p.name: p.read_bytes() for p in d.iterdir()}


def test_from_runs_refuses_existing_commitment(tmp_path, runs):
    root, _ = runs
    out = tmp_path / "out"
    assert cli("from-runs", root / "*", out).returncode == 0
    (out / "commitment.sha256.ots").write_bytes(b"receipt")
    before = _snapshot(out)
    r = cli("from-runs", root / "*", out)
    assert r.returncode != 0 and "commitment.json" in r.stderr
    assert _snapshot(out) == before


def test_from_runs_refuses_stray_receipt(tmp_path, runs):
    root, _ = runs
    out = tmp_path / "out"
    out.mkdir()
    (out / "commitment.sha256.ots").write_bytes(b"receipt")
    r = cli("from-runs", root / "*", out)
    assert r.returncode != 0 and "commitment.sha256.ots" in r.stderr
    assert _snapshot(out) == {"commitment.sha256.ots": b"receipt"}


def test_make_refuses_existing_commitment(tmp_path):
    arts, conf, out = tmp_path / "a.json", tmp_path / "c.yaml", tmp_path / "out"
    arts.write_text("[]"); conf.write_text("a: 1\n")
    assert cli("make", arts, conf, out).returncode == 0
    before = _snapshot(out)
    r = cli("make", arts, conf, out)
    assert r.returncode != 0 and "commitment.json" in r.stderr
    assert _snapshot(out) == before


@pytest.mark.parametrize(("rc", "expected"), [(0, 0), (1, 1)])
def test_verify_follows_ots_result(tmp_path, runs, monkeypatch, rc, expected):
    root, _ = runs
    mod = _load_module()
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["commit.py", "from-runs", str(root / "*"), str(out)])
    mod.main()
    (out / "commitment.sha256.ots").write_bytes(b"receipt")
    calls = []
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/fake/ots")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: calls.append(a[0]) or subprocess.CompletedProcess(a[0], rc))
    with pytest.raises(SystemExit) as e:
        mod.verify(out)
        sys.exit(0)
    assert e.value.code == expected
    assert calls == [["ots", "verify", str(out / "commitment.sha256.ots")]]
