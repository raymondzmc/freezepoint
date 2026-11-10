"""Tests for the command line. The data-free tests call `main` with the arm, the scorer and the proposal context
stubbed; the `data` test runs `python -m loop.run` end to end with the fake generator."""
import fcntl
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import replace

import pytest
import yaml

from harness.ledger.ledger import Ledger, Trial
from harness.splits import get as get_split
from loop import config, provenance
from loop import run as cli
from loop.generator import DEFAULT_FAKE_EXPRS

ROOT = pathlib.Path(__file__).resolve().parents[1]
QLIB = pathlib.Path.home() / ".qlib" / "qlib_data"
REAL_MISSING_DATA = cli.missing_data      # the `stubs` fixture replaces it
FILES = ("config.yaml", "provenance.json", "ledger.jsonl", "calls.jsonl", "chosen.json", "done")


def write_config(tmp_path, **changes):
    d = yaml.safe_load((ROOT / "configs" / "cn_refine.yaml").read_text())
    d["budget"].update(max_steps=1, n_candidates=2)
    for k, v in changes.items(): d[k] = v
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def fake_cfg(path):
    cfg = config.load(path)
    return replace(cfg, generator=replace(cfg.generator, provider="fake", model="fake", revision="fake",
                                          exprs=DEFAULT_FAKE_EXPRS))


@pytest.fixture
def stubs(monkeypatch):
    """Stub the three data-backed steps; record what ran. The arm writes chosen.json like the real one."""
    log = []
    def arm(cfg, seed, gen, ledger, *, context, root, tool_log_span):
        log.append(("arm", cfg, seed, context, tool_log_span))
        ledger.append(Trial(arm=cfg.arm, run=config.run_id(cfg, seed), step=0, artifact="$close"))
        (config.run_dir(cfg, seed, root) / "chosen.json").write_text("{}\n")
    def score(cfg, rid, ledger):
        log.append(("score", rid)); return 0
    def context(cfg, d):
        log.append(("context", d)); (d / "calls.jsonl").write_text("{}\n"); return "a context", (0, 1)
    monkeypatch.setattr(cli, "run_arm", arm)
    monkeypatch.setattr(cli, "score_run", score)
    monkeypatch.setattr(cli, "proposal_context", context)
    monkeypatch.setattr(cli, "missing_data", lambda cfg: [])
    return log


def main(*argv):
    return cli.main([str(a) for a in argv])


def hold_lock(runs, rid):
    """Take the run's lock the way a live invocation holds it; the caller closes the file to release it."""
    runs.mkdir(parents=True, exist_ok=True)
    f = open(runs / f"{rid}.lock", "a")     # noqa: SIM115 - held across the test
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return f


def lock_is_free(runs, rid) -> bool:
    with open(runs / f"{rid}.lock", "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def snapshot(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("state", ["fresh", "unfinished", "arm finished", "score-only"])
def test_a_second_invocation_on_a_held_lock_exits_non_zero_and_changes_nothing(tmp_path, stubs, capsys, state):
    cfg = write_config(tmp_path)
    runs = tmp_path / "runs"
    rid = config.run_id(fake_cfg(cfg), 0)
    d = runs / rid
    if state != "fresh":
        d.mkdir(parents=True)
        (d / "ledger.jsonl").write_text('{"live": "another invocation is writing here"}\n')
    if state in ("arm finished", "score-only"):
        (d / "chosen.json").write_text("{}\n")
    held = hold_lock(runs, rid)
    try:
        before = snapshot(runs)
        extra = ["--score-only"] if state == "score-only" else []
        assert main(cfg, "--seed", 0, "--provider", "fake", *extra, "--runs-dir", runs) != 0
        out, err = capsys.readouterr()
        assert out == "" and "another invocation" in err and rid in err
        assert stubs == [] and snapshot(runs) == before
        assert sorted(p.name for p in runs.iterdir()) == sorted([f"{rid}.lock", *([rid] if state != "fresh" else [])])
    finally:
        held.close()


def test_the_lock_is_released_when_main_returns_and_the_lock_file_stays(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    runs = tmp_path / "runs"
    assert main(cfg, "--seed", 0, "--provider", "fake", "--runs-dir", runs) == 0
    rid = capsys.readouterr().out.strip()
    assert (runs / f"{rid}.lock").is_file() and lock_is_free(runs, rid)


def test_the_lock_is_released_when_the_process_ends(tmp_path):
    """A process holding the lock is killed outright (no cleanup of its own); the lock is free afterwards."""
    runs, rid = tmp_path / "runs", "refine-cn-2018-2023-s0-deadbeef"
    holder = tmp_path / "holder.py"
    holder.write_text("import sys, time\nfrom pathlib import Path\nfrom loop.run import acquire_lock\n"
                      "if __name__ == '__main__':\n"
                      "    f = acquire_lock(Path(sys.argv[1]), sys.argv[2])\n"
                      "    print('held' if f else 'busy', flush=True)\n"
                      "    time.sleep(120)\n")
    p = subprocess.Popen([sys.executable, str(holder), str(runs), rid], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONPATH": str(ROOT)})
    try:
        assert p.stdout.readline().strip() == "held"
        assert not lock_is_free(runs, rid)
        assert cli.acquire_lock(runs, rid) is None
    finally:
        p.kill(); p.wait(timeout=20)
    assert lock_is_free(runs, rid)


@pytest.mark.parametrize("missing", ["cn", "us"])
def test_preflight_refuses_to_start_without_every_regions_data(tmp_path, stubs, capsys, monkeypatch, missing):
    """The real check against a data root that holds only the other region: nothing is generated or created."""
    from harness.pit import tools
    root = tmp_path / "qlib"
    for region, sub in (("cn", "cn_data"), ("us", "us_data")):
        if region != missing:
            (root / sub / "calendars").mkdir(parents=True); (root / sub / "calendars" / "day.txt").write_text("")
    monkeypatch.setattr(tools, "QLIB_DATA", root)
    monkeypatch.setattr(cli, "missing_data", REAL_MISSING_DATA)
    cfg = write_config(tmp_path)          # csi300 at home, csi500 and sp500 transfer: needs both regions
    runs = tmp_path / "runs"
    assert main(cfg, "--seed", 0, "--provider", "fake", "--runs-dir", runs) != 0
    out, err = capsys.readouterr()
    assert out == "" and f"no Qlib data for region '{missing}'" in err and stubs == []
    assert [p.name for p in runs.iterdir()] == [f"{config.run_id(fake_cfg(cfg), 0)}.lock"]


def test_preflight_names_each_missing_region_once(tmp_path, monkeypatch):
    from harness.pit import tools
    monkeypatch.setattr(tools, "QLIB_DATA", tmp_path / "nothing")
    assert REAL_MISSING_DATA(fake_cfg(write_config(tmp_path))) == ["cn", "us"]


def test_seed_not_in_config_seeds_fails(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    assert main(cfg, "--seed", 7, "--provider", "fake", "--runs-dir", tmp_path / "runs") != 0
    out, err = capsys.readouterr()
    assert out == "" and "seed 7" in err and "[0, 1, 2, 3, 4]" in err
    assert not (tmp_path / "runs").exists() and stubs == []


def test_score_only_without_a_run_fails(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    assert main(cfg, "--seed", 0, "--provider", "fake", "--score-only", "--runs-dir", tmp_path / "runs") != 0
    rid = config.run_id(fake_cfg(cfg), 0)
    assert "no run directory" in capsys.readouterr().err
    assert [p.name for p in (tmp_path / "runs").iterdir()] == [f"{rid}.lock"]     # only the lock it took
    (tmp_path / "runs" / rid).mkdir(parents=True)
    assert main(cfg, "--seed", 0, "--provider", "fake", "--score-only", "--runs-dir", tmp_path / "runs") != 0
    assert "chosen.json" in capsys.readouterr().err and stubs == []
    assert list((tmp_path / "runs" / rid).iterdir()) == []


def test_normal_run_writes_effective_config_and_prints_only_the_run_id(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    assert main(cfg, "--seed", 0, "--provider", "fake", "--runs-dir", tmp_path / "runs") == 0
    eff = fake_cfg(cfg)
    rid = config.run_id(eff, 0)
    assert capsys.readouterr().out == rid + "\n"
    d = tmp_path / "runs" / rid
    assert config.load(d / "config.yaml") == eff
    assert [s[0] for s in stubs] == ["context", "arm", "score"]
    assert stubs[1][3] == "a context" and stubs[1][4] == (0, 1) and stubs[2] == ("score", rid)
    assert (d / "done").exists() and sorted(p.name for p in d.iterdir()) == sorted(FILES)
    written, now = json.loads((d / "provenance.json").read_text()), provenance.compute(eff)
    assert {k: written[k] for k in provenance.COMPARED} == {k: now[k] for k in provenance.COMPARED}
    assert written["git_commit"] == now["git_commit"]


def test_overrides_change_the_run_id(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    ids = []
    for extra in ([], ["--model", "other-model"]):
        assert main(cfg, "--seed", 0, "--provider", "fake", *extra, "--runs-dir", tmp_path / "runs") == 0
        ids.append(capsys.readouterr().out.strip())
    base = fake_cfg(cfg)
    other = replace(base, generator=replace(base.generator, model="other-model", revision="other-model"))
    assert ids == [config.run_id(base, 0), config.run_id(other, 0)] and ids[0] != ids[1]
    assert config.run_id(config.load(cfg), 0) not in ids
    assert config.load(tmp_path / "runs" / ids[1] / "config.yaml") == other


def test_fake_provider_keeps_the_configs_own_exprs(tmp_path, stubs, capsys):
    raw = yaml.safe_load(write_config(tmp_path).read_text())
    raw["generator"].update(provider="fake", exprs=["$close", "$open"])
    p = tmp_path / "own.yaml"; p.write_text(yaml.safe_dump(raw))
    assert main(p, "--seed", 0, "--provider", "fake", "--runs-dir", tmp_path / "runs") == 0
    rid = capsys.readouterr().out.strip()
    assert config.load(tmp_path / "runs" / rid / "config.yaml").generator.exprs == ("$close", "$open")


def test_override_is_validated(tmp_path, stubs, capsys):
    raw = yaml.safe_load(write_config(tmp_path).read_text())
    raw["generator"].update(provider="local", temperature=0.7)
    p = tmp_path / "local.yaml"; p.write_text(yaml.safe_dump(raw))
    assert main(p, "--seed", 0, "--provider", "api", "--runs-dir", tmp_path / "runs") != 0
    assert "temperature" in capsys.readouterr().err and not (tmp_path / "runs").exists()


def test_already_done_run_is_left_alone(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    rid = config.run_id(fake_cfg(cfg), 0)
    d = tmp_path / "runs" / rid
    d.mkdir(parents=True)
    for name in FILES: (d / name).write_text(f"{name}\n")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()}
    assert main(cfg, "--seed", 0, "--provider", "fake", "--runs-dir", tmp_path / "runs") == 0
    assert capsys.readouterr().out == rid + "\n" and stubs == []
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()} == before
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [rid]


def test_unfinished_run_is_moved_aside_and_started_fresh(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    rid = config.run_id(fake_cfg(cfg), 0)
    d = tmp_path / "runs" / rid
    d.mkdir(parents=True)
    (d / "ledger.jsonl").write_text('{"half": "a dead run"}\n')
    assert main(cfg, "--seed", 0, "--provider", "fake", "--runs-dir", tmp_path / "runs") == 0
    assert capsys.readouterr().out == rid + "\n"
    (aborted,) = [p for p in (tmp_path / "runs").iterdir() if p.name not in (rid, f"{rid}.lock")]
    assert aborted.name.startswith(f"{rid}.aborted-") and aborted.name.endswith("Z")
    assert (aborted / "ledger.jsonl").read_text() == '{"half": "a dead run"}\n' and not (aborted / "done").exists()
    assert "a dead run" not in (d / "ledger.jsonl").read_text() and (d / "done").exists()
    assert [s[0] for s in stubs] == ["context", "arm", "score"]


def test_plain_run_on_a_finished_arm_scores_what_is_missing(tmp_path, stubs, capsys):
    """The arm finished (`chosen.json`) and only scoring died: a plain invocation scores it, it does not generate again
    and does not move the directory aside."""
    cfg = write_config(tmp_path)
    rid = config.run_id(fake_cfg(cfg), 0)
    d = tmp_path / "runs" / rid
    d.mkdir(parents=True)
    (d / "chosen.json").write_text("{}\n"); (d / "ledger.jsonl").write_text('{"kept": "a finished arm"}\n')
    assert main(cfg, "--seed", 0, "--provider", "fake", "--runs-dir", tmp_path / "runs") == 0
    assert capsys.readouterr().out == rid + "\n"
    assert stubs == [("score", rid)] and (d / "done").exists()
    assert (d / "ledger.jsonl").read_text() == '{"kept": "a finished arm"}\n'
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [rid, f"{rid}.lock"]


def test_score_only_runs_the_scorer_and_writes_done(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    rid = config.run_id(fake_cfg(cfg), 0)
    d = tmp_path / "runs" / rid
    d.mkdir(parents=True)
    (d / "chosen.json").write_text("{}\n"); (d / "ledger.jsonl").write_text("")
    assert main(cfg, "--seed", 0, "--provider", "fake", "--score-only", "--runs-dir", tmp_path / "runs") == 0
    assert capsys.readouterr().out == rid + "\n"
    assert stubs == [("score", rid)] and (d / "done").exists()
    assert sorted(p.name for p in d.iterdir()) == ["chosen.json", "done", "ledger.jsonl"]   # no config rewrite


def test_score_only_leaves_a_done_run_alone(tmp_path, stubs, capsys):
    cfg = write_config(tmp_path)
    rid = config.run_id(fake_cfg(cfg), 0)
    d = tmp_path / "runs" / rid
    d.mkdir(parents=True)
    for name in FILES: (d / name).write_text(f"{name}\n")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()}
    assert main(cfg, "--seed", 0, "--provider", "fake", "--score-only", "--runs-dir", tmp_path / "runs") == 0
    assert capsys.readouterr().out == rid + "\n" and stubs == []          # the scorer never ran
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()} == before


@pytest.mark.parametrize("error", [RuntimeError("API outage"), FileNotFoundError("no Qlib data for region 'us'")])
def test_exception_leaves_no_done(tmp_path, stubs, monkeypatch, capsys, error):
    cfg = write_config(tmp_path)
    def boom(*a, **kw): raise error
    monkeypatch.setattr(cli, "score_run", boom)
    with pytest.raises(type(error)):
        main(cfg, "--seed", 0, "--provider", "fake", "--runs-dir", tmp_path / "runs")
    d = tmp_path / "runs" / config.run_id(fake_cfg(cfg), 0)
    assert d.is_dir() and not (d / "done").exists() and capsys.readouterr().out == ""


def test_missing_data_in_score_only_leaves_no_done(tmp_path, stubs, monkeypatch, capsys):
    cfg = write_config(tmp_path)
    d = tmp_path / "runs" / config.run_id(fake_cfg(cfg), 0)
    d.mkdir(parents=True)
    (d / "chosen.json").write_text("{}\n"); (d / "ledger.jsonl").write_text("")
    def boom(*a, **kw): raise FileNotFoundError("no Qlib data for region 'us'")
    monkeypatch.setattr(cli, "score_run", boom)
    with pytest.raises(FileNotFoundError):
        main(cfg, "--seed", 0, "--provider", "fake", "--score-only", "--runs-dir", tmp_path / "runs")
    assert not (d / "done").exists() and capsys.readouterr().out == ""


def test_missing_data_exits_non_zero_from_the_command_line(tmp_path):
    """The real CLI and the real scorer with the data root pointed at an empty directory (by a small runner script that
    patches `tools.QLIB_DATA`, then calls `main`): non-zero exit, no `done`, nothing on stdout, no score written."""
    cfg = write_config(tmp_path)
    d = tmp_path / "runs" / config.run_id(fake_cfg(cfg), 0)
    d.mkdir(parents=True)
    (d / "chosen.json").write_text("{}\n")
    ledger = Ledger(d / "ledger.jsonl")
    ledger.append(Trial(arm="refine", run=d.name, step=0, artifact="$close"))
    shim = tmp_path / "shim.py"
    shim.write_text("import pathlib, sys\nfrom harness.pit import tools\n"
                    f"tools.QLIB_DATA = pathlib.Path({str(tmp_path / 'empty')!r})\n"
                    "from loop.run import main\nsys.exit(main())\n")
    p = subprocess.run([sys.executable, str(shim), str(cfg), "--seed", "0", "--provider", "fake", "--score-only",
                        "--runs-dir", str(tmp_path / "runs")], cwd=ROOT, capture_output=True, text=True, timeout=120,
                       check=False, env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONPATH": str(ROOT)})
    assert p.returncode != 0 and "FileNotFoundError" in p.stderr and p.stdout == ""
    assert not (d / "done").exists() and ledger.records()[-1].get("artifact") == "$close"


@pytest.mark.data
def test_cli_end_to_end_with_fake_generator(tmp_path):
    if not ((QLIB / "cn_data").exists() and (QLIB / "us_data").exists()): pytest.skip("no Qlib data")
    cfg_path = write_config(tmp_path, transfer_to=["csi500", "sp500"])
    runs = tmp_path / "runs"
    cmd = [sys.executable, "-m", "loop.run", str(cfg_path), "--seed", "0", "--provider", "fake",
           "--runs-dir", str(runs)]
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert p.returncode == 0, p.stderr[-4000:]
    rid = config.run_id(fake_cfg(cfg_path), 0)
    assert p.stdout.splitlines()[-1] == rid
    d = runs / rid
    assert all((d / f).is_file() for f in FILES)
    assert all((d / f).stat().st_mtime_ns <= (d / "done").stat().st_mtime_ns for f in FILES)

    ledger = Ledger(d / "ledger.jsonl")
    trials = [t for t in ledger.trials().values() if t["run"] == rid]
    assert len(trials) == 2 and [t["valid"] for t in trials] == [True, False]   # DEFAULT_FAKE_EXPRS[1] is invalid
    windows = {"feedback", "pilot", "heldout"} | {f"transfer:{u}:{w}" for u in ("csi500", "sp500")
                                                   for w in ("feedback", "pilot", "heldout")}
    for t in trials:
        assert set(t["scores"]) == (windows if t["valid"] else set())
        assert t["tool_log_span"] == [0, 1]

    (call,) = [json.loads(line) for line in (d / "calls.jsonl").read_text().splitlines()]
    assert call["tool"] == "universe" and call["as_of"] == get_split("cn-2018-2023").feedback.end
    assert call["args"] == {"market": "csi300"}

    snapshot = lambda: {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(runs.rglob("*")) if p.is_file()}
    before = snapshot()
    again = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert again.returncode == 0 and again.stdout.splitlines()[-1] == rid
    assert snapshot() == before
