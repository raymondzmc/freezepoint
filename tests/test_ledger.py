"""Ledger: views hide the right windows, invalid trials still count, ids are per-candidate, appends are atomic."""
from __future__ import annotations
import json, multiprocessing as mp
from harness.ledger.ledger import DiagnosticView, Ledger, LoopView, Trial

def test_loop_view_hides_pilot_heldout_and_transfer(tmp_path):
    led = Ledger(tmp_path / "l.jsonl"); tid = led.append(Trial(arm="refine", run="r", step=0, artifact="$close"))
    for w in ("feedback", "pilot", "heldout", "transfer:csi500"): led.score(tid, w, {"ic": 0.1})
    assert set(LoopView(led).trials()[tid]["scores"]) == {"feedback"}
    assert set(DiagnosticView(led).trials()[tid]["scores"]) == {"feedback", "pilot"}
    assert set(led.trials()[tid]["scores"]) == {"feedback", "pilot", "heldout", "transfer:csi500"}

def test_hide_accepts_exact_names_and_prefix_patterns(tmp_path):
    led = Ledger(tmp_path / "l.jsonl"); tid = led.append(Trial(arm="refine", run="r", step=0, artifact="$close"))
    for w in ("feedback", "pilot", "transfer:csi500", "transfer:sp500"): led.score(tid, w, {"ic": 0.1})
    assert set(led.trials(hide=("pilot",))[tid]["scores"]) == {"feedback", "transfer:csi500", "transfer:sp500"}
    assert set(led.trials(hide=("transfer:*",))[tid]["scores"]) == {"feedback", "pilot"}
    assert set(led.trials(hide=("transfer:csi500",))[tid]["scores"]) == {"feedback", "pilot", "transfer:sp500"}

def test_views_do_not_mutate_ledger_state(tmp_path):
    led = Ledger(tmp_path / "l.jsonl"); tid = led.append(Trial(arm="refine", run="r", step=0, artifact="$close"))
    led.score(tid, "pilot", {"ic": 0.1})
    LoopView(led).trials(); assert "pilot" in led.trials()[tid]["scores"]

def test_invalid_trial_counts(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    tid = led.append(Trial(arm="refine", run="r", step=1, artifact="Ref($close,-1)", valid=False, error="negative Ref"))
    assert led.count("r") == 1
    assert led.trials()[tid]["valid"] is False and led.trials()[tid]["error"] == "negative Ref"

def test_same_artifact_different_step_gets_two_records(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    a = led.append(Trial(arm="refine", run="r", step=0, artifact="$close"))
    b = led.append(Trial(arm="refine", run="r", step=1, artifact="$close", parent=a))
    assert a != b
    t = led.trials()
    assert set(t) == {a, b} and t[a]["artifact_sha"] == t[b]["artifact_sha"] and t[b]["parent"] == a
    assert led.count("r") == 2
    led.score(b, "feedback", {"ic": 0.2})
    assert led.trials()[a]["scores"] == {} and set(led.trials()[b]["scores"]) == {"feedback"}

def _worker(args):
    path, w = args
    led = Ledger(path)
    for i in range(200):
        tid = led.append(Trial(arm="refine", run=f"w{w}", step=i, artifact=f"$close*{w}+{i}"))
        if i % 20 == 0: led.score(tid, "feedback", {"ic": 0.1, "daily": [0.001 * j for j in range(1000)]})
        if i % 25 == 0: led.records()    # concurrent reads must never see a torn line

def test_parallel_appends_do_not_interleave(tmp_path):
    path = tmp_path / "l.jsonl"
    with mp.get_context("spawn").Pool(4) as pool: pool.map(_worker, [(path, w) for w in range(4)])
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    recs = [json.loads(l) for l in lines]    # every line parses
    assert len(recs) == 800 + 4 * 10
    led = Ledger(path)
    assert led.count() == 800 and len(led.trials()) == 800
    assert sum(1 for t in led.trials().values() if "feedback" in t["scores"]) == 40
