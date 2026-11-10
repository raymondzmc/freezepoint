"""Run configuration: loading validates, identity comes from the config object, run directories round-trip."""
import dataclasses
from pathlib import Path

import pytest
import yaml

from harness import splits
from loop import config as C

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
ARMS = ["frozen", "refine", "bestofn"]


def shipped(arm="refine"):
    return C.load(CONFIGS / f"cn_{arm}.yaml")


def raw(arm="refine"):
    return yaml.safe_load((CONFIGS / f"cn_{arm}.yaml").read_text())


def write(tmp_path, d, name="c.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d))
    return p


def test_run_id_is_stable_for_same_config():
    a, b = shipped(), shipped()
    assert C.run_id(a, 0) == C.run_id(b, 0)
    assert C.run_id(a, 0) != C.run_id(a, 1)
    assert C.run_id(a, 0).startswith("refine-cn-2018-2023-s0-")
    assert len(C.run_id(a, 0).rsplit("-", 1)[1]) == 8


def test_bestofn_budget_equals_refine_budget(tmp_path):
    for arm in ARMS:
        cfg = shipped(arm)
        assert cfg.budget.n_candidates == cfg.budget.max_steps + 1
    d = raw("bestofn")
    d["budget"]["n_candidates"] = d["budget"]["max_steps"]
    with pytest.raises(ValueError, match="n_candidates"):
        C.load(write(tmp_path, d))


def test_three_configs_differ_only_in_arm():
    dicts = {a: C.to_dict(shipped(a)) for a in ARMS}
    assert {d["arm"] for d in dicts.values()} == set(ARMS)
    for d in dicts.values():
        d.pop("arm")
    assert dicts["frozen"] == dicts["refine"] == dicts["bestofn"]
    assert len({C.run_id(shipped(a), 0).rsplit("-", 1)[1] for a in ARMS}) == 3


@pytest.mark.parametrize("arm", ARMS)
def test_shipped_configs_load(arm):
    cfg = shipped(arm)
    assert cfg.arm == arm and cfg.generator.model == "claude-fable-5-2" and cfg.generator.temperature is None
    assert cfg.transfer_to == ("csi500", "sp500") and cfg.seeds == (0, 1, 2, 3, 4)
    assert cfg.generator.max_output_tokens == 16000 and cfg.generator.effort is None
    assert "effort: null" in (CONFIGS / f"cn_{arm}.yaml").read_text()


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max", None])
def test_effort_values_the_api_documents_load(tmp_path, effort):
    d = raw()
    d["generator"]["effort"] = effort
    assert C.load(write(tmp_path, d)).generator.effort == effort


def test_roundtrip_through_run_dir(tmp_path):
    cfg = shipped()
    d = C.run_dir(cfg, 3, root=tmp_path)
    assert d == tmp_path / C.run_id(cfg, 3) and d.is_dir()
    again = C.load(d / "config.yaml")
    assert again == cfg and C.run_id(again, 3) == C.run_id(cfg, 3)


def test_run_dir_default_root_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = shipped()
    d1 = C.run_dir(cfg, 0)
    d2 = C.run_dir(cfg, 0)
    assert d1 == d2 == Path("runs") / C.run_id(cfg, 0)
    assert (tmp_path / d1 / "config.yaml").exists()


def test_identity_ignores_comments_and_key_order_but_not_values(tmp_path):
    base = C.run_id(shipped(), 0)
    d = raw()
    reordered = dict(reversed(list(d.items())))
    reordered["generator"] = dict(reversed(list(d["generator"].items())))
    p = tmp_path / "r.yaml"
    p.write_text("# a new comment\n" + yaml.safe_dump(reordered, sort_keys=False) + "\n# trailing\n")
    assert C.run_id(C.load(p), 0) == base
    changes = [("universe", "csi500"), ("seeds", [0, 1]),
               ("generator.model", "other"), ("generator.max_output_tokens", 4096), ("budget.tokens_per_arm", 1),
               ("verifier.window_days", 500), ("stopping.rule", "fixed_k"), ("control_generator.model", "x"),
               ("transfer_to", ["csi500"])]
    for key, val in changes:
        e = raw()
        node, *rest = key.split(".")
        if rest:
            e[node][rest[0]] = val
        else:
            e[node] = val
        assert C.run_id(C.load(write(tmp_path, e)), 0) != base, key


def test_in_memory_change_changes_id():
    cfg = shipped()
    changed = dataclasses.replace(cfg, budget=dataclasses.replace(cfg.budget, tokens_per_arm=1))
    assert C.run_id(changed, 0) != C.run_id(cfg, 0)
    assert dataclasses.replace(cfg, split="x-split") != cfg
    assert C.run_id(dataclasses.replace(cfg, split="x-split"), 0) != C.run_id(cfg, 0)


def test_attribute_access_and_frozen():
    cfg = shipped()
    assert cfg.budget.max_steps == 20 and cfg.generator.exprs is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.arm = "frozen"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.budget.max_steps = 1


def _set(path, val):
    def f(d):
        node = d
        *parents, last = path.split(".")
        for p in parents:
            node = node[p]
        node[last] = val
    return f


def _fake(d):
    d["generator"].update(provider="fake", temperature=None)


BAD = {
    "unknown top key": _set("armm", 1),
    "unknown generator key": _set("generator.temprature", 0.1),
    "unknown budget key": _set("budget.extra", 1),
    "unknown verifier key": _set("verifier.extra", 1),
    "unknown stopping key": _set("stopping.extra", 1),
    "unknown control key": _set("control_generator.extra", 1),
    "bad arm": _set("arm", "best_of_n"),
    "bad split": _set("split", "nope"),
    "bad universe": _set("universe", "csi100"),
    "bad transfer": _set("transfer_to", ["csi500", "ftse"]),
    "candidates mismatch": _set("budget.n_candidates", 20),
    "bad provider": _set("generator.provider", "openai"),
    "api with temperature": _set("generator.temperature", 0.7),
    "unknown effort": _set("generator.effort", "minimal"),
    "effort not text": _set("generator.effort", 3),
    "fake without exprs": _fake,
    "fake with empty exprs": lambda d: (_fake(d), d["generator"].update(exprs=[])),
    "missing section": lambda d: d.pop("budget"),
    "missing key": lambda d: d["budget"].pop("max_steps"),
}


@pytest.mark.parametrize("name", list(BAD))
def test_invalid_config_raises(tmp_path, name):
    d = raw()
    BAD[name](d)
    with pytest.raises(ValueError):
        C.load(write(tmp_path, d))


def test_fake_provider_with_exprs_loads(tmp_path):
    d = raw()
    _fake(d)
    d["generator"]["exprs"] = ["Mean($close, 5)"]
    cfg = C.load(write(tmp_path, d))
    assert cfg.generator.exprs == ("Mean($close, 5)",)
    assert C.load(C.run_dir(cfg, 0, root=tmp_path) / "config.yaml") == cfg


def test_api_temperature_error_explains_why(tmp_path):
    d = raw()
    d["generator"]["temperature"] = 0.7
    with pytest.raises(ValueError, match="sampling"):
        C.load(write(tmp_path, d))


@pytest.mark.parametrize("universe,region", [("csi300", "cn"), ("csi500", "cn"), ("csi800", "cn"),
                                              ("csi1000", "cn"), ("sp500", "us")])
def test_region_of(universe, region):
    assert C.region_of(universe) == region


def test_region_of_unknown():
    with pytest.raises(ValueError, match="csi100"):
        C.region_of("csi100")


def test_splits_get():
    assert splits.get("cn-2018-2023") is splits.DEFAULT
    with pytest.raises(ValueError, match="cn-2018-2023"):
        splits.get("nope")
