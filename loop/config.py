"""Run configuration. One YAML file describes one arm, and the run's identity is a hash of the config object
(not of the file's text), so comments and key order never change an id while any changed value, including one
applied in memory with `dataclasses.replace`, does; the one exception is `seeds`, which is recorded but not hashed
(the run's own seed is in its id). `load` rejects what it does not understand: an unknown key
in a hashed config must not pass silently."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from harness import splits

ARMS = ("frozen", "refine", "bestofn")
PROVIDERS = ("api", "local", "fake")
EFFORTS = ("low", "medium", "high", "xhigh", "max")   # the Messages API's `output_config.effort` values
UNIVERSE_REGION = {"csi300": "cn", "csi500": "cn", "csi800": "cn", "csi1000": "cn", "sp500": "us"}   # region per universe


def region_of(universe: str) -> str:
    if universe not in UNIVERSE_REGION:
        raise ValueError(f"unknown universe {universe!r}; known universes: {sorted(UNIVERSE_REGION)}")
    return UNIVERSE_REGION[universe]


@dataclass(frozen=True)
class Generator:
    provider: str
    model: str
    revision: str
    temperature: float | None
    max_output_tokens: int
    effort: str | None = None             # api only: `output_config.effort`; None sends nothing (the API's default)
    exprs: tuple[str, ...] | None = None  # the fixed expressions a `fake` provider returns


@dataclass(frozen=True)
class ControlGenerator:
    provider: str
    model: str


@dataclass(frozen=True)
class Budget:
    max_steps: int
    n_candidates: int
    tokens_per_arm: int


@dataclass(frozen=True)
class Verifier:
    metric: str
    window_days: int


@dataclass(frozen=True)
class Stopping:
    rule: str


@dataclass(frozen=True)
class RunConfig:
    arm: str
    split: str
    universe: str
    transfer_to: tuple[str, ...]
    seeds: tuple[int, ...]
    generator: Generator
    control_generator: ControlGenerator
    budget: Budget
    verifier: Verifier
    stopping: Stopping


_SECTIONS = {"generator": Generator, "control_generator": ControlGenerator, "budget": Budget,
             "verifier": Verifier, "stopping": Stopping}
_TUPLES = {"transfer_to", "seeds", "exprs"}


def _build(cls, d, where: str):
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(d).__name__}")  # noqa: TRY004
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; known keys: {sorted(fields)}")
    missing = sorted(n for n, f in fields.items() if f.default is dataclasses.MISSING and n not in d)
    if missing:
        raise ValueError(f"{where}: missing key(s) {missing}")
    kw = {}
    for name, v in d.items():
        if name in _SECTIONS and cls is RunConfig:
            v = _build(_SECTIONS[name], v, name)
        elif name in _TUPLES and v is not None:
            if not isinstance(v, list):
                raise ValueError(f"{where}.{name}: expected a list, got {v!r}")
            v = tuple(v)
        kw[name] = v
    return cls(**kw)


def validate(cfg: RunConfig) -> RunConfig:
    """Raise ValueError if `cfg` is not a config `load` would accept; return it unchanged otherwise. For configs
    changed in memory (`dataclasses.replace`), which bypass `load`."""
    if cfg.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {cfg.arm!r}")
    splits.get(cfg.split)
    for u in (cfg.universe, *cfg.transfer_to):
        region_of(u)
    if cfg.budget.n_candidates != cfg.budget.max_steps + 1:
        raise ValueError(f"budget.n_candidates ({cfg.budget.n_candidates}) must equal budget.max_steps + 1 "
                         f"({cfg.budget.max_steps + 1}) so that every arm gets equal compute")
    g = cfg.generator
    if g.provider not in PROVIDERS:
        raise ValueError(f"generator.provider must be one of {PROVIDERS}, got {g.provider!r}")
    if g.provider == "api" and g.temperature is not None:
        raise ValueError("the Messages API no longer accepts sampling parameters, so generator.temperature must be "
                         f"null for the api provider (got {g.temperature!r}); a config that says otherwise would "
                         "misdescribe the run")
    if cfg.arm == "bestofn" and g.provider == "local":
        raise ValueError("arm 'bestofn' cannot use generator.provider 'local': the local generator decodes greedily, so "
                         "its proposals for one prompt would be identical and best-of-N would log the same candidate "
                         f"{cfg.budget.n_candidates} times")
    if g.effort is not None and (not isinstance(g.effort, str) or g.effort not in EFFORTS):
        raise ValueError(f"generator.effort must be null (the API's default) or one of {EFFORTS}, got {g.effort!r}")
    if g.provider == "fake" and not g.exprs:
        raise ValueError("generator.provider 'fake' needs a non-empty generator.exprs")
    return cfg


def load(path) -> RunConfig:
    d = yaml.safe_load(Path(path).read_text())
    return validate(_build(RunConfig, d, "config"))


def to_dict(cfg: RunConfig) -> dict:
    def conv(v):
        if dataclasses.is_dataclass(v):
            return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v)}
        return [conv(x) for x in v] if isinstance(v, tuple) else v
    return conv(cfg)


def run_id(cfg: RunConfig, seed: int) -> str:
    """`<arm>-<split>-s<seed>-<hash>`: the hash is over the config without `seeds`, since the seed is already in the id
    and adding a seed to the list later must not change the ids of the runs already made."""
    d = to_dict(cfg)
    del d["seeds"]
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    sha = hashlib.sha256(canonical.encode()).hexdigest()
    return f"{cfg.arm}-{cfg.split}-s{seed}-{sha[:8]}"


def run_dir(cfg: RunConfig, seed: int, root="runs") -> Path:
    """The run's directory under `root`, created with the effective `config.yaml` in it; `cfg` is validated first."""
    validate(cfg)
    d = Path(root) / run_id(cfg, seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    return d
