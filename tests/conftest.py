"""The suite's guard and fixtures.

Guard: a test marked `api` (calls a paid API) is skipped unless FREEZEPOINT_RUN_API=1, and one marked `gpu` (loads a
local model) unless FREEZEPOINT_RUN_GPU=1, whatever `-m` selects; `addopts` in pyproject.toml also deselects both by
default, but any `-m` on the command line replaces that.

Fixtures for the data-backed tool tests: each builds a registry whose call log lives under tmp_path and skips its test
when the data it needs is not on disk."""
import os
import pathlib

import pytest

from harness.pit.tools import data_present, make_registry

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "fnspid" / "index"
GUARDS = {"api": "FREEZEPOINT_RUN_API", "gpu": "FREEZEPOINT_RUN_GPU"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, var in GUARDS.items():
            if item.get_closest_marker(marker) and os.environ.get(var) != "1":
                item.add_marker(pytest.mark.skip(reason=f"marked {marker}: runs only with {var}=1 in the environment"))


def _registry(region, tmp_path, present, where, **kw):
    if not present: pytest.skip(f"no data at {where}")
    return make_registry(region, tmp_path / "calls.jsonl", **kw)

@pytest.fixture
def cn_registry(tmp_path): return _registry("cn", tmp_path, data_present("cn"), "the cn Qlib data")

@pytest.fixture
def us_registry(tmp_path): return _registry("us", tmp_path, data_present("us"), "the us Qlib data")

@pytest.fixture
def news_registry(tmp_path): return _registry("cn", tmp_path, NEWS.exists(), NEWS, news_root=NEWS)
