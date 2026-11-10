"""Fixtures for the data-backed tool tests: each builds a registry whose call log lives under tmp_path and
skips its test when the data it needs is not on disk."""
import pathlib

import pytest

from harness.pit.tools import make_registry

ROOT = pathlib.Path(__file__).resolve().parents[1]
QLIB = pathlib.Path.home() / ".qlib" / "qlib_data"
NEWS = ROOT / "data" / "fnspid" / "index"

def _registry(region, tmp_path, data_dir, **kw):
    if not data_dir.exists(): pytest.skip(f"no data at {data_dir}")
    return make_registry(region, tmp_path / "calls.jsonl", **kw)

@pytest.fixture
def cn_registry(tmp_path): return _registry("cn", tmp_path, QLIB / "cn_data")

@pytest.fixture
def us_registry(tmp_path): return _registry("us", tmp_path, QLIB / "us_data")

@pytest.fixture
def news_registry(tmp_path): return _registry("cn", tmp_path, NEWS, news_root=NEWS)
