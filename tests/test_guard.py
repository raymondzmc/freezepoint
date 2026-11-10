"""The conftest guard: `api` and `gpu` tests are skipped unless FREEZEPOINT_RUN_API=1 / FREEZEPOINT_RUN_GPU=1, whatever
`-m` says. Checked on two dummy tests in a scratch directory that uses this suite's conftest; nothing paid runs."""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUMMY = ("import pytest\n\n@pytest.mark.api\ndef test_paid():\n    pass\n\n"
         "@pytest.mark.gpu\ndef test_gpu():\n    pass\n\ndef test_plain():\n    pass\n")


def run(tmp_path, *args, **env):
    d = tmp_path / "suite"
    d.mkdir(exist_ok=True)
    shutil.copy(ROOT / "tests" / "conftest.py", d / "conftest.py")
    (d / "test_dummy.py").write_text(DUMMY)
    base = {k: v for k, v in os.environ.items() if not k.startswith("FREEZEPOINT_RUN_")}
    p = subprocess.run([sys.executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"), "--rootdir", str(ROOT),
                        "-p", "no:cacheprovider", "-q", "-rs", *args, str(d)], cwd=ROOT, capture_output=True, text=True,
                       timeout=120, check=False, env={**base, **env, "PYTHONPATH": str(ROOT)})
    return p.stdout


@pytest.mark.parametrize("args, env, summary", [
    (["-m", "api or gpu"], {}, "2 skipped, 1 deselected"),
    (["-m", "not data"], {}, "1 passed, 2 skipped"),
    (["-m", ""], {}, "1 passed, 2 skipped"),
    (["-m", "api or gpu"], {"FREEZEPOINT_RUN_API": "1"}, "1 passed, 1 skipped, 1 deselected"),
    (["-m", "api or gpu"], {"FREEZEPOINT_RUN_API": "1", "FREEZEPOINT_RUN_GPU": "1"}, "2 passed, 1 deselected"),
    (["-m", "api or gpu"], {"FREEZEPOINT_RUN_API": "yes"}, "2 skipped, 1 deselected"),
    ([], {"FREEZEPOINT_RUN_API": "1", "FREEZEPOINT_RUN_GPU": "1"}, "1 passed, 2 deselected"),   # addopts still applies
])
def test_api_and_gpu_tests_need_their_variable_whatever_the_marker_expression(tmp_path, args, env, summary):
    out = run(tmp_path, *args, **env)
    assert summary in out, out
    if "skipped" in summary:
        assert "FREEZEPOINT_RUN_" in out, out
