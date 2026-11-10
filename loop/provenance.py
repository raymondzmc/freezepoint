"""Provenance: what produced a run, beyond its config and seed. `loop.run` writes `compute(cfg)` into
`<run dir>/provenance.json` when it creates the run directory, before the arm starts; this module is the one place it
is computed.

It records the git commit and whether the work tree was dirty; a SHA-256 over the prompt templates
(`loop/prompts/*.md`); one over the Python source of `harness/`, `loop/` and `analysis/`; one per data lock file
(`data/*.lock.json`) and for `uv.lock`; the Python, Qlib and Anthropic SDK versions; and the split's windows as dates.
`COMPARED` names the fields every run of one cell must agree on (`analysis.curves.load_cell` refuses a cell whose runs
differ in one); the git commit and the dirty flag are recorded but not compared, so an unrelated commit between seeds
does not break a cell. Outside a git work tree both are None."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from harness import splits

ROOT = Path(__file__).resolve().parents[1]
FILE = "provenance.json"
SOURCE_DIRS = ("harness", "loop", "analysis")
COMPARED = ("prompts_sha256", "source_sha256", "lock_files", "split")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_sha(root: Path, files) -> str:
    """One hash over the files' paths (relative to `root`) and contents, in path order."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        h.update(f.relative_to(root).as_posix().encode() + b"\0" + _sha(f.read_bytes()).encode() + b"\n")
    return h.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30, check=False)
    except OSError:
        return None
    return p.stdout if p.returncode == 0 else None


def _window(w) -> list[str] | None:
    return [w.start, w.end] if w else None


def compute(cfg, root: Path = ROOT) -> dict:
    head, status = _git(root, "rev-parse", "HEAD"), _git(root, "status", "--porcelain")
    sources = [p for d in SOURCE_DIRS for p in (root / d).rglob("*.py") if "__pycache__" not in p.parts]
    locks = [*sorted((root / "data").glob("*.lock.json")), root / "uv.lock"]
    s = splits.get(cfg.split)
    return {
        "git_commit": head.strip() if head is not None else None,
        "git_dirty": bool(status.strip()) if status is not None and head is not None else None,
        "prompts_sha256": _tree_sha(root, (root / "loop" / "prompts").glob("*.md")),
        "source_sha256": _tree_sha(root, sources),
        "lock_files": {f.relative_to(root).as_posix(): _sha(f.read_bytes()) for f in locks if f.is_file()},
        "versions": {"python": sys.version.split()[0], "pyqlib": importlib.metadata.version("pyqlib"),
                     "anthropic": importlib.metadata.version("anthropic")},
        "split": {"name": s.name, "feedback": _window(s.feedback), "pilot": _window(s.pilot),
                  "heldout": _window(s.heldout), "forward": _window(s.forward), "embargo_days": s.embargo_days},
    }


def write(cfg, run_dir) -> Path:
    path = Path(run_dir) / FILE
    path.write_text(json.dumps(compute(cfg), indent=2) + "\n")
    return path
