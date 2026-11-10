"""The command line: one arm for one seed, then the post-run scorer, leaving a finished run directory.

    python -m loop.run <config> --seed N [--score-only] [--provider P] [--model M] [--runs-dir DIR]

Guarantees. (1) The run id is computed from the config after the overrides, and that effective config is what the
run directory records. (2) One invocation per run at a time: before it looks at the run directory, every path
(normal, resume, `--score-only`) takes a non-blocking exclusive `flock` on `<runs-dir>/<run_id>.lock` and holds it for
the rest of the invocation (the OS releases it when the process ends, however it ends; the file is left in place). If
another process holds it, the command exits non-zero having changed nothing. Only the no-op on a run that already has
`done` skips the lock. Under the lock: a directory with `done` is left alone. One with `chosen.json` but no `done` (the
arm finished and only the scoring died) has its missing scores filled in and `done` written, with or without
`--score-only`. One with neither is an arm that died part-way (no live process can be writing to it, since none holds
the lock): it is renamed `<run_id>.aborted-<UTC timestamp>` (never deleted) and the run starts fresh, or, with
`--score-only`, the command refuses. So a run's ledger is never appended to by a second attempt, a finished arm is never
generated again, and `run_arm` itself refuses a ledger that already holds a trial of the run. (3) `done` is written
last and is the only sign that a run is complete; nothing is written to the directory after it. (4) The proposal
context comes from one point-in-time `universe` call as of the feedback window's end, logged in `calls.jsonl`; the
trials record that call's line range (0-based, end exclusive) as `tool_log_span`. (5) Standard output carries only
the run id, as its last line; everything else goes to standard error. Exit 0 on success, non-zero otherwise, and an
exception never leaves `done` behind."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import sys
from pathlib import Path

from harness import splits
from harness.ledger.ledger import Ledger
from harness.pit.tools import make_registry
from loop import config
from loop.arms import run_arm
from loop.generator import DEFAULT_FAKE_EXPRS, make_generator
from loop.score import score_run


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def effective_config(cfg: config.RunConfig, provider: str | None, model: str | None) -> config.RunConfig:
    """`cfg` with the command-line overrides applied, validated as `load` validates."""
    g = cfg.generator
    if provider == "fake":
        g = dataclasses.replace(g, provider="fake", model="fake", revision="fake", exprs=g.exprs or DEFAULT_FAKE_EXPRS)
    elif provider is not None:
        g = dataclasses.replace(g, provider=provider)
    if model is not None:
        g = dataclasses.replace(g, model=model, revision=model)
    return config.validate(dataclasses.replace(cfg, generator=g))


def _lines(path: Path) -> int:
    if not path.exists(): return 0
    with path.open() as f: return sum(1 for _ in f)


def proposal_context(cfg: config.RunConfig, run_dir: Path) -> tuple[str, tuple[int, int]]:
    """The setting the generator is told about, from the point-in-time tools as of the feedback window's end, and
    the line range those tool calls occupy in `calls.jsonl`."""
    log, fb = run_dir / "calls.jsonl", splits.get(cfg.split).feedback
    start = _lines(log)
    reg = make_registry(config.region_of(cfg.universe), log)
    names = reg.call("universe", as_of=fb.end, market=cfg.universe)
    text = (f"The universe is {cfg.universe}, which had {len(names)} member stocks on {fb.end}. The factor is "
            f"scored on daily data from {fb.start} to {fb.end}.")
    return text, (start, _lines(log))


def _write_done(d: Path) -> None:
    (d / "done").write_text(dt.datetime.now(dt.UTC).isoformat() + "\n")


def _run(cfg: config.RunConfig, seed: int, rid: str, runs: Path) -> None:
    d = runs / rid
    if d.exists():
        aborted = d.with_name(f"{rid}.aborted-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}")
        if aborted.exists(): raise FileExistsError(f"{aborted} already exists; not moving {d} onto it")
        d.rename(aborted)
        _say(f"{rid}: an earlier attempt did not finish; moved it to {aborted.name} and starting fresh")
    d = config.run_dir(cfg, seed, runs)
    gen = make_generator(cfg.generator)
    context, span = proposal_context(cfg, d)
    _say(f"{rid}: context: {context}")
    _say(f"{rid}: running the {cfg.arm} arm ({cfg.generator.provider} generator {cfg.generator.model})")
    ledger = Ledger(d / "ledger.jsonl")
    run_arm(cfg, seed, gen, ledger, context=context, root=runs, tool_log_span=span)
    _say(f"{rid}: scoring held-out and transfer windows")
    _say(f"{rid}: wrote {score_run(cfg, rid, ledger)} scores")
    _write_done(d)


def _score_only(cfg: config.RunConfig, rid: str, d: Path) -> None:
    _say(f"{rid}: scoring what is missing")
    _say(f"{rid}: wrote {score_run(cfg, rid, Ledger(d / 'ledger.jsonl'))} scores")
    _write_done(d)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m loop.run", description="Run one arm for one seed, then score it.")
    p.add_argument("config", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--score-only", action="store_true", help="only fill in missing post-run scores of a finished arm")
    p.add_argument("--provider", choices=config.PROVIDERS, help="override generator.provider")
    p.add_argument("--model", help="override generator.model and generator.revision")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return p


def acquire_lock(runs: Path, rid: str):
    """The open lock file `<runs>/<rid>.lock` with a non-blocking exclusive `flock` taken on it, or None when another
    process holds it. The lock lasts until the file is closed or the process ends, whichever comes first; the file
    itself is never removed."""
    runs.mkdir(parents=True, exist_ok=True)
    f = open(runs / f"{rid}.lock", "a")     # noqa: SIM115 - returned open: closing it releases the lock
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = effective_config(config.load(args.config), args.provider, args.model)
    except (OSError, ValueError) as e:
        _say(f"error: {args.config}: {e}")
        return 2
    if args.seed not in cfg.seeds:
        _say(f"error: seed {args.seed} is not one of the config's seeds {list(cfg.seeds)}")
        return 2
    rid = config.run_id(cfg, args.seed)
    if (args.runs_dir / rid / "done").exists():      # complete: nothing to do, and no lock needed to see it
        _say(f"{rid}: already complete, nothing to do")
        print(rid, flush=True)
        return 0
    lock = acquire_lock(args.runs_dir, rid)
    if lock is None:
        _say(f"error: another invocation of run {rid} is live (it holds {args.runs_dir / (rid + '.lock')}); "
             "nothing was changed. Wait for it to finish or stop it, then run this again.")
        return 3
    try:
        return _locked(args, cfg, rid)
    finally:
        lock.close()


def _locked(args, cfg: config.RunConfig, rid: str) -> int:
    """Everything that looks at or writes the run directory, under the run's lock."""
    d = args.runs_dir / rid
    if (d / "done").exists():
        _say(f"{rid}: already complete, nothing to do")
    elif args.score_only or (d / "chosen.json").is_file():
        if not d.is_dir():
            _say(f"error: --score-only: no run directory {d}; run without --score-only first")
            return 2
        if not (d / "chosen.json").is_file():
            _say(f"error: --score-only: {d} has no chosen.json, so its arm never finished; run without --score-only")
            return 2
        with contextlib.redirect_stdout(sys.stderr):
            _score_only(cfg, rid, d)
    else:
        with contextlib.redirect_stdout(sys.stderr):
            _run(cfg, args.seed, rid, args.runs_dir)
    print(rid, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
