"""The post-run scorer: the scores the loop was never allowed to see, written after the arm has finished.

For every valid trial of one run it writes `heldout` on the home universe and, for each universe `u` in
`cfg.transfer_to`, `transfer:<u>:feedback`, `transfer:<u>:pilot` and `transfer:<u>:heldout`. Guarantees. (1) It only
fills in what is missing: a window a trial already has a score for (a failed one included) is never scored again, so
a second call on a fully scored run writes nothing and an interrupted one can be resumed. (2) Scores are new ledger
records, in the same shape the arms write (`score_record`); nothing already in the ledger changes, and trials of other
runs and invalid trials get nothing. (3) Universes are the outer loop and trials the inner one, so Qlib is
re-initialised for a new region at most once per universe. (4) It reads the full ledger, since the loop is over; what
`LoopView` hides is decided by the window names, which this module only adds to."""
from __future__ import annotations

from harness import splits
from harness.backtest import evaluate, score_record
from harness.ledger.ledger import Ledger
from loop.config import RunConfig, region_of

TRANSFER_WINDOWS = ("feedback", "pilot", "heldout")


def transfer_window(universe: str, window: str) -> str:
    """The ledger's name for `window` scored on transfer universe `universe`: `transfer:<universe>:<window>`. The one
    place it is built; `LoopView` hides every name with the `transfer:` prefix."""
    return f"transfer:{universe}:{window}"


def _plan(cfg: RunConfig) -> list[tuple[str, list[tuple[str, str]]]]:
    """(universe, [(window, score name), ...]) in scoring order: the home universe first, then `transfer_to`."""
    plan = [(cfg.universe, [("heldout", "heldout")])]
    plan += [(u, [(w, transfer_window(u, w)) for w in TRANSFER_WINDOWS]) for u in cfg.transfer_to]
    return plan


def score_run(cfg: RunConfig, run_id: str, ledger: Ledger, evaluate=evaluate) -> int:
    """Write the missing held-out and transfer scores for every valid trial of `run_id`; return the number written."""
    split = splits.get(cfg.split)
    trials = [t for t in ledger.trials().values() if t["run"] == run_id and t["valid"]]
    written = 0
    for universe, windows in _plan(cfg):
        region = region_of(universe)
        for t in trials:
            for window, name in windows:
                if name in t["scores"]: continue
                fs = evaluate(t["artifact"], getattr(split, window), universe, region)
                ledger.score(t["id"], name, score_record(fs))
                t["scores"][name] = True        # mark it taken, in case a universe appears twice
                written += 1
    return written
