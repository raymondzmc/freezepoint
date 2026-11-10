"""The three arms of the controlled comparison: `frozen` (one proposal), `refine` (a proposal, then `max_steps`
refinements of the best so far) and `bestofn` (`n_candidates` independent proposals).

Guarantees. (1) Every candidate the model returns is appended to the ledger before it is scored; an invalid one is
appended with `valid=False` and the validator's reason and never scored. (2) Every decision (which candidate is the
best so far, what feedback and history the generator is handed, which candidate is chosen) is made from the ledger as
`LoopView` shows it, so the loop can never read the pilot, held-out, forward or transfer scores; the pilot score is
written after each candidate and never read here. (3) Each arm spends exactly its budget of generations (1,
`max_steps + 1`, `n_candidates`), invalid and failed candidates included. A generator exception is not a candidate: it
propagates, nothing is logged for it, and no `chosen.json` is written. A ledger that already holds a trial of the run
is refused before anything is generated, so two loops can never share one run's trials. (4) Only a valid candidate whose feedback score
is `ok` can be chosen; ties keep the earlier candidate (keep-if-strictly-better on `ic`).

Not acted on in this plan: `verifier.window_days` (the whole feedback window is scored), `budget.tokens_per_arm`
(the budget is counted in generations) and `control_generator` (no arm uses a control generator yet); all three are
carried in the config and so in the run id. A stopping rule other
than `none`, or a verifier metric other than `ic`, raises."""
from __future__ import annotations

import json

from harness import splits
from harness.backtest import evaluate, score_record
from harness.factors import validate
from harness.ledger.ledger import Ledger, LoopView, Trial
from loop.config import RunConfig, region_of, run_dir, run_id
from loop.config import validate as validate_config
from loop.generator import FEEDBACK_KEYS, Generator


def _feedback(rec: dict) -> dict:
    return rec["scores"].get("feedback") or {}

def eligible(rec: dict) -> bool:
    """A valid candidate whose feedback score is ok: the only kind that can be the best so far or be chosen."""
    return bool(rec["valid"] and _feedback(rec).get("ok"))

def best_so_far(recs: list[dict]) -> dict | None:
    """The first candidate with the highest feedback ic among the eligible ones (greedy keep-if-strictly-better)."""
    best = None
    for r in recs:
        if eligible(r) and (best is None or _feedback(r)["ic"] > _feedback(best)["ic"]): best = r
    return best

def _history(recs: list[dict]) -> list[dict]:
    """One entry per earlier candidate: its ic when it scored, else why it did not (validator or backtest)."""
    return [{"expr": r["artifact"], "ic": _feedback(r)["ic"]} if eligible(r) else
            {"expr": r["artifact"], "error": r["error"] if not r["valid"] else _feedback(r).get("reason", "")}
            for r in recs]


class _Run:
    """One arm for one seed: generates, logs, scores, and reads back only through LoopView."""
    def __init__(self, cfg, seed, gen, ledger, evaluate, context, tool_log_span):
        self.cfg, self.seed, self.gen, self.ledger, self.evaluate = cfg, seed, gen, ledger, evaluate
        self.context, self.tool_log_span = context, tool_log_span
        self.split, self.region, self.run = splits.get(cfg.split), region_of(cfg.universe), run_id(cfg, seed)
        self.view, self.ids = LoopView(ledger), []

    def seen(self) -> list[dict]:
        """This run's candidates in generation order, as the loop is allowed to see them."""
        trials = self.view.trials()
        return [trials[i] for i in self.ids]

    def propose(self, step: int, seed: int) -> None:
        (expr,) = self.gen.propose(self.context, 1, seed)
        self._candidate(expr, step, None)

    def refine(self, step: int, best: dict, recs: list[dict]) -> None:
        feedback = {k: _feedback(best)[k] for k in FEEDBACK_KEYS}
        expr = self.gen.refine(best["artifact"], feedback, _history(recs), self.seed, context=self.context)
        self._candidate(expr, step, best["id"])

    def _candidate(self, expr: str, step: int, parent: str | None) -> None:
        cost = dict(self.gen.last_cost or {})
        ok, why = validate(expr, self.region)
        tid = self.ledger.append(Trial(arm=self.cfg.arm, run=self.run, step=step, artifact=expr, parent=parent,
                                       model=cost.get("model") or self.cfg.generator.model,
                                       tool_log_span=self.tool_log_span, cost=cost, valid=ok, error=why))
        self.ids.append(tid)
        if not ok: return
        for name in ("feedback", "pilot"):     # pilot is written for the diagnostic and never read back here
            fs = self.evaluate(expr, getattr(self.split, name), self.cfg.universe, self.region)
            self.ledger.score(tid, name, score_record(fs))


def run_arm(cfg: RunConfig, seed: int, gen: Generator, ledger: Ledger, evaluate=evaluate, *, context: str = "",
            root="runs", tool_log_span: tuple[int, int] | None = None) -> str | None:
    """Run `cfg.arm` for `seed`; write `chosen.json` in the run directory and return the chosen trial id, or None
    when no candidate was valid and scored. `cfg` is validated first, so a config built in memory cannot skip the
    equal-compute check."""
    validate_config(cfg)
    if cfg.stopping.rule != "none":
        raise ValueError(f"stopping.rule {cfg.stopping.rule!r} is not implemented in this plan; use 'none'")
    if cfg.verifier.metric != "ic":
        raise ValueError(f"verifier.metric {cfg.verifier.metric!r} is not implemented in this plan; use 'ic'")
    r = _Run(cfg, seed, gen, ledger, evaluate, context, tool_log_span)
    if any(t["run"] == r.run for t in r.view.trials().values()):
        raise RuntimeError(f"{ledger.path} already holds trials of run {r.run}; a run's ledger is appended to by one "
                           "arm only, so nothing was generated")
    if cfg.arm == "frozen":
        r.propose(0, seed)
    elif cfg.arm == "refine":
        for step in range(cfg.budget.max_steps + 1):
            recs = r.seen()
            best = best_so_far(recs)
            if best is None: r.propose(step, seed)      # nothing to refine yet: a fresh proposal, same budget
            else: r.refine(step, best, recs)
    elif cfg.arm == "bestofn":
        for i in range(cfg.budget.n_candidates):
            r.propose(i, seed + i)
    else:
        raise ValueError(f"unknown arm {cfg.arm!r}")
    chosen = best_so_far(r.seen())
    pick = {k: chosen[k] if chosen else None for k in ("id", "artifact", "step")}
    out = {"id": pick["id"], "artifact": pick["artifact"], "arm": cfg.arm, "run": r.run, "seed": seed,
           "step": pick["step"]}
    (run_dir(cfg, seed, root) / "chosen.json").write_text(json.dumps(out, indent=2) + "\n")
    return out["id"]
