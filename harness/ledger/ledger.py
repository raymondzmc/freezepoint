"""Append-only JSONL ledger. One record per candidate, keyed by a content hash of the artifact, with
the arm, the step, the parent candidate, the data the agent saw (tool-log span), and the scores by
window. The loop may write `feedback` and read it back; `pilot` is written by the diagnostic;
`heldout` and `forward` are written only by the post-run scorer, and the loop's view of the ledger
(LoopView) never returns them."""
from __future__ import annotations
import hashlib, json, pathlib, time
from dataclasses import dataclass, asdict, field

@dataclass
class Trial:
    arm: str                       # "frozen" | "refine" | "bestofn" | ...
    run: str                       # run id (seed, split, config hash)
    step: int                      # refinement step, 0 for the first generation
    artifact: str                  # the candidate itself (expression, code, text)
    parent: str | None = None      # content hash of the candidate it was refined from
    model: str = ""                # generator id and revision
    tool_log_span: tuple[int, int] | None = None   # line range in the tool call log
    scores: dict = field(default_factory=dict)      # window name -> metric dict
    cost: dict = field(default_factory=dict)        # tokens, seconds
    t: float = field(default_factory=time.time)
    @property
    def id(self): return hashlib.sha256(self.artifact.encode()).hexdigest()[:16]

class Ledger:
    SECRET = ("heldout", "forward")
    def __init__(self, path):
        self.path = pathlib.Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def append(self, trial: Trial):
        rec = asdict(trial); rec["id"] = trial.id
        with open(self.path, "a") as f: f.write(json.dumps(rec, default=str) + "\n")
        return trial.id
    def score(self, trial_id, window, metrics):
        """scores are added as new records (never edited), so the ledger stays append-only"""
        with open(self.path, "a") as f: f.write(json.dumps({"id": trial_id, "score": {window: metrics}, "t": time.time()}) + "\n")
    def records(self):
        with open(self.path) as f: return [json.loads(l) for l in f if l.strip()]
    def trials(self, hide=()):
        out = {}
        for r in self.records():
            if "score" in r:
                for w, m in r["score"].items():
                    if w not in hide and r["id"] in out: out[r["id"]]["scores"][w] = m
            else:
                r["scores"] = {w: m for w, m in r.get("scores", {}).items() if w not in hide}; out[r["id"]] = r
        return out
    def count(self, run=None):
        """the trial count the deflated Sharpe ratio needs: every candidate ever scored, not the ones kept"""
        return sum(1 for r in self.records() if "artifact" in r and (run is None or r["run"] == run))

class LoopView:
    """what the refinement loop is allowed to see: the ledger without held-out or forward scores"""
    def __init__(self, ledger: Ledger): self._l = ledger
    def append(self, trial): return self._l.append(trial)
    def trials(self): return self._l.trials(hide=Ledger.SECRET)
    def count(self, run=None): return self._l.count(run)
