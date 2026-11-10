"""Append-only JSONL ledger. One record per candidate (per run and step, so a repeated expression gets its
own record), with the arm, the step, the parent candidate's trial id, the data the agent saw (tool-log
span), and the scores by window. Trial.id = sha256(run|step|artifact)[:16]; every record also carries
artifact_sha = sha256(artifact)[:16], the pure content hash, so repeats of one expression can be grouped.
The loop may write and read `feedback`. The loop's runner writes `pilot` after each candidate and never
reads it: the loop's view (LoopView) hides it, along with `heldout`, `forward` and every `transfer:*` window.
`heldout` and `forward` are written only by the post-run scorer. DiagnosticView sees `pilot` but not
`heldout`, `forward` or `transfer:*`. Invalid candidates are logged too (valid=False, error=...) and count
toward the trial total. Appends and reads are serialised with flock, so parallel writers never interleave
and a reader never parses a half-written line, even for records larger than the file buffer."""
from __future__ import annotations
import fcntl, hashlib, json, pathlib, time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field

def _sha(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()[:16]

@dataclass
class Trial:
    arm: str                       # "frozen" | "refine" | "bestofn" | ...
    run: str                       # run id (seed, split, config hash)
    step: int                      # refinement step, 0 for the first generation
    artifact: str                  # the candidate itself (expression, code, text)
    parent: str | None = None      # trial id of the candidate it was refined from
    model: str = ""                # generator id and revision
    tool_log_span: tuple[int, int] | None = None   # line range in the tool call log
    scores: dict = field(default_factory=dict)      # window name -> metric dict
    cost: dict = field(default_factory=dict)        # tokens, seconds
    valid: bool = True             # False when the candidate was rejected before scoring (still counts as a trial)
    error: str = ""                # why it was rejected
    t: float = field(default_factory=time.time)
    @property
    def id(self): return _sha(f"{self.run}|{self.step}|{self.artifact}")
    @property
    def artifact_sha(self): return _sha(self.artifact)

def _hidden(window: str, hide) -> bool:
    """exact names hide one window; a name ending in `*` hides every window with that prefix ("transfer:*")"""
    return any(window.startswith(h[:-1]) if h.endswith("*") else window == h for h in hide)

class Ledger:
    SECRET = ("pilot", "heldout", "forward")
    def __init__(self, path):
        self.path = pathlib.Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    @contextmanager
    def _open(self, mode, lock):
        with open(self.path, mode) as f:
            fcntl.flock(f, lock)
            try: yield f
            finally: fcntl.flock(f, fcntl.LOCK_UN)
    def _write(self, rec):
        line = json.dumps(rec, default=str) + "\n"
        with self._open("a", fcntl.LOCK_EX) as f: f.write(line); f.flush()    # flush before the lock is released
    def append(self, trial: Trial) -> str:
        rec = asdict(trial); rec["id"] = trial.id; rec["artifact_sha"] = trial.artifact_sha
        self._write(rec); return trial.id
    def score(self, trial_id, window, metrics):
        """scores are added as new records (never edited), so the ledger stays append-only"""
        self._write({"id": trial_id, "score": {window: metrics}, "t": time.time()})
    def records(self):
        if not self.path.exists(): return []
        with self._open("r", fcntl.LOCK_SH) as f: return [json.loads(l) for l in f if l.strip()]
    def trials(self, hide=()):
        """trial id -> record with its scores. `hide` lists window names to leave out; `transfer:*` hides a prefix."""
        out = {}
        for r in self.records():
            if "score" in r:
                for w, m in r["score"].items():
                    if not _hidden(w, hide) and r["id"] in out: out[r["id"]]["scores"][w] = m
            else:
                r["scores"] = {w: m for w, m in r.get("scores", {}).items() if not _hidden(w, hide)}; out[r["id"]] = r
        return out
    def count(self, run=None):
        """the trial count the deflated Sharpe ratio needs: every candidate ever scored, not the ones kept"""
        return sum(1 for r in self.records() if "artifact" in r and (run is None or r["run"] == run))

class LoopView:
    """what the refinement loop is allowed to see: the ledger without pilot, held-out, forward or transfer scores"""
    HIDE = Ledger.SECRET + ("transfer:*",)
    def __init__(self, ledger: Ledger): self._l = ledger
    def append(self, trial): return self._l.append(trial)
    def trials(self): return self._l.trials(hide=self.HIDE)
    def count(self, run=None): return self._l.count(run)

class DiagnosticView(LoopView):
    """the diagnostic's view: pilot is visible, held-out, forward and transfer scores are not"""
    HIDE = ("heldout", "forward", "transfer:*")
