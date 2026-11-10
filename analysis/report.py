"""The report of one cell: `curve.csv`, `selection.csv` and `report.md`.

    python -m analysis.report <run_dir> [<run_dir> ...] --out DIR

Guarantees. (1) It reads only one finished cell (`analysis.curves.load_cell`), so it cannot mix smoke runs with real
ones. (2) When the cell's generator is `fake`, the first line after the title of `report.md` says the runs are a
pipeline check, not a result. (3) The go/no-go statement (does the refine arm's held-out score at `max_steps` sit below
its peak by more than the 95% interval half-width) is made only when the refine arm has at least two seeds with a
held-out value at `max_steps`; otherwise the report says it cannot be answered and why. Standard output carries only
the path of `report.md`; errors go to standard error with a non-zero exit."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from analysis.curves import curve, load_cell, selection_stats

ARM_NAMES = {"frozen": "frozen", "refine": "refine", "bestofn": "best-of-N"}

ARMS_TEXT = (
    "The arms: frozen makes one proposal and keeps it; refine makes a proposal and then refines the best candidate so "
    "far at every step; best-of-N makes independent proposals, one per step, and keeps the best. In every arm the best "
    "candidate is the one with the highest rank IC on the feedback window, the earliest one on a tie.")
METRIC_TEXT = (
    "The metric is the rank IC (information coefficient): on each day, the Spearman rank correlation across the "
    "universe's stocks between the factor's values and the following period's returns, averaged over the days of a "
    "window. The feedback window is the one the loop sees and selects on; the held-out window comes later and the loop "
    "never sees its scores.")
CURVE_TEXT = (
    "Each row is a step k, the number of refinements spent so far, from 0 to {K}; at step k the refine and best-of-N "
    "arms have used k + 1 generations, and the frozen arm's single value is repeated on every row. Each cell is the "
    "mean over seeds of the held-out rank IC of the candidate that run holds after step k (the best on feedback rank IC "
    "so far), followed in brackets by the 95% confidence interval of that mean over seeds, from the t distribution. An "
    "interval reads n/a when fewer than two seeds have a value; when some seeds have no held-out value at a step "
    "(nothing valid yet, or the held-out score failed), the cell says how many of the arm's seeds it averages.")
SELECTION_TEXT = (
    "One row per run, computed on the feedback window, because that is the window the selection was made on. `run` is "
    "the run id and `seed` its seed. `n_trials` is the number of candidates the run generated, including the ones the "
    "validator rejected; `n_valid` is the number that were valid and scored on the feedback window. `dsr_of_chosen` is "
    "the deflated Sharpe ratio of the chosen candidate: the probability that its daily long-short Sharpe ratio on the "
    "feedback window beats the best Sharpe ratio expected from `n_trials` candidates with no skill (n/a when nothing "
    "was chosen). `pbo` is the probability of backtest overfitting, estimated by combinatorially symmetric "
    "cross-validation with 8 blocks over the valid candidates' daily long-short returns: the share of "
    "in-sample/out-of-sample splits in which the in-sample best candidate ranks below the out-of-sample median (n/a "
    "with fewer than two candidates or fewer than 8 common days).")
FILES_TEXT = (
    "`curve.csv` holds the curve for every window (feedback, pilot, held-out, and held-out on each transfer universe), "
    "one row per window, arm and step, with the mean, the interval bounds `lo` and `hi`, and `n`, the number of seeds "
    "with a value. `selection.csv` holds the selection statistics above at full precision.")
FAKE_TEXT = (
    "These runs came from the fake generator, which returns expressions from a fixed list instead of asking a language "
    "model. They are a check that the pipeline runs end to end, not a result.")


def windows(transfer_to) -> list[str]:
    return ["feedback", "pilot", "heldout", *(f"transfer:{u}:heldout" for u in transfer_to)]


def _f(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join([*lines, *("| " + " | ".join(r) + " |" for r in rows)])


def _cell(row: pd.Series, n_runs: int) -> str:
    if row.n == 0: return "n/a"
    s = f"{_f(row['mean'])} [{_f(row.lo)}, {_f(row.hi)}]"
    return s if row.n == n_runs else f"{s} ({row.n} of {n_runs} seeds)"


def _seed_count(n: int) -> str:
    return "one seed" if n == 1 else f"{n} seeds"


def _statement(ho: pd.DataFrame, fb: pd.DataFrame, n_refine: int, K: int) -> str:
    """The go/no-go paragraph, from the refine arm's held-out and feedback curves."""
    head = "The question cannot be answered from these runs: "
    if n_refine == 0:
        return head + "there is no refine arm among them."
    if n_refine < 2:
        return head + f"the refine arm has only {_seed_count(n_refine)}, and an interval over seeds needs at least two."
    h, f = ho[ho.arm == "refine"].set_index("step"), fb[fb.arm == "refine"].set_index("step")
    last = h.loc[K]
    if last.n < 2:
        return head + (f"only {_seed_count(int(last.n))} of the refine arm's {n_refine} "
                       f"{'has' if last.n == 1 else 'have'} a held-out value at step {K}, and an interval over seeds "
                       "needs at least two.")
    peak_step = int(h["mean"].idxmax())
    peak = h.loc[peak_step]
    drop, half = peak["mean"] - last["mean"], (last.hi - last.lo) / 2
    verdict = (f"The drop is larger than the half-width: the held-out score at step {K} is below its peak by more "
               "than the half-width of its interval." if drop > half else
               f"The drop is not larger than the half-width: the held-out score at step {K} is not below its peak by "
               "more than the half-width of its interval.")
    return (f"The refine arm's mean held-out rank IC peaks at {_f(peak['mean'])} at step {peak_step} "
            f"({_seed_count(int(peak.n))}) and is {_f(last['mean'])} at step {K}, the last step "
            f"({_seed_count(int(last.n))}). The drop from the peak to step {K} is {_f(drop)}; the half-width of the 95% "
            f"interval at step {K} is {_f(half)}. {verdict} This compares two means over seeds against the interval at "
            "one step; it is not a paired test.\n\n"
            f"On the feedback window, which the loop selects on, the refine arm's mean rank IC is "
            f"{_f(f.loc[0, 'mean'])} at step 0 and {_f(f.loc[K, 'mean'])} at step {K}.")


def render(runs, curves: dict[str, pd.DataFrame], sel: pd.DataFrame) -> str:
    cfg = runs[0].cfg
    K, g = cfg.budget.max_steps, cfg.generator
    arms = [a for a in ARM_NAMES if any(r.arm == a for r in runs)]
    seeds = {a: sorted(r.seed for r in runs if r.arm == a) for a in arms}
    ho, fb = curves["heldout"], curves["feedback"]
    cell = (f"This report covers {len(runs)} finished runs that share one configuration apart from the arm. The split "
            f"is `{cfg.split}` and the home universe is {cfg.universe}; the transfer universes are "
            f"{', '.join(cfg.transfer_to) or 'none'}. The generator is the `{g.provider}` provider with model "
            f"`{g.model}`. Each refining arm has a budget of {K} refinement steps, so {K + 1} generations in all.")
    seeds_text = "Seeds present: " + "; ".join(f"{ARM_NAMES[a]} {', '.join(map(str, seeds[a]))}" for a in arms) + "."
    curve_table = _table(["step", *(ARM_NAMES[a] for a in arms)],
                         [[str(k), *(_cell(ho[(ho.arm == a) & (ho.step == k)].iloc[0], len(seeds[a])) for a in arms)]
                          for k in range(K + 1)])
    sel_table = _table(list(sel.columns), [[str(r.run), ARM_NAMES[r.arm], str(r.seed), str(r.n_trials),
                                            str(r.n_valid), _f(r.dsr_of_chosen), _f(r.pbo)]
                                           for r in sel.itertuples()])
    blocks = [f"# Refinement-count curve: {cfg.split}, {cfg.universe}", *([FAKE_TEXT] if g.provider == "fake" else []),
              "## The cell", cell, ARMS_TEXT, seeds_text, METRIC_TEXT,
              "## Refine arm: does held-out fall while feedback rises?", _statement(ho, fb, len(seeds.get("refine", [])), K),
              "## Held-out curve", CURVE_TEXT.format(K=K), curve_table,
              "## Selection statistics", SELECTION_TEXT, sel_table,
              "## Files", FILES_TEXT]
    return "\n\n".join(blocks) + "\n"


def report(run_dirs, out) -> Path:
    """Write `curve.csv`, `selection.csv` and `report.md` for one cell into `out`; return the path of `report.md`."""
    runs = load_cell(run_dirs)
    out = Path(out)
    curves = {w: curve(run_dirs, window=w) for w in windows(runs[0].cfg.transfer_to)}
    sel = selection_stats(run_dirs)
    out.mkdir(parents=True, exist_ok=True)
    stacked = pd.concat([c.assign(window=w) for w, c in curves.items()], ignore_index=True)
    stacked[["window", *curves["heldout"].columns]].to_csv(out / "curve.csv", index=False)
    sel.to_csv(out / "selection.csv", index=False)
    path = out / "report.md"
    path.write_text(render(runs, curves, sel))
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.report", description="Report one finished cell.")
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        path = report(args.run_dirs, args.out)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
