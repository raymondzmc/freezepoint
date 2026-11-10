"""The report of one cell: `curve.csv`, `selection.csv`, `report.md` and `cost.md`.

    python -m analysis.report <run_dir> [<run_dir> ...] --out DIR

Guarantees. (1) It reads only one finished cell (`analysis.curves.load_cell`), so it cannot mix smoke runs with real
ones. (2) When the cell's generator is `fake`, the first line after the title of `report.md` says the runs are a
pipeline check, not a result. (3) Every comparison in the go/no-go statement is paired by seed
(`analysis.curves.paired_difference`): the refine arm's held-out score at its peak step and at step 0, each minus the
score at `max_steps`, and the feedback score's change from step 0 to `max_steps`, each over the seeds with a value at
both steps, with the left-out seeds counted. So a seed dropping out of an average can never pass for a fall. A verdict
is given only when at least two seeds pair up; otherwise the report says it cannot be answered and why. The same two
held-out comparisons follow for each transfer universe's held-out window, each under its own heading after the home
universe's statement, with that universe's mean names per day (the scores' `n_names`). (4) `cost.md`
sums the `cost` dict of every trial per arm (an empty dict, as the fake generator leaves, counts as zero). Standard
output carries only the path of `report.md`; errors go to standard error with a non-zero exit."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from analysis.curves import Paired, curve, load_cell, paired_difference, run_curve, selection_stats
from loop.score import transfer_window

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
    "with a value. `selection.csv` holds the selection statistics above at full precision. `cost.md` has the generation cost per arm.")
FAKE_TEXT = (
    "These runs came from the fake generator, which returns expressions from a fixed list instead of asking a language "
    "model. They are a check that the pipeline runs end to end, not a result.")


def windows(transfer_to) -> list[str]:
    return ["feedback", "pilot", "heldout", *(transfer_window(u, "heldout") for u in transfer_to)]


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


def _left_out(p: Paired, n_refine: int, a: int, b: int, window: str) -> str:
    if not p.left_out: return ""
    one = p.left_out == 1
    return (f" {p.left_out} of the refine arm's {n_refine} seeds {'was' if one else 'were'} left out because "
            f"{'its' if one else 'their'} {window} value at step {a} or at step {b} was missing.")


def _paired_sentence(label: str, p: Paired, n_refine: int, a: int, b: int, K: int, what: str,
                     held: str = "held-out") -> str:
    """One held-out comparison, `value(a) − value(b)` paired by seed, with its verdict (or why there is none). `held`
    names the window in the text: "held-out" for the home universe, "<u> held-out" for a transfer universe."""
    if p.n < 2:
        return (f"{label}: this cannot be answered from these runs, because only {_seed_count(p.n)} of the refine arm's "
                f"{n_refine} {'has' if p.n == 1 else 'have'} a {held} value at both step {a} and step {b}, and an "
                "interval over seeds needs at least two.")
    below = p.mean > p.half_width
    return (f"{label}: paired by seed, the {held} rank IC at step {a} minus that at step {b} averages {_f(p.mean)} "
            f"over {_seed_count(p.n)}, with 95% interval [{_f(p.lo)}, {_f(p.hi)}] (half-width {_f(p.half_width)})."
            + _left_out(p, n_refine, a, b, held)
            + f" The {held} score at step {K} is {'' if below else 'not '}below {what} by more than the half-width "
            f"of that interval{' (the interval excludes zero)' if below else ''}.")


HEAD = "The question cannot be answered from these runs: "


def _unanswerable(refine: list) -> str | None:
    if not refine:
        return HEAD + "there is no refine arm among them."
    if len(refine) < 2:
        return HEAD + f"the refine arm has only {_seed_count(len(refine))}, and an interval over seeds needs at least two."
    return None


def _peak_and_start(refine: list, c: pd.DataFrame, window: str, K: int, held: str, table: str) -> list[str] | str:
    """The two paired comparisons on `window` (its curve `c`): step K against the peak step and against step 0; or the
    reason they cannot be made."""
    if why := _unanswerable(refine):
        return why
    means = c[c.arm == "refine"].set_index("step")["mean"]
    if means.isna().all():
        return HEAD + f"no refine seed has a {held} value at any step."
    per_run = [run_curve(r, window) for r in refine]
    k_star = int(means.idxmax())
    peak = _paired_sentence("Peak against final", paired_difference(per_run, k_star, K), len(refine), k_star, K, K,
                            f"its peak (step {k_star}, the step with the highest mean {table})", held)
    start = _paired_sentence("Start against final", paired_difference(per_run, 0, K), len(refine), 0, K, K,
                             "its step-0 value", held)
    return [peak, start]


def _statement(refine: list, ho: pd.DataFrame, K: int) -> str:
    """The go/no-go paragraphs: the refine arm's held-out score at step K against its peak and against step 0, and
    its feedback score from step 0 to step K, each paired by seed."""
    pair = _peak_and_start(refine, ho, "heldout", K, "held-out", "in the table below")
    if isinstance(pair, str):
        return pair
    peak, start = pair
    n_refine = len(refine)
    fbc = paired_difference([run_curve(r, "feedback") for r in refine], K, 0)
    feedback = (f"Feedback: on the feedback window, which the loop selects on, the change in the refine arm's rank IC "
                f"from step 0 to step {K}, paired by seed, averages {_f(fbc.mean)} over {_seed_count(fbc.n)}, with 95% "
                f"interval [{_f(fbc.lo)}, {_f(fbc.hi)}]." + _left_out(fbc, n_refine, 0, K, "feedback"))
    note = ("The start-against-final comparison does not depend on picking a step after the fact, and it is the one "
            "that answers whether refinement hurts the held-out score across seeds. The peak-against-final comparison "
            "does depend on it: the peak step is the largest of several noisy means, chosen after seeing them, so that "
            "comparison overstates the drop.")
    return f"{peak}\n\n{start}\n\n{note}\n\n{feedback}"


def _names_per_day(runs, window: str, universe: str) -> str:
    """How many names per day `universe` had: the mean of `n_names` over every ok score of the cell on `window`."""
    xs = [t["scores"][window]["n_names"] for r in runs for t in r.trials
          if (t["scores"].get(window) or {}).get("ok") and t["scores"][window].get("n_names") is not None]
    if not xs:
        return f"No candidate has a {universe} held-out score that records its names per day."
    return (f"Over the {len(xs)} {universe} held-out scores in this cell, {universe} had on average "
            f"{sum(xs) / len(xs):.1f} names per day with both a factor value and a return.")


def _transfer_statement(runs, refine: list, c: pd.DataFrame, universe: str, K: int) -> str:
    """For one transfer universe: its names per day, then the same two paired comparisons as the home universe."""
    window = transfer_window(universe, "heldout")
    intro = (f"The same comparisons on {universe}, a universe the loop never saw, scored on the held-out window. "
             + _names_per_day(runs, window, universe))
    pair = _peak_and_start(refine, c, window, K, f"{universe} held-out", "on this window in `curve.csv`")
    return "\n\n".join([intro, pair] if isinstance(pair, str) else [intro, *pair])


def render(runs, curves: dict[str, pd.DataFrame], sel: pd.DataFrame) -> str:
    cfg = runs[0].cfg
    K, g = cfg.budget.max_steps, cfg.generator
    arms = [a for a in ARM_NAMES if any(r.arm == a for r in runs)]
    seeds = {a: sorted(r.seed for r in runs if r.arm == a) for a in arms}
    ho = curves["heldout"]
    refine = [r for r in runs if r.arm == "refine"]
    cell = (f"This report covers {len(runs)} finished runs that share one configuration apart from the arm. The split "
            f"is `{cfg.split}` and the home universe is {cfg.universe}; the transfer universes are "
            f"{', '.join(cfg.transfer_to) or 'none'}. The generator is the `{g.provider}` provider with model "
            f"`{g.model}`. The frozen arm makes one proposal; the refine arm makes a proposal and then {K} "
            f"refinements, and the best-of-N arm makes {K + 1} proposals, so those two have {K + 1} generations each.")
    seeds_text = "Seeds present: " + "; ".join(f"{ARM_NAMES[a]} {', '.join(map(str, seeds[a]))}" for a in arms) + "."
    curve_table = _table(["step", *(ARM_NAMES[a] for a in arms)],
                         [[str(k), *(_cell(ho[(ho.arm == a) & (ho.step == k)].iloc[0], len(seeds[a])) for a in arms)]
                          for k in range(K + 1)])
    sel_table = _table(list(sel.columns), [[str(r.run), ARM_NAMES[r.arm], str(r.seed), str(r.n_trials),
                                            str(r.n_valid), _f(r.dsr_of_chosen), _f(r.pbo)]
                                           for r in sel.itertuples()])
    blocks = [f"# Refinement-count curve: {cfg.split}, {cfg.universe}", *([FAKE_TEXT] if g.provider == "fake" else []),
              "## The cell", cell, ARMS_TEXT, seeds_text, METRIC_TEXT,
              "## Refine arm: does held-out fall while feedback rises?",
              _statement(refine, ho, K),
              *(x for u in cfg.transfer_to
                for x in (f"## Transfer to {u}: held-out window",
                          _transfer_statement(runs, refine, curves[transfer_window(u, "heldout")], u, K))),
              "## Held-out curve", CURVE_TEXT.format(K=K), curve_table,
              "## Selection statistics", SELECTION_TEXT, sel_table,
              "## Files", FILES_TEXT]
    return "\n\n".join(blocks) + "\n"


COST_TEXT = (
    "The numbers cover the generations (the calls to the language model that proposed or refined a candidate) of the "
    "finished runs in this report, summed over each arm's trials from the cost the generator recorded: tokens read, "
    "tokens written, and seconds spent waiting for the generator. A trial from the fake generator records no cost and "
    "counts as zero. Scoring time is not in them.")


def cost_table(runs) -> str:
    """`cost.md`: one row per arm present, then a total row, summed over the trials' `cost` dicts."""
    rows, total = [], [0, 0, 0, 0, 0.0]
    for a in (a for a in ARM_NAMES if any(r.arm == a for r in runs)):
        ts = [t for r in runs if r.arm == a for t in r.trials]
        row = [sum(r.arm == a for r in runs), len(ts), *(sum((t.get("cost") or {}).get(k) or 0 for t in ts)
                                                         for k in ("input_tokens", "output_tokens", "seconds"))]
        total = [x + y for x, y in zip(total, row)]
        rows.append([ARM_NAMES[a], *map(str, row[:4]), f"{row[4]:.1f}"])
    rows.append(["total", *map(str, total[:4]), f"{total[4]:.1f}"])
    head = ["arm", "runs", "generations", "input tokens", "output tokens", "generation seconds"]
    return "# Cost\n\n" + COST_TEXT + "\n\n" + _table(head, rows) + "\n"


def report(run_dirs, out) -> Path:
    """Write `curve.csv`, `selection.csv`, `cost.md` and `report.md` for one cell into `out`; return the path of
    `report.md`."""
    runs = load_cell(run_dirs)
    out = Path(out)
    curves = {w: curve(run_dirs, window=w) for w in windows(runs[0].cfg.transfer_to)}
    sel = selection_stats(run_dirs)
    out.mkdir(parents=True, exist_ok=True)
    stacked = pd.concat([c.assign(window=w) for w, c in curves.items()], ignore_index=True)
    stacked[["window", *curves["heldout"].columns]].to_csv(out / "curve.csv", index=False)
    sel.to_csv(out / "selection.csv", index=False)
    (out / "cost.md").write_text(cost_table(runs))
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
