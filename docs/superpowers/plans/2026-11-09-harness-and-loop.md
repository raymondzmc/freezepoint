# Harness and Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the controlled comparison the week-5 go/no-go needs: frozen vs K-step refinement vs best-of-N at equal compute, on CSI 300 transferring to CSI 500 and the S&P 500, five seeds, with every candidate logged and scored on feedback, pilot, held-out and transfer windows the loop cannot see.

**Architecture:** Agents only touch data through `harness.pit` tools whose availability rule is declared and checked; every candidate is appended to `harness.ledger` before it is scored, and the loop's view of the ledger hides the pilot, held-out and forward columns. The loop (`loop/`) has three arms sharing one generator interface (API or local time-locked model), one backtest (`harness.backtest`, Qlib expressions with the Alpha158 label convention), and one post-run scorer; `analysis/` turns the ledger into the refinement-count curve with confidence intervals, DSR and PBO.

**Tech Stack:** Python 3.11, Qlib 0.9, pandas, scipy, statsmodels, pyyaml, pytest; the Anthropic Python SDK for the API generator; transformers for the local generator (optional extra).

**Spec:** `~/person-agent/research/2026-11-09-research-plan-A-B-C.md` sections 1 and 2 (weeks 1–5 of the stopping-rule paper; layers 2 and 3 of the protocol), plus the rules in `README.md`.

This is plan 1 of 4. Plans 2–4 (the diagnostic and stopping rule; leak planting and layer validation; the code-repair domain) consume this plan's interfaces and are written after the controlled comparison runs.

## Global Constraints

- Python `>=3.11,<3.12`; dependencies only through `pyproject.toml` and `uv.lock`.
- Nothing an agent sees comes from after the decision date: every data access goes through `harness.pit.Registry.call`, which raises `LookAheadError` on a violation.
- Every candidate is appended to the ledger before it is scored; the ledger is append-only (scores are new records).
- The loop reads the ledger only through `LoopView`, which hides `pilot`, `heldout`, `forward` and every transfer window.
- Label convention (Qlib Alpha158): forward return `Ref($close, -2) / Ref($close, -1) - 1`; a candidate expression may not contain `Ref(..., -k)` for `k > 0`.
- Equal compute across arms: `budget.max_steps + 1` generations for refine and `budget.n_candidates = budget.max_steps + 1` for best-of-N; the frozen arm uses one.
- Five seeds per cell (`seeds: [0,1,2,3,4]`); runs are reproducible from the config, the seed and the lock files.
- Default split `cn-2018-2023`: feedback 2018-01-01..2021-12-31, pilot 2022-01-17..2022-06-30, heldout 2022-07-15..2023-12-31, embargo 10 trading days; transfer universes `csi500` and `sp500` scored on the same windows.
- Raw data never enters git; runs write to `runs/<run_id>/` (gitignored).

## Review Focus

1. An expression that evaluates to all-NaN or constant on the feedback window has no IC; the backtest returns a failed score (`ok=False`, reason) and the trial still counts. Test in Task 4.
2. A generator returns text that is not a valid expression or contains a negative `Ref`; it is logged as an invalid trial, counts toward the trial count, and the loop continues. Tests in Tasks 3 and 8.
3. `as_of` falls on a weekend or holiday; the prices tool returns data up to the last trading day on or before `as_of`, never after. Test in Task 2.
4. A news row whose `published` has no timezone is treated as UTC, and the news tool returns nothing published on or after the cutoff instant for `as_of`. Test in Task 2.
5. Two processes append to one ledger (five seeds in parallel); records never interleave mid-line. Test in Task 1.

---

### Task 1: Ledger views and locking

**Files:**
- Modify: `harness/ledger/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Produces: `Ledger(path)`, `Ledger.append(Trial) -> str`, `Ledger.score(trial_id, window, metrics)`, `Ledger.trials(hide=()) -> dict[str, dict]`, `Ledger.count(run=None) -> int`; `Ledger.SECRET = ("pilot", "heldout", "forward")`; `LoopView(ledger)` hides `SECRET` plus any window whose name starts with `transfer:`; `DiagnosticView(ledger)` hides `("heldout", "forward")` plus `transfer:*`; `Trial` dataclass as in the file, plus field `valid: bool = True` and `error: str = ""`.

- [ ] **Step 1: Write the failing tests**

```python
def test_loop_view_hides_pilot_heldout_and_transfer(tmp_path):
    led = Ledger(tmp_path / "l.jsonl"); tid = led.append(Trial(arm="refine", run="r", step=0, artifact="$close"))
    for w in ("feedback", "pilot", "heldout", "transfer:csi500"): led.score(tid, w, {"ic": 0.1})
    assert set(LoopView(led).trials()[tid]["scores"]) == {"feedback"}
    assert set(DiagnosticView(led).trials()[tid]["scores"]) == {"feedback", "pilot"}
    assert set(led.trials()[tid]["scores"]) == {"feedback", "pilot", "heldout", "transfer:csi500"}

def test_invalid_trial_counts(tmp_path):
    led = Ledger(tmp_path / "l.jsonl"); led.append(Trial(arm="refine", run="r", step=1, artifact="Ref($close,-1)", valid=False, error="negative Ref"))
    assert led.count("r") == 1

def test_parallel_appends_do_not_interleave(tmp_path):
    # 4 processes x 200 appends via multiprocessing; then every line parses as JSON and count == 800
```

- [ ] **Step 2: Run tests, expect failures** — `uv run pytest tests/test_ledger.py -v` → FAIL (`DiagnosticView` undefined; `valid` unexpected).
- [ ] **Step 3: Implement** — add `valid`/`error` to `Trial`; `SECRET = ("pilot", "heldout", "forward")`; hide `transfer:*` by prefix in `trials(hide=...)`; add `DiagnosticView`; wrap each append in `fcntl.flock(f, LOCK_EX)` and write the line with a single `f.write`.
- [ ] **Step 4: Run tests, expect pass** — `uv run pytest tests/test_ledger.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "ledger: diagnostic view, pilot hidden from the loop, invalid trials, locked appends"`.

### Task 2: Point-in-time tools over Qlib and the news index

**Files:**
- Create: `harness/pit/tools.py`, `harness/pit/news.py`
- Test: `tests/test_tools.py` (marked `@pytest.mark.data`, skipped when `~/.qlib/qlib_data/cn_data` or `data/fnspid/index` is absent)

**Interfaces:**
- Consumes: `harness.pit.registry.Registry`, `harness.splits.Split`.
- Produces: `make_registry(region: str, log_path) -> Registry` with tools:
  - `prices(as_of, instruments: list[str], fields: list[str], lookback_days: int = 250) -> pd.DataFrame` indexed `(instrument, datetime)`, `fields` are Qlib field names like `$close`; data through the last trading day `<= as_of`; registered with `lag_days=0`, date from the index level `datetime`.
  - `universe(as_of, market: str) -> list[str]` via `D.list_instruments` with `start_time=end_time=as_of`; registered `value_dependent=False`, returns a DataFrame with a `date` column equal to `as_of` so the registry check passes.
  - `news(as_of, symbols: list[str], days: int = 5, cutoff_hour_utc: int = 0) -> pd.DataFrame` with columns `published, stock_symbol, article_title, publisher, url`, rows with `published < as_of 00:00 UTC + cutoff_hour_utc`; registered `date_col="published"`, `lag_days=0`.
  - `qlib_init(region)` idempotent helper in `tools.py`.
- `harness/pit/news.py`: `NewsIndex(root=data/fnspid/index)`, `.query(symbols, start_utc, end_utc) -> pd.DataFrame` reading only the year partitions the range touches.

- [ ] **Step 1: Write the failing tests**

```python
def test_prices_never_after_as_of(cn_registry):
    df = cn_registry.call("prices", "2022-01-02", instruments=["SH600000"], fields=["$close"], lookback_days=20)  # a Sunday
    assert df.index.get_level_values("datetime").max() <= pd.Timestamp("2021-12-31")

def test_prices_leak_is_refused(cn_registry, monkeypatch):
    # monkeypatch the tool fn to return one extra day; expect LookAheadError from registry.call

def test_news_cutoff_is_exclusive_and_utc(news_registry):
    df = news_registry.call("news", "2019-03-05", symbols=["AAPL"], days=3)
    assert df["published"].max() < pd.Timestamp("2019-03-05", tz="UTC")
    assert str(df["published"].dt.tz) == "UTC"

def test_universe_is_point_in_time(us_registry):
    assert "AABA" in us_registry.call("universe", "2017-03-01", market="sp500") and "AABA" not in us_registry.call("universe", "2018-03-01", market="sp500")
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/test_tools.py -v -m data` → FAIL (module missing).
- [ ] **Step 3: Implement `tools.py` and `news.py`** — `prices` uses `D.features(instruments, fields, start_time=as_of - lookback_days, end_time=as_of)`; the registry check reads the `datetime` index level (extend `Registry.call` to accept a MultiIndex level named `datetime`, already handled); `universe` wraps the list in a one-column DataFrame for the check and returns the list; `news` uses `NewsIndex.query` with `end_utc = Timestamp(as_of, tz="UTC") + Timedelta(hours=cutoff_hour_utc)` and strict `<`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "pit tools: prices, universe, news over Qlib and the FNSPID index"`.

### Task 3: Expression validation

**Files:**
- Create: `harness/factors.py`
- Test: `tests/test_factors.py`

**Interfaces:**
- Produces: `validate(expr: str, region: str = "cn") -> tuple[bool, str]`; `FORBIDDEN = re.compile(r"Ref\s*\([^)]*,\s*-\s*\d")`; `ALLOWED_OPS` = the Qlib operator names Alpha158 uses (`Ref, Mean, Std, Max, Min, Sum, Rank, Corr, Cov, Delta, Slope, Rsquare, Resi, Quantile, Abs, Log, Sign, Power, If, Greater, Less, EMA, WMA, Mad, Idxmax, Idxmin`) plus `$open,$high,$low,$close,$volume,$factor`; `LABEL = "Ref($close, -2) / Ref($close, -1) - 1"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_negative_ref_is_rejected():
    ok, why = validate("Ref($close, -1) / $close"); assert not ok and "future" in why
def test_unknown_operator_is_rejected():
    ok, _ = validate("Magic($close, 5)"); assert not ok
def test_alpha158_style_expression_passes():
    ok, _ = validate("Mean($close, 5) / $close - 1"); assert ok
def test_label_itself_is_allowed_only_as_label():
    assert not validate(LABEL)[0]
```

- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement** — regex reject first; then parse with Qlib's `qlib.data.base.Expression` via `D.features` on one instrument and 30 days wrapped in try/except, returning the error text.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "factors: expression validation, no look-ahead operators"`.

### Task 4: Backtest

**Files:**
- Create: `harness/backtest.py`
- Test: `tests/test_backtest.py` (`@pytest.mark.data`)

**Interfaces:**
- Consumes: `validate`, `LABEL` (Task 3); `Window` (`harness.splits`); `qlib_init`.
- Produces: `@dataclass FactorScore: ok: bool; reason: str; ic: float; icir: float; ls_mean: float; ls_sharpe: float; turnover: float; n_days: int; daily_ls: list[float]`; `evaluate(expr: str, window: Window, universe: str, region: str, quantile: float = 0.2) -> FactorScore`; `score_dict(fs) -> dict` (everything but `daily_ls`, which the ledger stores under `"daily_ls"` for PBO).

Definitions the implementer must use: rank IC = daily Spearman correlation between the expression value and `LABEL` across the universe, averaged over days; ICIR = mean / std of daily IC; long-short = top-quantile mean label minus bottom-quantile mean label per day; `ls_sharpe = mean/std * sqrt(252)`; turnover = mean daily fraction of the top quantile that changed from the previous day. Days with fewer than 30 valid names are dropped; `ok=False` with `reason` when fewer than 60 days remain or IC is NaN.

- [ ] **Step 1: Write the failing tests**

```python
def test_momentum_factor_scores_on_csi300(cn):
    fs = evaluate("$close / Ref($close, 20) - 1", Window("2019-01-01", "2019-12-31"), "csi300", "cn")
    assert fs.ok and fs.n_days > 200 and abs(fs.ic) < 0.2 and len(fs.daily_ls) == fs.n_days
def test_constant_expression_fails_cleanly(cn):
    fs = evaluate("$close * 0 + 1", Window("2019-01-01", "2019-03-31"), "csi300", "cn")
    assert not fs.ok and "ic" in fs.reason.lower()
def test_score_dict_has_no_series():
    assert "daily_ls" not in score_dict(FactorScore(True, "", 0, 0, 0, 0, 0, 10, [0.0]*10))
```

- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement `evaluate`** — one `D.features(instruments, [expr, LABEL], start, end)` call, then pandas groupby on `datetime`.
- [ ] **Step 4: Run, expect pass** (the momentum test takes under a minute on the CN data).
- [ ] **Step 5: Commit** — `git commit -m "backtest: rank IC, ICIR, long-short Sharpe, turnover with the Alpha158 label"`.

### Task 5: Generator interface, API and local implementations

**Files:**
- Create: `loop/generator.py`, `loop/prompts/propose.md`, `loop/prompts/refine.md`
- Test: `tests/test_generator.py` (uses `FakeGenerator`; API and local tests marked `@pytest.mark.api` / `@pytest.mark.gpu`, skipped by default)

**Interfaces:**
- Produces: `class Generator(Protocol): propose(self, context: str, n: int, seed: int) -> list[str]; refine(self, expr: str, feedback: dict, history: list[dict], seed: int) -> str`; `FakeGenerator(exprs: list[str])` cycling through a list; `ApiGenerator(model: str, temperature: float, max_output_tokens: int)` using the Anthropic Messages API with the key from `.env` (`ANTHROPIC_API_KEY`), recording `usage` tokens into `last_cost: dict`; `LocalGenerator(name: str)` loading `data/models/<name>` with transformers, greedy decoding with the seed set; `parse_expression(text) -> str` taking the last fenced block or the last line that starts with `$` or an operator name.
- Prompt files: `propose.md` states the operator list from Task 3, the fields, the window dates, and asks for exactly one expression in a fenced block; `refine.md` adds the feedback dict (`ic, icir, ls_sharpe, turnover`) and the last three history entries and asks for one improved expression.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_expression_takes_last_fenced_block():
    assert parse_expression("reasoning\n```\nMean($close, 5)/$close - 1\n```\n") == "Mean($close, 5)/$close - 1"
def test_fake_generator_cycles_and_is_seed_stable():
    g = FakeGenerator(["$close", "$open"]); assert g.propose("", 3, 0) == ["$close", "$open", "$close"]
def test_prompt_templates_render_without_missing_keys():
    render("propose", {...all keys...}); render("refine", {...})
```

- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement** — templates rendered with `str.format_map`; `ApiGenerator` retries 3 times on rate limit with backoff; `LocalGenerator` behind `try: import torch, transformers` with a clear error naming `uv sync --extra models`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "generator: protocol, prompts, fake, API and local implementations"`.

### Task 6: Run configuration and run directory

**Files:**
- Create: `loop/config.py`
- Modify: `configs/example.yaml` (rename to `configs/cn_refine.yaml`; add `configs/cn_frozen.yaml`, `configs/cn_bestofn.yaml` differing only in `arm`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `@dataclass RunConfig` mirroring the YAML (`arm, split, universe, transfer_to, seeds, generator, control_generator, budget, verifier, stopping`); `load(path) -> RunConfig`; `run_id(cfg, seed) -> str` = `f"{cfg.arm}-{cfg.split}-s{seed}-{sha256(canonical yaml)[:8]}"`; `run_dir(cfg, seed) -> Path` = `runs/<run_id>/`, created with `config.yaml` copied in.

- [ ] **Step 1: Tests** — `test_run_id_is_stable_for_same_config`, `test_bestofn_budget_equals_refine_budget` (asserts `n_candidates == max_steps + 1` after load, raising `ValueError` otherwise), `test_three_configs_differ_only_in_arm`.
- [ ] **Step 2: Run, expect failure.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "loop: run config, run ids, three arm configs"`.

### Task 7: The three arms

**Files:**
- Create: `loop/arms.py`
- Test: `tests/test_arms.py` (uses `FakeGenerator` and a fake `evaluate` monkeypatched to score by expression length, so no data is needed)

**Interfaces:**
- Consumes: `Generator`, `RunConfig`, `Ledger`/`LoopView` (Task 1), `validate` (Task 3), `evaluate`/`score_dict` (Task 4), `Split`.
- Produces: `run_arm(cfg: RunConfig, seed: int, gen: Generator, ledger: Ledger, evaluate=evaluate) -> str` returning the chosen trial id; behaviour per arm:
  - `frozen`: one proposal, scored on feedback, chosen.
  - `refine`: one proposal, then `max_steps` refinements; each candidate validated, appended, scored on feedback; the loop keeps the best-so-far by `ic` (greedy keep-if-better) and refines from it; an invalid candidate is appended with `valid=False` and the loop refines from the current best; chosen = best-so-far at the end.
  - `bestofn`: `n_candidates` independent proposals (seed offset by index), all scored on feedback, chosen = best `ic`.
  - All arms: the feedback score the loop sees comes through `LoopView`; the pilot score is written by `run_arm` after each candidate but never read by it; cost (tokens, seconds) recorded per trial; the chosen id is written to `runs/<run_id>/chosen.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_refine_logs_every_candidate_and_keeps_best(tmp_path): ... assert ledger.count(run) == cfg.budget.max_steps + 1
def test_bestofn_uses_equal_generations(tmp_path): ... assert ledger.count(run) == cfg.budget.n_candidates
def test_invalid_candidate_is_logged_and_loop_continues(tmp_path): gen yields "Ref($close,-1)" once; assert one record has valid False and the run still finishes
def test_loop_never_reads_pilot(tmp_path): spy on LoopView.trials; assert "pilot" not in any returned scores
```

- [ ] **Step 2: Run, expect failure.** - [ ] **Step 3: Implement `run_arm`.** - [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "loop: frozen, refine and best-of-N arms at equal compute"`.

### Task 8: Post-run scoring and the CLI

**Files:**
- Create: `loop/score.py`, `loop/run.py`
- Test: `tests/test_score.py` (fake evaluate), `tests/test_cli.py` (runs `python -m loop.run` with the fake generator via `generator.provider: fake`)

**Interfaces:**
- Produces: `score_run(cfg, run_id, ledger, evaluate=evaluate)` writing, for every trial in the run (valid ones), `heldout` on the home universe and `transfer:<u>` for each `u` in `cfg.transfer_to` on feedback, pilot and heldout windows (names `transfer:csi500:heldout` etc.); `python -m loop.run <config> --seed N [--score-only]` producing `runs/<run_id>/{config.yaml, ledger.jsonl, calls.jsonl, chosen.json, done}`; exit code 0 and the run id on stdout.
- Modify: `loop/generator.py` gains `provider: fake` with `exprs` from the config for tests.

- [ ] **Step 1: Tests** — `test_score_run_writes_heldout_and_transfer_for_every_valid_trial`, `test_cli_end_to_end_with_fake_generator(tmp_path)` asserting the five files exist and `done` is written last.
- [ ] **Step 2: Run, expect failure.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "loop: post-run scoring on held-out and transfer windows; CLI"`.

### Task 9: The refinement-count curve and the report

**Files:**
- Create: `analysis/__init__.py`, `analysis/curves.py`, `analysis/report.py`
- Test: `tests/test_curves.py` (synthetic ledgers written by a helper)

**Interfaces:**
- Consumes: `Ledger.trials()` (full view), `harness.ledger.stats.deflated_sharpe`, `pbo_cscv`.
- Produces: `curve(run_dirs: list[Path], window: str = "heldout") -> pd.DataFrame` with columns `arm, step, mean, lo, hi, n` where for the refine arm the value at step `k` is the `window` score of the best-so-far (by feedback `ic`) after `k` candidates, for best-of-N the best-of-first-`k` by feedback, and for frozen the single value repeated; `lo/hi` the 95% CI over seeds (t-distribution); `selection_stats(run_dirs) -> pd.DataFrame` with per run: `n_trials, dsr_of_chosen, pbo` (PBO from the `daily_ls` series of all valid trials, `S=8`); `report(run_dirs, out: Path)` writing `curve.csv`, `selection.csv` and `report.md` with the two tables and one line stating whether the refine arm's held-out score at `max_steps` is below its peak by more than the CI half-width.

- [ ] **Step 1: Tests** — `test_curve_bestsofar_is_monotone_on_feedback_and_not_on_heldout`, `test_frozen_is_flat`, `test_ci_width_shrinks_with_more_seeds`, `test_report_writes_three_files`.
- [ ] **Step 2: Run, expect failure.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "analysis: refinement-count curve with CIs, DSR and PBO per run, report"`.

### Task 10: Commitment of a run set

**Files:**
- Modify: `commit/commit.py` (add `from-runs`)
- Test: `tests/test_commit.py`

**Interfaces:**
- Produces: `python commit/commit.py from-runs <runs glob> <out dir>` collecting every `chosen.json` artifact and each run's `config.yaml` into `commitment.json`, then the existing `make`/`anchor`/`verify`; `verify` exits 1 on a mismatch.

- [ ] **Step 1: Tests** — `test_from_runs_collects_chosen_artifacts(tmp_path)`, `test_verify_detects_tampering(tmp_path)` (edit one byte of `commitment.json`, expect exit 1).
- [ ] **Step 2: Run, expect failure.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -m "commit: commitment from a run set, verify exits non-zero on mismatch"`.

### Task 11: The controlled comparison, first cell

**Files:**
- Create: `scripts/run_cell.sh` (loops seeds 0–4 over the three configs, then `score`, then `analysis.report`)
- Modify: `README.md` (how to run a cell; where the report lands)

- [ ] **Step 1: Smoke cell with the fake generator** — `GEN=fake bash scripts/run_cell.sh configs/cn_*.yaml` → `runs/report/report.md` exists, 15 run dirs, each with `done`.
- [ ] **Step 2: One real seed on a mid-tier API model** — `uv run python -m loop.run configs/cn_refine.yaml --seed 0`; expected: 21 trials in the ledger, every one with feedback and pilot scores, no `LookAheadError` in `calls.jsonl`, cost recorded.
- [ ] **Step 3: Full cell, five seeds, three arms, frontier model** — record wall-clock and token cost in `runs/report/cost.md`.
- [ ] **Step 4: Read the curve** — the go/no-go input: does the refine arm's held-out score fall below its peak outside the CI while feedback keeps rising? Write the answer into the Notion project's status box through `/checkpoint`.
- [ ] **Step 5: Commit** — `git commit -m "scripts: run a full cell; README"`.

---

## Self-review notes

- Spec coverage: layers 2 and 3 (Tasks 1–4), the three arms at equal compute with every candidate logged (6–8), the curve with CIs and DSR/PBO (9), commitment (10), CSI 300 → CSI 500 and S&P 500 (`transfer_to` in configs, Task 8), five seeds (11). Not in this plan, by design: the bull/bear regime split (a second `Split` with `regime` set, run as a second cell after Task 11), the diagnostic and stopping rule, leak planting, the code domain, the cutoff probe for time-locked models.
- Interfaces named consistently: `evaluate`/`score_dict`/`FactorScore` (4) used in 7–9; `LoopView`/`DiagnosticView` (1) used in 7; `run_id`/`run_dir` (6) used in 8–9; window names `feedback`, `pilot`, `heldout`, `transfer:<u>:<window>` throughout.
- Review-focus items each have a test in the owning task (1: Task 4; 2: Tasks 3 and 7; 3 and 4: Task 2; 5: Task 1).
