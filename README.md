# freezepoint

Two papers, one harness.

- **The stopping-rule paper.** When an LLM loop refines an artifact against a verifier that is noisy and drifts, out-of-window transfer peaks early and then falls while the verifier score keeps rising. Two run-time quantities predict the turn (online verifier discrimination and the decision margin), and a stopping rule on them beats fixed-K refinement, best-of-N at equal compute, no refinement and the published stopping rules, in two domains.
- **The protocol paper.** A four-layer executable protocol for evaluating agent-discovered artifacts: audited time-locked generation, checked point-in-time tools, a trial ledger with deflation for dependent trials, and an externally timestamped commitment with a forward window. Each layer is validated by planting leaks at graded strengths and measuring detection and false-alarm rates.

The protocol is the harness the stopping-rule paper runs on.

## Layout

```
analysis/    the refinement-count curve, selection statistics, cost summary and report of one cell
commit/      the commitment layer: hash the artifact set and model config, anchor the hash externally
configs/     one YAML per experiment arm
data/        download scripts; raw data is never committed (see .gitignore)
envs/        environment notes
harness/     the protocol: point-in-time tool registry, trial ledger and statistics, splits, backtest
loop/        generate -> verify -> refine, every candidate logged; `python -m loop.run` is one run
scripts/     `run_cell.sh`, which runs a whole cell and its report
tests/       the test suite
```

## Setup

```bash
uv sync --extra dev                       # Python 3.11 venv with pinned deps, plus pytest and ruff
bash data/qlib_cn.sh                      # CN daily prices -> ~/.qlib/qlib_data/cn_data (544 MB)
uv run python data/fnspid.py --prices     # FNSPID price history (562 MB)
uv run python data/fnspid.py --news       # FNSPID news with text (5.5 GB); add --all for the 22 GB file
uv run python data/us_prices.py           # FNSPID prices -> ~/.qlib/qlib_data/us_data
uv run python data/sp500_pit.py           # point-in-time S&P 500 membership -> us_data/instruments/sp500.txt
uv run python data/models.py pit-4b-ft-202412   # a time-locked generator, revision pinned in data/models.lock.json
uv run python -m harness.smoke            # loads both Qlib regions and computes Alpha158 on a few names
```

Every download pins a release tag or a Hugging Face revision, recorded in `data/*.lock.json`, so a clean checkout reproduces the same bytes. The fake-generator smoke cell below needs only the two Qlib datasets, not the news or the models.

To use a language model through the Anthropic API, put the key in a file named `.env` at the repo root, which git ignores:

```
ANTHROPIC_API_KEY=sk-ant-...
```

An `ANTHROPIC_API_KEY` set in the environment takes precedence over the file. Keys never go in configs. Set `MLFLOW_DISABLE_AGENT_HINT=1` in the shell if you want to silence the one log line that Qlib's mlflow import prints per process (`scripts/run_cell.sh` sets it for you).

## Running the tests

```bash
uv run pytest -q
```

Tests carry markers. A test marked `data` needs the Qlib or FNSPID data on disk and skips itself when it is absent; it runs by default. A test marked `api` calls a paid API and needs a key. A test marked `gpu` loads a local model and needs the `models` extra, the downloaded weights and a GPU. The default run deselects `api` and `gpu` tests through `addopts` in `pyproject.toml`. To run them on purpose, select them by name, which replaces the default deselection:

```bash
uv run pytest -q -m api       # spends money; needs the key
uv run pytest -q -m gpu       # needs `uv sync --extra models` and data/models/
uv run pytest -q -m "data or api or gpu"   # everything
```

## Running one run

A run is one arm of the comparison with one seed. The three arms are `frozen` (one proposal, kept as it is), `refine` (a proposal, then `max_steps` refinements of the best candidate so far) and `bestofn` (`n_candidates` independent proposals, the best kept). Refine and best-of-N both use `max_steps + 1` generations, so they have equal compute; frozen uses one.

```bash
uv run python -m loop.run configs/cn_refine.yaml --seed 0 --runs-dir runs
uv run python -m loop.run configs/cn_refine.yaml --seed 0 --provider fake      # no API, for checking the pipeline
uv run python -m loop.run configs/cn_refine.yaml --seed 0 --model <model id>    # override generator.model
```

`--provider` overrides `generator.provider` (`api`, `local` or `fake`) and `--model` overrides the model id and its pinned revision. The run id is computed from the effective config after the overrides, and a run's directory is `runs/<run_id>/`. Standard output carries only the run id; progress goes to standard error. The command exits non-zero on any failure.

A finished run directory contains:

- `config.yaml`, the effective config the run used.
- `ledger.jsonl`, the append-only ledger: one record per candidate, written before the candidate is scored, and a new record for each score. Each trial records the expression, whether it was valid, the generator's cost (tokens and seconds) and the line range it occupies in `calls.jsonl`.
- `calls.jsonl`, every point-in-time tool call the run made, with the date it was made as of.
- `chosen.json`, the candidate the arm ended with: the valid one with the highest rank IC on the feedback window.
- `done`, written last. A run without it is not finished and the analysis refuses it.

Running the same command again is safe. A run with `done` is left alone. A run that has `chosen.json` but no `done` (the arm finished and only the scoring died) has its missing scores filled in without generating anything again. A run with neither is renamed `<run_id>.aborted-<timestamp>`, never deleted, and started fresh. `--score-only` does the scoring step alone and refuses a run whose arm did not finish.

## Running a cell

A cell is the controlled comparison for one setting: the three arms, one config each, times the seeds (five by default), so fifteen runs, followed by the report over exactly those runs.

First run the smoke cell, which uses the fake generator and no API and checks that every stage works. It goes in its own runs directory so that fake runs and real runs never share one:

```bash
GEN=fake JOBS=5 RUNS=runs/smoke bash scripts/run_cell.sh configs/cn_frozen.yaml configs/cn_refine.yaml configs/cn_bestofn.yaml
```

On an Apple-silicon laptop this command, fifteen runs and 215 generations at `JOBS=5`, finished in 236 seconds; other settings were not timed. A real cell is slower, because every candidate is a different expression (the fake generator repeats seven, and repeated scores are cached within a run) and each generation waits on the API.

Then a real cell, with a model on the Anthropic API:

```bash
GEN=api MODEL=<model id> JOBS=3 bash scripts/run_cell.sh configs/cn_frozen.yaml configs/cn_refine.yaml configs/cn_bestofn.yaml
```

The script takes these environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `GEN` | generator provider, passed as `--provider` | the config's own |
| `MODEL` | model id, passed as `--model` | the config's own |
| `SEEDS` | seeds to run, space separated | `"0 1 2 3 4"` |
| `JOBS` | how many runs at once | `1` |
| `RUNS` | the runs directory | `runs` |
| `OUT` | the report directory | `$RUNS/report` |

It runs every config for every seed, keeps each run's standard error in `$OUT/logs/`, and passes exactly the run directories it produced to `python -m analysis.report`; it never reads other runs that happen to sit in `$RUNS`, so several cells can share one directory if each has its own `OUT` (smoke runs still get their own `RUNS`, as above). If any run fails the script names the config and seed and the log, exits non-zero and writes no report. Scoring happens inside each run, so after a failure you can run the same command again: finished runs are left alone and a run that died while scoring is scored without generating again. The last line printed is the path of `report.md`.

The report directory holds:

- `report.md`, the report in prose and tables: the cell, the go/no-go statement, the held-out curve and the selection statistics.
- `curve.csv`, the curve for every window (feedback, pilot, held-out and held-out on each transfer universe) with the mean, the 95% interval and the number of seeds per arm and step.
- `selection.csv`, one row per run: the number of trials, the number valid, the deflated Sharpe ratio of the chosen candidate and the probability of backtest overfitting.
- `cost.md`, the generations, tokens and generation seconds per arm, and the wall-clock time of the cell. Scoring time is not in the per-arm numbers.
- `logs/`, the standard error of each run.

To rebuild a report from finished runs without running anything, call the analysis directly:

```bash
uv run python -m analysis.report runs/<run_id> runs/<run_id> ... --out runs/report
```

## Reading the report

The step axis of the curve is the number of refinements spent, from 0 to `max_steps`, the same for every arm. At step k the refine and best-of-N arms have used k + 1 generations, so a row compares arms at equal compute. The frozen arm has one proposal, and its value is repeated on every row. Each cell is the mean over seeds of the score of the candidate the run holds after that step (the best on feedback rank IC so far), with a 95% confidence interval over seeds from the t distribution.

The go/no-go question is whether refinement makes the held-out score fall while the feedback score keeps rising. The section titled "Refine arm: does held-out fall while feedback rises?" answers it, and its "Start against final" paragraph is the statement to rely on: it pairs each seed's held-out score at step 0 with its score at the last step, so it does not depend on choosing a peak after seeing the means. "Peak against final" does depend on that choice and overstates a drop. The "Feedback" paragraph says how the feedback score moved over the same steps. When fewer than two seeds pair up, the report says the question cannot be answered instead of giving a verdict.

A report from the fake generator, which returns expressions from a fixed list, is a check that the pipeline runs end to end and not a result. The report says so in the line after its title.

## The commitment

After a cell is finished, its chosen artifacts and configs are hashed into a commitment. The commitment file stays private; only the hash is published.

```bash
uv run python commit/commit.py from-runs runs/<run_id> runs/<run_id> ... runs/commitment-cn-2018-2023
uv run python commit/commit.py verify runs/commitment-cn-2018-2023
```

Pass the real cell's run directories (the report's `selection.csv` lists their ids). A glob such as `"runs/*"` is expanded by the command and matches every finished run under the directory it names, so only use one on a directory that holds nothing but the cell you mean to commit. The command refuses unfinished runs and fake-generator runs, which are pipeline checks and are never committed. Check the list of runs it reports before you anchor.

`from-runs` takes finished run directories or globs (each with `done`; `.aborted-` directories are skipped) and writes `commitment.json` and `commitment.sha256` into the output directory. A commitment is written once: the command refuses an output directory that already holds one, so a new commitment goes in a new directory. `verify` recomputes the hash and exits non-zero on a mismatch, and, when the OpenTimestamps client and a receipt exist, checks the receipt too.

```bash
uv run python commit/commit.py anchor runs/commitment-cn-2018-2023
```

`anchor` publishes the hash to public timestamping servers through `ots stamp` and cannot be undone, so it is run by hand, once, when you are ready to commit. The OpenTimestamps client is installed with `uv tool install opentimestamps-client`.

## Rules the code enforces

- Nothing an agent sees comes from after the decision date. Every data access goes through the point-in-time registry, which declares each tool's availability lag and refuses a call that would violate it.
- The loop's view of the ledger hides the pilot, held-out and transfer scores. The loop sees feedback scores only.
- Candidates are validated against reading the future before anything is evaluated: an expression that looks ahead (a `Ref` with a negative offset) is recorded as invalid and never scored.
- Every candidate the loop generates is written to the ledger before it is scored. The ledger is append-only.
- The price tool returns rebased windows, so only ratios survive and a later split or other price adjustment cannot show through.
- The news cutoff is fixed when the registry is built, not at the call.
- A run aborts on missing data or an API failure. It never records the failure as a candidate, and an aborted run never gets a `done` file.
- The artifact set and the generator config are hashed and anchored before the forward window opens.

## Current limits

- The Messages API takes no sampling parameters, so `generator.temperature` must be null in every config that uses the `api` provider; the config validator rejects anything else.
- The local-model generator (`provider: local`) has not been run on real weights yet. Only its code path under test doubles has been exercised.
