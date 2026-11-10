# freezepoint

Two papers, one harness.

- **The stopping-rule paper.** When an LLM loop refines an artifact against a verifier that is noisy and drifts, out-of-window transfer peaks early and then falls while the verifier score keeps rising. Two run-time quantities predict the turn (online verifier discrimination and the decision margin), and a stopping rule on them beats fixed-K refinement, best-of-N at equal compute, no refinement and the published stopping rules, in two domains.
- **The protocol paper.** A four-layer executable protocol for evaluating agent-discovered artifacts: audited time-locked generation, checked point-in-time tools, a trial ledger with deflation for dependent trials, and an externally timestamped commitment with a forward window. Each layer is validated by planting leaks at graded strengths and measuring detection and false-alarm rates.

The protocol is the harness the stopping-rule paper runs on.

## Layout

```
data/        download scripts; raw data is never committed (see .gitignore)
harness/     the protocol: point-in-time tool registry, trial ledger and statistics, splits
loop/        generate -> verify -> refine, every candidate logged
commit/      the commitment layer: hash the artifact set and model config, anchor externally
configs/     one YAML per experiment arm
envs/        environment notes
```

## Setup

```bash
uv sync                                   # Python 3.11 venv with pinned deps
bash data/qlib_cn.sh                      # CN daily prices -> ~/.qlib/qlib_data/cn_data (544 MB)
uv run python data/fnspid.py --prices     # FNSPID price history (562 MB)
uv run python data/fnspid.py --news       # FNSPID news with text (5.5 GB); add --all for the 22 GB file
uv run python data/us_prices.py           # FNSPID prices -> ~/.qlib/qlib_data/us_data
uv run python data/sp500_pit.py           # point-in-time S&P 500 membership -> us_data/instruments/sp500.txt
uv run python data/models.py pit-4b-ft-202412   # a time-locked generator, revision pinned in data/models.lock.json
uv run python -m harness.smoke            # loads both Qlib regions and computes Alpha158 on a few names
```

Every download pins a release tag or a Hugging Face revision, recorded in `data/*.lock.json`, so a clean checkout reproduces the same bytes.

## Rules the code enforces

- Nothing an agent sees comes from after the decision date. Tools declare their availability lag in the registry and the registry refuses a call that would violate it.
- Every candidate the loop generates is written to the ledger before it is scored. The ledger is append-only and the loop cannot read the held-out column.
- The artifact set and the generator config are hashed and anchored before the forward window opens.
