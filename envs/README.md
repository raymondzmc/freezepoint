# Environments

- Python 3.11 through `uv sync` (pyproject.toml pins the ranges; `uv.lock` pins the exact versions and is committed).
- Qlib data lives outside the repo at `~/.qlib/qlib_data/{cn_data,us_data}`; override with `QLIB_CN_DIR`.
- Time-locked models download to `data/models/<name>` (gitignored) and need `uv sync --extra models` plus a GPU with 16 GB for the 4B ones. On a Mac, `torch` runs them on MPS for probing, not for experiments.
- API keys go in `.env` (gitignored), never in configs; configs record only the model id and revision.
- `ots` (OpenTimestamps client) for the commitment layer: `uv tool install opentimestamps-client`.
