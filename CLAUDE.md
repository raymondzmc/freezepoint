# freezepoint

Two papers, one harness; see README.md for the layout and the rules the code enforces.

Notion project (the status box and Log on the page are the shared state with the personal agent):
- Refine-vs-freeze: a stopping rule for noisy verifiers, https://app.notion.com/p/40e0acc50542818f8fc6d286455e2c70. It covers both papers: the contamination-proof protocol project was merged into it on 2026-11-10 (its notes and plan are the page's "Protocol" section).
- Goal: "Submit the refine-vs-freeze paper to ICML 2027, on the contamination-proof harness".

Session habits:
- Start with "where was I" (the checkpoint skill reads the project page back).
- Before the session ends, run `/checkpoint`: it writes Latest and Next step on the project page, appends the Log entry, updates Tasks and Papers, and records the commit range it covered. The morning plan in #daily reads those pages.
- Data lives outside the repo (`~/.qlib/qlib_data`, `data/fnspid/`, `data/models/`); never commit it. Every download pins a tag or revision in `data/*.lock.json`.
- API keys in `.env` only. Nothing an agent sees comes from after the decision date; every data access goes through `harness.pit`.
