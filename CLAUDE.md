# freezepoint

Two papers, one harness; see README.md for the layout and the rules the code enforces.

Notion projects (the status box and Log on each page are the shared state with the personal agent):
- Refine-vs-freeze: a stopping rule for noisy verifiers (the stopping-rule paper), https://app.notion.com/p/40e0acc50542818f8fc6d286455e2c70
- Contamination-proof evaluation for agent-discovered artifacts (the protocol paper), https://app.notion.com/p/40e0acc5054281b39360e3ca9720d59c
- Goal: "Submit the refine-vs-freeze paper to ICML 2027, on the contamination-proof harness".

Session habits:
- Start with "where was I" (the checkpoint skill reads the project page back).
- Before the session ends, run `/checkpoint`: it writes Latest and Next step on the project page, appends the Log entry, updates Tasks and Papers, and records the commit range it covered. The morning plan in #daily reads those pages.
- Data lives outside the repo (`~/.qlib/qlib_data`, `data/fnspid/`, `data/models/`); never commit it. Every download pins a tag or revision in `data/*.lock.json`.
- API keys in `.env` only. Nothing an agent sees comes from after the decision date; every data access goes through `harness.pit`.
