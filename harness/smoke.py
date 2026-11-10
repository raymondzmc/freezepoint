"""Smoke test: Qlib initializes on each region that is present, Alpha158 computes for a few names, the
registry refuses a look-ahead, the ledger hides the held-out column, the statistics run on noise.
usage: python -m harness.smoke"""
import pathlib
import tempfile

import numpy as np
import pandas as pd


def qlib_region(region, provider):
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(provider), region=region)
    inst = D.instruments(market="all")
    names = D.list_instruments(inst, start_time="2022-01-01", end_time="2022-01-31", as_list=True)[:3]
    cal = D.calendar(start_time="2022-01-01", end_time="2022-01-31")
    from qlib.contrib.data.handler import Alpha158
    h = Alpha158(instruments=names, start_time="2022-01-04", end_time="2022-01-31", fit_start_time="2022-01-04", fit_end_time="2022-01-31")
    df = h.fetch(col_set="feature")
    print(f"{region}: {len(cal)} trading days in Jan 2022, Alpha158 on {names} -> {df.shape}")

def main():
    from harness.pit.tools import REGIONS, data_dir, data_present
    for region in REGIONS:
        if data_present(region): qlib_region(region, data_dir(region))
        else: print(f"{region}: no data at {data_dir(region)} (run the download first)")
    from harness.pit.registry import LookAheadError, Registry
    reg = Registry(pathlib.Path(tempfile.mkdtemp()) / "calls.jsonl")
    @reg.register("prices", lag_days=0, doc="daily closes up to as_of")
    def prices(as_of, days=5):
        end = pd.Timestamp(as_of) + pd.Timedelta(days=days)          # deliberately leaky
        return pd.DataFrame({"date": pd.date_range(end=end, periods=10), "close": np.arange(10.0)})
    try: reg.call("prices", "2022-03-01"); print("registry: FAILED to catch look-ahead")
    except LookAheadError as e: print("registry: caught look-ahead ->", str(e)[:80])
    from harness.ledger.ledger import Ledger, LoopView, Trial
    led = Ledger(pathlib.Path(tempfile.mkdtemp()) / "ledger.jsonl")
    tid = led.append(Trial(arm="refine", run="smoke", step=0, artifact="Ref($close, 1)/$close - 1"))
    led.score(tid, "feedback", {"ic": 0.03}); led.score(tid, "heldout", {"ic": -0.01})
    view = LoopView(led).trials()[tid]["scores"]
    print("ledger: loop sees", view, "| full:", led.trials()[tid]["scores"])
    from harness.ledger.stats import deflated_sharpe, pbo_cscv
    rng = np.random.default_rng(0); R = rng.normal(0, 0.01, size=(480, 50))
    print("stats: DSR of the best of 50 null trials =", round(deflated_sharpe(float((R.mean(0)/R.std(0)).max()), 50, float((R.mean(0)/R.std(0)).var()), 480), 3),
          "| PBO =", pbo_cscv(R, S=8)["pbo"])

if __name__ == "__main__": main()
