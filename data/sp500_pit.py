#!/usr/bin/env python3
"""Point-in-time S&P 500 membership as a Qlib instruments file, so US cross-sections use the index as it
was on each date rather than today's list. Source: fja05680/sp500 (MIT), `sp500_ticker_start_end.csv`,
membership intervals since 1996 reconstructed from the published change history; pinned by commit in
data/sp500_pit.lock.json. Writes <us_data>/instruments/sp500.txt and reports how many members have
price data in the region (FNSPID's coverage is not complete, and tickers that changed names need care).
usage: sp500_pit.py [--dest ~/.qlib/qlib_data/us_data]"""
import argparse, datetime, json, pathlib, urllib.request
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
LOCK = HERE / "sp500_pit.lock.json"
REPO = "fja05680/sp500"; FILE = "sp500_ticker_start_end.csv"

def commit():
    if LOCK.exists(): return json.load(open(LOCK))["commit"]
    d = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/commits?per_page=1"))
    sha = d[0]["sha"]; json.dump({"repo": REPO, "file": FILE, "commit": sha, "license": "MIT", "pinned": datetime.date.today().isoformat()}, open(LOCK, "w"), indent=1)
    return sha

def main():
    a = argparse.ArgumentParser(); a.add_argument("--dest", default=str(pathlib.Path.home() / ".qlib" / "qlib_data" / "us_data")); args = a.parse_args()
    dest = pathlib.Path(args.dest); sha = commit()
    raw = HERE / "raw" / FILE; raw.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{REPO}/{sha}/{FILE}", raw)
    df = pd.read_csv(raw); df.columns = [c.lower() for c in df.columns]
    cal = pd.read_csv(dest / "calendars" / "day.txt", header=None)[0]
    last = cal.iloc[-1]
    df["end_date"] = df["end_date"].fillna(last)
    df["ticker"] = df["ticker"].str.upper().str.replace(".", "_", regex=False)   # Qlib stores BRK.B as BRK_B style names only if dumped so; keep a mapping
    have = {l.split("\t")[0] for l in open(dest / "instruments" / "all.txt")}
    out = dest / "instruments" / "sp500.txt"
    rows = [(t, s, e) for t, s, e in df[["ticker", "start_date", "end_date"]].itertuples(index=False)]
    with open(out, "w") as f:
        for t, s, e in rows: f.write(f"{t}\t{s}\t{e}\n")
    covered = sum(1 for t, _, _ in rows if t in have)
    print(f"{len(rows)} membership intervals since {df['start_date'].min()} -> {out}")
    print(f"{covered} of {len(rows)} intervals have price data in {dest.name}; missing are mostly delisted or renamed tickers")
    missing = sorted({t for t, _, _ in rows if t not in have})
    (HERE / "raw" / "sp500_missing_prices.txt").write_text("\n".join(missing))
    print(f"missing list -> data/raw/sp500_missing_prices.txt ({len(missing)} tickers)")

if __name__ == "__main__": main()
