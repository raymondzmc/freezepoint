#!/usr/bin/env python3
"""FNSPID (Dong et al. 2024, arXiv 2402.06698): 15.7M news items and the daily price history of the
US names they cover, 1999-2023. Downloads the chosen files from Hugging Face at a pinned revision
(data/fnspid.lock.json) and builds a point-in-time news index: one parquet per year keyed by the
publish timestamp, which is the only date an agent may use to see an item.
usage: fnspid.py --prices | --news | --all | --index
  --prices  Stock_price/full_history.zip (562 MB)
  --news    Stock_news/All_external.csv (5.5 GB): headline, text, publish date, tickers
  --all     also Stock_news/nasdaq_exteral_data.csv (22 GB)
  --index   build data/fnspid/index/<year>.parquet from whatever news files are present"""
import argparse, json, pathlib, sys, datetime
from huggingface_hub import hf_hub_download, HfApi

REPO = "Zihan1004/FNSPID"
HERE = pathlib.Path(__file__).resolve().parent
DEST = HERE / "fnspid"
LOCK = HERE / "fnspid.lock.json"
FILES = {"prices": ["Stock_price/full_history.zip"], "news": ["Stock_news/All_external.csv"],
         "all": ["Stock_news/All_external.csv", "Stock_news/nasdaq_exteral_data.csv"]}

def revision():
    if LOCK.exists(): return json.load(open(LOCK))["revision"]
    sha = HfApi().dataset_info(REPO).sha
    json.dump({"repo": REPO, "revision": sha, "pinned": datetime.date.today().isoformat()}, open(LOCK, "w"), indent=1)
    return sha

def fetch(which):
    rev = revision()
    for f in FILES[which]:
        p = hf_hub_download(REPO, f, repo_type="dataset", revision=rev, local_dir=DEST)
        print("ok", p)

def index():
    import pandas as pd
    out = DEST / "index"; out.mkdir(parents=True, exist_ok=True)
    srcs = [p for p in (DEST / "Stock_news").glob("*.csv")] if (DEST / "Stock_news").exists() else []
    if not srcs: sys.exit("no news files under data/fnspid/Stock_news; run --news first")
    for src in srcs:
        print("indexing", src.name)
        for chunk in pd.read_csv(src, usecols=lambda c: c.lower() in {"date", "article_title", "stock_symbol", "url", "publisher", "article"},
                                 chunksize=200_000, low_memory=False):
            chunk.columns = [c.lower() for c in chunk.columns]
            chunk["published"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True)
            chunk = chunk.dropna(subset=["published"])
            for year, g in chunk.groupby(chunk["published"].dt.year):
                p = out / f"{int(year)}.parquet"
                g = g.drop(columns=["date"])
                if p.exists(): g = pd.concat([pd.read_parquet(p), g])
                g.sort_values("published").to_parquet(p, index=False)
    print("index:", sorted(p.name for p in out.glob("*.parquet")))

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--prices", action="store_true"); a.add_argument("--news", action="store_true")
    a.add_argument("--all", action="store_true"); a.add_argument("--index", action="store_true"); args = a.parse_args()
    if args.prices: fetch("prices")
    if args.all: fetch("all")
    elif args.news: fetch("news")
    if args.index: index()
    if not any(vars(args).values()): a.print_help()
