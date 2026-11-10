#!/usr/bin/env python3
"""FNSPID's daily price history -> Qlib binary format at ~/.qlib/qlib_data/us_data, with an instruments
file for the covered names. The universe is the set of names FNSPID carries, which is a survivorship-
biased list: use it for the news leg, and build point-in-time index constituents separately before
any cross-sectional claim on US data.
usage: us_prices.py [--zip data/fnspid/Stock_price/full_history.zip] [--dest ~/.qlib/qlib_data/us_data]"""
import argparse, io, pathlib, subprocess, sys, tempfile, zipfile
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--zip", default=str(HERE / "fnspid" / "Stock_price" / "full_history.zip"))
    a.add_argument("--dest", default=str(pathlib.Path.home() / ".qlib" / "qlib_data" / "us_data"))
    args = a.parse_args()
    z = zipfile.ZipFile(args.zip)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fnspid_csv_"))
    n = 0
    for name in z.namelist():
        if not name.lower().endswith(".csv") or name.startswith("__MACOSX/") or pathlib.Path(name).name.startswith("._"): continue
        sym = pathlib.Path(name).stem.upper()
        df = pd.read_csv(io.BytesIO(z.read(name)))
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns: continue
        keep = {"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        if "adj close" in df.columns: keep["adj close"] = "adjclose"
        df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
        df["factor"] = (df["adjclose"] / df["close"]) if "adjclose" in df.columns else 1.0   # Qlib's adjustment factor
        df["symbol"] = sym
        df.to_csv(tmp / f"{sym}.csv", index=False); n += 1
    print(f"{n} symbols extracted to {tmp}")
    # Qlib's own converter (scripts/dump_bin.py ships with the qlib source, not the wheel): fetch it once
    dump = HERE / "cache" / "dump_bin.py"; dump.parent.mkdir(exist_ok=True)
    if not dump.exists():
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/microsoft/qlib/main/scripts/dump_bin.py", dump)
    subprocess.run([sys.executable, str(dump), "dump_all", "--data_path", str(tmp), "--qlib_dir", args.dest,
                    "--include_fields", "open,high,low,close,volume,factor", "--date_field_name", "date", "--symbol_field_name", "symbol"], check=True)
    print("ok:", args.dest)

if __name__ == "__main__": main()
