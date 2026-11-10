#!/usr/bin/env python3
"""Time-locked generators for the protocol's first layer, downloaded at a pinned Hugging Face revision
and recorded in data/models.lock.json with the cutoff each one claims. The claimed cutoff is a claim:
harness/cutoff_probe.py measures the effective one before a model is used.
usage: models.py <name> [<name> ...]      names below, or any HF model id
       models.py --list"""
import datetime, json, pathlib, sys
from huggingface_hub import snapshot_download, HfApi

HERE = pathlib.Path(__file__).resolve().parent
DEST = HERE / "models"
LOCK = HERE / "models.lock.json"
NAMED = {   # short name -> (repo id, claimed training cutoff, source)
    "pit-4b-ft-202412": ("Diamegs/PIT-4B-FT-202412", "2024-12-31", "Kelly et al., arXiv 2607.11889 (monthly series 2013-12..2024-12; base and FT variants)"),
    "pit-4b-ft-201912": ("Diamegs/PIT-4B-FT-201912", "2019-12-31", "same"),
    "pit-4b-ft-201512": ("Diamegs/PIT-4B-FT-201512", "2015-12-31", "same"),
    "chrono-gpt-2019": ("manelalab/chrono-gpt-v1-20191231", "2019-12-31", "ChronoGPT, arXiv 2510.11677 (yearly 1999..2024)"),
    "chrono-gpt-2023": ("manelalab/chrono-gpt-v1-20231231", "2023-12-31", "same"),
    "datedgpt-2019-instruct": ("datedgpt/datedgpt-2019-instruct", "2019-12-31", "DatedGPT, arXiv 2603.11838 (yearly 1.3B 2013..2024, base and instruct)"),
    "datedgpt-2023-instruct": ("datedgpt/datedgpt-2023-instruct", "2023-12-31", "same"),
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        for k, (rid, cut, src) in NAMED.items(): print(f"{k:24} {rid:40} cutoff {cut}")
        return
    lock = json.load(open(LOCK)) if LOCK.exists() else {}
    for name in sys.argv[1:]:
        rid, cut, src = NAMED.get(name, (name, None, "user-supplied"))
        rev = lock.get(name, {}).get("revision") or HfApi().model_info(rid).sha
        path = snapshot_download(rid, revision=rev, local_dir=DEST / name)
        lock[name] = {"repo": rid, "revision": rev, "claimed_cutoff": cut, "source": src, "path": str(path),
                      "fetched": datetime.date.today().isoformat()}
        json.dump(lock, open(LOCK, "w"), indent=1); print("ok", name, rid, rev[:8], "->", path)

if __name__ == "__main__": main()
