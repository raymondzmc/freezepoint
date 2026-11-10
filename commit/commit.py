"""The commitment layer (protocol layer 4). A hash proves identity, not time (Mohanty, arXiv
2608.20304), so the hash is anchored externally with OpenTimestamps and the receipt is kept next to it.
What is committed: the frozen artifact set, the generator id and revision, the serving config, the tool
manifest, the split, and the forward window. The committed set stays private; only the hash is public,
because a visible set is an enumerable axis a search can fingerprint (Gallego, arXiv 2608.08722).
usage: commit.py make <artifacts.json> <config.yaml> <out dir>    writes commitment.json and its sha256
       commit.py from-runs <run dir or glob>... <out dir>         same, from finished run directories: every
                                                                  chosen.json plus each run's config.yaml
       commit.py anchor <out dir>                                 runs `ots stamp` on the sha256 file
       commit.py verify <out dir>                                 recomputes the hash (exit 1 on a mismatch) and,
                                                                  if `ots` and the .ots receipt exist, runs `ots verify`
`ots` is the OpenTimestamps client (pip install opentimestamps-client)."""
import glob, hashlib, json, pathlib, shutil, subprocess, sys, datetime

def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def _write(body, out):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "commitment.json").write_bytes(canonical(body))
    sha = hashlib.sha256(canonical(body)).hexdigest()
    (out / "commitment.sha256").write_text(sha + "\n")
    print("commitment", sha); print("keep commitment.json private; publish commitment.sha256 and the .ots receipt")

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

def make(artifacts, config, out):
    import yaml
    _write({"artifacts": json.load(open(artifacts)), "config": yaml.safe_load(open(config)), "made": _now()}, out)

def find_runs(patterns):
    """Run directories among `patterns` (paths or globs, expanded or not): has chosen.json, not `.aborted-`."""
    found = {}
    for pat in patterns:
        for p in map(pathlib.Path, sorted(glob.glob(str(pathlib.Path(pat).expanduser())))):
            if p.is_dir() and ".aborted-" not in p.name and (p / "chosen.json").is_file(): found[p.resolve()] = p
    return list(found.values())

def from_runs(patterns, out):
    import yaml
    runs = find_runs(patterns)
    if not runs: sys.exit("from-runs: no run directory matched (a run is a directory with chosen.json)")
    unfinished = [r.name for r in runs if not (r / "done").exists()]
    if unfinished: sys.exit("from-runs: refusing unfinished run(s) without `done`: " + ", ".join(unfinished))
    chosen = sorted((json.loads((r / "chosen.json").read_text()) for r in runs), key=lambda c: c["run"])
    configs = {json.loads((r / "chosen.json").read_text())["run"]: yaml.safe_load((r / "config.yaml").read_text())
               for r in runs}
    _write({"artifacts": chosen, "configs": configs, "made": _now()}, out)

def anchor(out):
    if not shutil.which("ots"): sys.exit("install the OpenTimestamps client: pip install opentimestamps-client")
    subprocess.run(["ots", "stamp", str(pathlib.Path(out) / "commitment.sha256")], check=True)

def verify(out):
    out = pathlib.Path(out)
    sha = hashlib.sha256((out / "commitment.json").read_bytes()).hexdigest()
    ok = sha == (out / "commitment.sha256").read_text().strip()
    print("hash matches:", ok)
    receipt = out / "commitment.sha256.ots"
    if shutil.which("ots") and receipt.exists(): subprocess.run(["ots", "verify", str(receipt)], check=False)
    if not ok: sys.exit(1)

def main():
    cmd = sys.argv[1:]
    if cmd[:1] == ["make"]: make(*cmd[1:4])
    elif cmd[:1] == ["from-runs"] and len(cmd) >= 3: from_runs(cmd[1:-1], cmd[-1])
    elif cmd[:1] == ["anchor"]: anchor(cmd[1])
    elif cmd[:1] == ["verify"]: verify(cmd[1])
    else: sys.exit(__doc__)

if __name__ == "__main__": main()
