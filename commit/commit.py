"""The commitment layer (protocol layer 4). A hash proves identity, not time (Mohanty, arXiv
2608.20304), so the hash is anchored externally with OpenTimestamps and the receipt is kept next to it.
What is committed: the frozen artifact set, the generator id and revision, the serving config, the tool
manifest, the split, and the forward window. The committed set stays private; only the hash is public,
because a visible set is an enumerable axis a search can fingerprint (Gallego, arXiv 2608.08722).
usage: commit.py make <artifacts.json> <config.yaml> <out dir>    writes commitment.json and its sha256
       commit.py anchor <out dir>                                 runs `ots stamp` on the sha256 file
       commit.py verify <out dir>                                 recomputes the hash and runs `ots verify`
`ots` is the OpenTimestamps client (pip install opentimestamps-client)."""
import hashlib, json, pathlib, shutil, subprocess, sys, datetime

def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def make(artifacts, config, out):
    import yaml
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    body = {"artifacts": json.load(open(artifacts)), "config": yaml.safe_load(open(config)),
            "made": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    (out / "commitment.json").write_bytes(canonical(body))
    sha = hashlib.sha256(canonical(body)).hexdigest()
    (out / "commitment.sha256").write_text(sha + "\n")
    print("commitment", sha); print("keep commitment.json private; publish commitment.sha256 and the .ots receipt")

def anchor(out):
    if not shutil.which("ots"): sys.exit("install the OpenTimestamps client: pip install opentimestamps-client")
    subprocess.run(["ots", "stamp", str(pathlib.Path(out) / "commitment.sha256")], check=True)

def verify(out):
    out = pathlib.Path(out)
    sha = hashlib.sha256((out / "commitment.json").read_bytes()).hexdigest()
    ok = sha == (out / "commitment.sha256").read_text().strip()
    print("hash matches:", ok)
    if shutil.which("ots"): subprocess.run(["ots", "verify", str(out / "commitment.sha256.ots")], check=False)

if __name__ == "__main__":
    cmd = sys.argv[1:]
    if cmd[:1] == ["make"]: make(*cmd[1:4])
    elif cmd[:1] == ["anchor"]: anchor(cmd[1])
    elif cmd[:1] == ["verify"]: verify(cmd[1])
    else: sys.exit(__doc__)
