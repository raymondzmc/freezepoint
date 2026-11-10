"""The commitment layer (protocol layer 4). A hash proves identity, not time (Mohanty, arXiv
2608.20304), so the hash is anchored externally with OpenTimestamps and the receipt is kept next to it.
What `from-runs` commits: each run's chosen artifact (its `chosen.json`), its effective config (`config.yaml`: the
generator id and revision, the serving parameters, the split name, the budget) and its `provenance.json` (git commit,
prompt and source hashes, data lock files, library versions, the split's windows as dates), plus the time it was made.
The tool manifest and a forward window are not committed: no forward window is defined yet, and anchoring before one
opens is a manual step. The committed set stays private; only the hash is public,
because a visible set is an enumerable axis a search can fingerprint (Gallego, arXiv 2608.08722).
usage: commit.py make <artifacts.json> <config.yaml> <out dir>    writes commitment.json and its sha256
       commit.py from-runs <run dir or glob>... <out dir>         same, from finished run directories: every
                                                                  chosen.json, config.yaml and provenance.json
       commit.py anchor <out dir>                                 runs `ots stamp` on the sha256 file
       commit.py verify <out dir>                                 recomputes the hash (exit 1 on a mismatch) and,
                                                                  if `ots` and the .ots receipt exist, runs `ots verify`
from-runs refuses: unfinished runs (no `done`), runs whose generator provider is `fake`, runs without provenance.json,
two directories claiming the same run id, a literal path that is not a run directory (a glob skips what is not one),
and an out dir that is itself a run directory (holds chosen.json or config.yaml: the out dir argument was forgotten).
A commitment is made once: make and from-runs refuse an out dir that already holds commitment.json,
commitment.sha256 or commitment.sha256.ots; a new commitment goes in a new directory. verify also exits 1 when
`ots verify` fails, and says when a receipt exists but its timestamp was not checked because `ots` is not installed.
`ots` is the OpenTimestamps client (uv tool install opentimestamps-client)."""
import datetime
import glob
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys


def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def _write(body, out):
    out = pathlib.Path(out)
    taken = [n for n in ("commitment.json", "commitment.sha256", "commitment.sha256.ots") if (out / n).exists()]
    if taken: sys.exit(f"refusing to write into {out}: {', '.join(taken)} already exists; a commitment is made once, use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "commitment.json").write_bytes(canonical(body))
    sha = hashlib.sha256(canonical(body)).hexdigest()
    (out / "commitment.sha256").write_text(sha + "\n")
    print("commitment", sha); print("keep commitment.json private; publish commitment.sha256 and the .ots receipt")

def _now(): return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

def make(artifacts, config, out):
    import yaml
    body = {"artifacts": json.loads(pathlib.Path(artifacts).read_text()),
            "config": yaml.safe_load(pathlib.Path(config).read_text()), "made": _now()}
    _write(body, out)

def _is_run(p): return p.is_dir() and ".aborted-" not in p.name and (p / "chosen.json").is_file()

def find_runs(patterns):
    """Run directories among `patterns` (paths or globs, expanded or not): has chosen.json, not `.aborted-`. A glob
    skips what is not a run directory; a literal path (no glob characters) that is not one is an error."""
    found = {}
    for pat in patterns:
        path = pathlib.Path(pat).expanduser()
        if not glob.has_magic(str(path)):
            if not _is_run(path): sys.exit(f"from-runs: {path} is not a run directory (it does not exist, is an .aborted- directory, or has no chosen.json)")
            found[path.resolve()] = path
            continue
        for p in map(pathlib.Path, sorted(glob.glob(str(path)))):
            if _is_run(p): found[p.resolve()] = p
    return list(found.values())

def from_runs(patterns, out):
    import yaml
    if any((pathlib.Path(out) / n).exists() for n in ("chosen.json", "config.yaml")):
        sys.exit(f"from-runs: refusing to write into {out}: it is a run directory (it holds chosen.json or config.yaml); "
                 "the last argument is the output directory, a new one")
    runs = find_runs(patterns)
    if not runs: sys.exit("from-runs: no run directory matched (a run is a directory with chosen.json)")
    unfinished = [r.name for r in runs if not (r / "done").exists()]
    if unfinished: sys.exit("from-runs: refusing unfinished run(s) without `done`: " + ", ".join(unfinished))
    fake = [r.name for r in runs if (yaml.safe_load((r / "config.yaml").read_text()).get("generator") or {}).get("provider") == "fake"]
    if fake: sys.exit("from-runs: refusing fake-generator run(s): " + ", ".join(fake) + "; fake-generator runs are pipeline checks and are never committed")
    unproven = [r.name for r in runs if not (r / "provenance.json").is_file()]
    if unproven: sys.exit("from-runs: refusing run(s) without provenance.json (what code, prompts and data produced them is unknown): " + ", ".join(unproven))
    chosen = {r: json.loads((r / "chosen.json").read_text()) for r in runs}      # each parsed once
    claims = {}
    for r, c in chosen.items(): claims.setdefault(c["run"], []).append(str(r))
    dup = {rid: ds for rid, ds in claims.items() if len(ds) > 1}
    if dup: sys.exit("from-runs: directories claim the same run id: " + "; ".join(f"{rid} in {', '.join(ds)}" for rid, ds in sorted(dup.items())))
    configs = {chosen[r]["run"]: yaml.safe_load((r / "config.yaml").read_text()) for r in runs}
    provenance = {chosen[r]["run"]: json.loads((r / "provenance.json").read_text()) for r in runs}
    artifacts = sorted(chosen.values(), key=lambda c: c["run"])
    print(f"from-runs: committing {len(runs)} runs:", *sorted(r.name for r in runs), sep="\n  ", file=sys.stderr)
    _write({"artifacts": artifacts, "configs": configs, "provenance": provenance, "made": _now()}, out)

def anchor(out):
    if not shutil.which("ots"): sys.exit("install the OpenTimestamps client: uv tool install opentimestamps-client")
    subprocess.run(["ots", "stamp", str(pathlib.Path(out) / "commitment.sha256")], check=True)

def verify(out):
    out = pathlib.Path(out)
    sha = hashlib.sha256((out / "commitment.json").read_bytes()).hexdigest()
    ok = sha == (out / "commitment.sha256").read_text().strip()
    print("hash matches:", ok)
    receipt = out / "commitment.sha256.ots"
    stamped = True
    if receipt.exists() and shutil.which("ots"):
        stamped = subprocess.run(["ots", "verify", str(receipt)], check=False).returncode == 0
    elif receipt.exists():
        print(f"timestamp not checked: {receipt.name} exists but the OpenTimestamps client `ots` is not installed "
              "(uv tool install opentimestamps-client)")
    if not (ok and stamped): sys.exit(1)

def main():
    cmd = sys.argv[1:]
    if cmd[:1] == ["make"]: make(*cmd[1:4])
    elif cmd[:1] == ["from-runs"] and len(cmd) >= 3: from_runs(cmd[1:-1], cmd[-1])
    elif cmd[:1] == ["anchor"]: anchor(cmd[1])
    elif cmd[:1] == ["verify"]: verify(cmd[1])
    else: sys.exit(__doc__)

if __name__ == "__main__": main()
