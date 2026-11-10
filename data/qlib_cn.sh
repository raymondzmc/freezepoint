#!/usr/bin/env bash
# CN daily prices in Qlib's binary format, from the chenditc/investment_data releases (built from public
# sources, refreshed daily). Pins the release tag in data/qlib_cn.lock.json so a clean checkout gets the
# same data. usage: bash data/qlib_cn.sh [release tag]   (default: the tag in the lock file, else latest)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); LOCK="$HERE/qlib_cn.lock.json"
DEST="${QLIB_CN_DIR:-$HOME/.qlib/qlib_data/cn_data}"
TAG="${1:-}"
if [[ -z "$TAG" && -f "$LOCK" ]]; then TAG=$(python3 -c "import json;print(json.load(open('$LOCK'))['tag'])"); fi
if [[ -z "$TAG" ]]; then TAG=$(curl -sS https://api.github.com/repos/chenditc/investment_data/releases/latest | python3 -c "import sys,json;print(json.load(sys.stdin)['tag_name'])"); fi
URL="https://github.com/chenditc/investment_data/releases/download/$TAG/qlib_bin.tar.gz"
mkdir -p "$DEST" "$HERE/raw"
echo "release $TAG -> $DEST"
curl -L --fail --progress-bar -o "$HERE/raw/qlib_bin.tar.gz" "$URL"
tar -xzf "$HERE/raw/qlib_bin.tar.gz" -C "$DEST" --strip-components=1
SHA=$(shasum -a 256 "$HERE/raw/qlib_bin.tar.gz" | cut -d' ' -f1)
python3 - "$LOCK" "$TAG" "$SHA" "$URL" <<'EOF'
import json, sys, datetime
lock, tag, sha, url = sys.argv[1:]
json.dump({"tag": tag, "sha256": sha, "url": url, "fetched": datetime.date.today().isoformat()}, open(lock, "w"), indent=1)
EOF
echo "ok: $(ls "$DEST/instruments" | tr '\n' ' ')"
