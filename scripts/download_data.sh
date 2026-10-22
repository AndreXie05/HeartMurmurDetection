#!/usr/bin/env bash
# Download the CirCor DigiScope dataset (v1.0.3) into data/ from PhysioNet's
# open-data S3 mirror, then verify every file against the official SHA256SUMS.
#
# Usage: scripts/download_data.sh [dest_dir]   (default: data/)
set -euo pipefail

DEST="${1:-$(dirname "$0")/../data}"
BUCKET=https://physionet-open.s3.amazonaws.com
PREFIX=circor-heart-sound/1.0.3/
JOBS=32

mkdir -p "$DEST"
KEYS=$(mktemp)
trap 'rm -f "$KEYS"' EXIT

# List every object under the prefix (S3 returns at most 1000 keys per page).
token=""
while :; do
    url="$BUCKET/?list-type=2&prefix=$PREFIX&max-keys=1000"
    if [ -n "$token" ]; then
        url="$url&continuation-token=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$token")"
    fi
    page=$(curl -sf --retry 5 "$url")
    grep -o '<Key>[^<]*</Key>' <<<"$page" | sed 's/<[^>]*>//g' >>"$KEYS"
    token=$(grep -o '<NextContinuationToken>[^<]*' <<<"$page" | sed 's/.*>//' || true)
    [ -z "$token" ] && break
done
echo "Downloading $(wc -l <"$KEYS") files into $DEST ..."

# Download in parallel, mirroring the bucket layout under $DEST.
sed "s|^$PREFIX||" "$KEYS" | grep -v '/$' |
    xargs -P "$JOBS" -I{} sh -c 'mkdir -p "$(dirname "$1/$3")" && curl -sf --retry 5 -o "$1/$3" "$2$3" || echo "FAILED: $3"' \
        _ "$DEST" "$BUCKET/$PREFIX" {}

cd "$DEST"
sha256sum -c --quiet SHA256SUMS.txt
echo "Done: all files match SHA256SUMS.txt"
