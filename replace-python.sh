#!/bin/bash
# Replace embedded Python in Tellar.app from a full python-build-standalone tarball.
# Used for the iconography diagnostic — preserves site-packages and the rest of the bundle.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/Tellar.app"
PY_DIR="$APP/Contents/Resources/python"
CACHE="$ROOT/build-cache"

TARBALL="$(ls -1 $CACHE/cpython-3.12*-aarch64-apple-darwin-pgo+lto-full.tar.zst 2>/dev/null | head -1 || true)"
if [ -z "$TARBALL" ]; then
    echo "ERROR: pgo+lto-full tarball not found in $CACHE/"
    exit 1
fi
echo "==> Using $TARBALL"

if [ ! -d "$APP" ]; then
    echo "ERROR: Tellar.app not found at $APP"
    exit 1
fi

WORK="$(mktemp -d /tmp/pybs-XXXXXX)"
trap "rm -rf $WORK" EXIT

echo "==> Extracting to $WORK"
zstd -d -c "$TARBALL" | tar -xf - -C "$WORK"

if [ ! -d "$WORK/python/install" ]; then
    echo "ERROR: unexpected archive layout (no python/install/)"
    ls "$WORK"
    exit 1
fi

echo "==> Replacing $PY_DIR"
rm -rf "$PY_DIR"
mkdir -p "$PY_DIR"
# Copy the install/ tree contents into Resources/python/
cp -R "$WORK/python/install/." "$PY_DIR/"

NEW_PY="$PY_DIR/bin/python3.12"
if [ ! -x "$NEW_PY" ]; then
    echo "ERROR: $NEW_PY not found after extract"
    ls "$PY_DIR/bin"
    exit 1
fi
echo "    new: $($NEW_PY --version)"

echo "==> Re-creating MacOS/tellar-py hardlink"
rm -f "$APP/Contents/MacOS/tellar-py"
ln "$NEW_PY" "$APP/Contents/MacOS/tellar-py"

echo "==> Done"
echo "    Run: open ~/tellar/Tellar.app"
echo "    Or:  ~/tellar/Tellar.app/Contents/MacOS/tellar"
