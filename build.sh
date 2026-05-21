#!/bin/bash
# build.sh — assemble Tellar.app from source + python-build-standalone tarball + pip deps.
# Idempotent: safe to rerun. Each run cleans Resources/python and Resources/app/site-packages
# and recompiles the native launcher from launcher.c.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/Tellar.app"
RES="$APP/Contents/Resources"
MACOS="$APP/Contents/MacOS"
PY_DIR="$RES/python"
APP_DIR="$RES/app"
SITE="$APP_DIR/site-packages"
CACHE="$ROOT/build-cache"

# We need the full python-build-standalone build, not install_only — the full
# archive ships headers and libpython3.12.dylib that the C launcher links against.
PY_TARBALL_GLOB="$CACHE/cpython-3.12*-aarch64-apple-darwin-pgo+lto-full*.tar.zst"

echo "==> Tellar build"
echo "    APP=$APP"

# --- 1. Verify python-build-standalone tarball ---
TARBALL="$(ls -1 $PY_TARBALL_GLOB 2>/dev/null | head -1 || true)"
if [ -z "$TARBALL" ]; then
    echo "ERROR: no python-build-standalone tarball found in $CACHE/"
    echo "       expected pattern: cpython-3.12*-aarch64-apple-darwin-pgo+lto-full*.tar.zst"
    echo "       download from https://github.com/astral-sh/python-build-standalone/releases"
    exit 1
fi
echo "    using $TARBALL"

# --- 2. Clean previous build artifacts ---
echo "==> Cleaning previous build"
rm -rf "$PY_DIR" "$APP_DIR/tellar" "$SITE"
rm -f "$MACOS/tellar" "$MACOS/tellar-py"
mkdir -p "$PY_DIR" "$APP_DIR" "$SITE" "$MACOS"

# --- 3. Extract embedded Python (zstd-compressed tarball) ---
echo "==> Extracting Python runtime"
# pgo+lto-full layout: tarball has python/{PYTHON.json,build,install,licenses}.
# The runnable interpreter and its libs/headers live under python/install/.
# We extract to a temp dir, then copy only install/ contents into PY_DIR —
# build/ and licenses/ are CPython source-build artifacts we don't need.
# BSD tar on macOS 13+ understands --zstd natively.
WORK="$(mktemp -d /tmp/tellar-pybs-XXXXXX)"
tar --zstd -xf "$TARBALL" -C "$WORK"
if [ ! -d "$WORK/python/install" ]; then
    echo "ERROR: unexpected tarball layout — no python/install/ dir"
    rm -rf "$WORK"
    exit 1
fi
cp -R "$WORK/python/install/." "$PY_DIR/"
rm -rf "$WORK"
EMBEDDED_PY="$PY_DIR/bin/python3.12"
if [ ! -x "$EMBEDDED_PY" ]; then
    echo "ERROR: embedded python not found at $EMBEDDED_PY after extraction"
    exit 1
fi
echo "    embedded Python: $($EMBEDDED_PY --version)"

# --- 4. Copy Tellar source ---
echo "==> Copying tellar package"
cp -R "$ROOT/tellar" "$APP_DIR/tellar"
find "$APP_DIR/tellar" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$APP_DIR/tellar" -name "*.pyc" -delete

# --- 5. Install runtime deps with embedded Python ---
echo "==> Installing pip dependencies into $SITE"
"$EMBEDDED_PY" -m pip install --upgrade pip wheel
"$EMBEDDED_PY" -m pip install --target "$SITE" \
    "mlx-whisper>=0.4" \
    "mlx>=0.21" \
    "huggingface-hub>=0.20" \
    "PyQt6>=6.6" \
    "pyobjc-framework-Cocoa>=10.0" \
    "pyobjc-framework-Quartz>=10.0" \
    "pyaudio>=0.2" \
    "numpy>=1.24"

# --- 5b. Patch mlx_whisper for lazy timing import ---
# transcribe.py eagerly imports .timing at module load, which pulls in scipy
# and numba (and through numba, llvmlite — ~190 MB combined). add_word_timestamps
# is the only consumer and runs only when word_timestamps=True; we never set
# that flag. Move the import inside the conditional so timing.py is loaded
# lazily — then we can drop scipy/numba/llvmlite below.
echo "==> Patching mlx_whisper (lazy timing import)"
( cd "$SITE" && patch -p1 < "$ROOT/patches/mlx-whisper-lazy-timing.patch" )

# --- 6. Strip caches and unused locale data to reduce bundle size ---
echo "==> Stripping caches"
find "$SITE" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$SITE" -name "*.pyc" -delete
find "$SITE" -name "tests" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# --- 6b. Prune unused transitive dependencies ---
# mlx_whisper declares torch as a runtime dep, but the only file that imports
# it (torch_whisper.py) is a reference implementation; mlx_whisper/__init__.py
# does not pull it in, and our transcribe() path is pure-MLX. Removing torch
# saves ~390 MB; sympy/networkx are torch-only deps and follow it out.
# torchgen / functorch are torch's own subpackages installed separately.
#
# scipy/numba/llvmlite are pulled in by mlx_whisper.timing (word-level
# timestamps). The patch in step 5b makes that import lazy, so we never
# load these modules and can drop them from disk too.
echo "==> Pruning unused transitive deps"
for pkg in torch torchgen functorch sympy networkx scipy numba llvmlite; do
    find "$SITE" -maxdepth 1 \
        \( -name "$pkg" -o -name "${pkg}-*.dist-info" \) \
        -exec rm -rf {} + 2>/dev/null || true
done

# --- 7. Compile native launcher ---
# launcher.c embeds CPython via Py_Initialize() so the binary at MacOS/tellar
# is the actual process — not a bash script that exec()s python3.12. This is
# what makes SystemUIServer accept our NSStatusItem and TCC bind permissions
# to Tellar.app rather than to a generic interpreter.
echo "==> Compiling launcher.c"
clang "$ROOT/launcher.c" \
    -I "$PY_DIR/include/python3.12" \
    -L "$PY_DIR/lib" \
    -lpython3.12 \
    -Wl,-rpath,@executable_path/../Resources/python/lib \
    -o "$MACOS/tellar"
echo "    launcher: $(file "$MACOS/tellar" | sed 's|^[^:]*: ||')"

# --- 8. Touch .app to invalidate Launch Services cache ---
touch "$APP"

BUNDLE_SIZE=$(du -sh "$APP" | cut -f1)
echo
echo "==> Build complete"
echo "    Bundle size: $BUNDLE_SIZE"
echo "    Run: open $APP"
echo "    Or:  $APP/Contents/MacOS/tellar    (with stdout/stderr to terminal)"
