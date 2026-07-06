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
# Stale _CodeSignature is worse than no signature: macOS sees the manifest
# and tries to validate hashes against current Resources/, fails, and refuses
# to launch the app with no override option. Wipe it; we re-sign in step 9.
echo "==> Cleaning previous build"
rm -rf "$PY_DIR" "$APP_DIR/tellar" "$SITE" "$APP/Contents/_CodeSignature"
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

# Bundle the P&C model (int8, ~266 MB) inside the package. Every multi-chunk
# dictation uses it and it is NOT on HuggingFace (locally quantized), so we
# ship it rather than host + download. Sourced from the build machine's HF cache.
echo "==> Bundling P&C model"
PNC_ONNX="$(find "$HOME/.cache/huggingface" -name model.int8.onnx -path '*xlm-roberta_punct*' 2>/dev/null | head -1)"
PNC_SP="$(find "$HOME/.cache/huggingface" -name sp.model -path '*xlm-roberta_punct*' 2>/dev/null | head -1)"
if [ -z "$PNC_ONNX" ] || [ -z "$PNC_SP" ]; then
    echo "    ERROR: P&C model not in HF cache (model.int8.onnx / sp.model)."
    echo "    Quantize it first: tools/pnc_quantize_test.py"
    exit 1
fi
mkdir -p "$APP_DIR/tellar/assets/pnc"
cp "$PNC_ONNX" "$PNC_SP" "$APP_DIR/tellar/assets/pnc/"
echo "    bundled P&C model ($(du -h "$PNC_ONNX" | cut -f1))"

# Release safety: force DIAGNOSTIC_MODE off in the SHIPPED copy (source stays
# True for dev) — never ship a build that logs transcripts or shows the dev UI.
sed -i '' 's/^DIAGNOSTIC_MODE = True/DIAGNOSTIC_MODE = False/' "$APP_DIR/tellar/app.py"
grep -q '^DIAGNOSTIC_MODE = False' "$APP_DIR/tellar/app.py" \
    && echo "    DIAGNOSTIC_MODE forced off in bundle" \
    || { echo "    ERROR: failed to disable DIAGNOSTIC_MODE"; exit 1; }

# --- 4b. Copy app icon ---
# Info.plist references CFBundleIconFile=icon, so macOS expects icon.icns
# in Resources/. The pre-built .icns lives in assets/ (regenerable from
# assets/icon.png via the iconset/iconutil pipeline if the source changes).
if [ -f "$ROOT/assets/icon.icns" ]; then
    cp "$ROOT/assets/icon.icns" "$RES/icon.icns"
    echo "    icon: $RES/icon.icns"
fi

# --- 5. Install runtime deps with embedded Python ---
# pip on this machine cannot reach pypi.org directly (Apple corporate proxy
# returns 403). Apple's internal mirror at pypi.apple.com proxies the public
# index transparently. Pinning --index-url here makes the build work on
# Apple network without depending on env/global pip config.
echo "==> Installing pip dependencies into $SITE"
PIP_INDEX="https://pypi.apple.com/simple/"
"$EMBEDDED_PY" -m pip install --index-url "$PIP_INDEX" --upgrade pip wheel
"$EMBEDDED_PY" -m pip install --index-url "$PIP_INDEX" --target "$SITE" \
    "mlx-whisper==0.4.3" \
    "mlx==0.31.2" \
    "mlx-lm==0.31.3" \
    "transformers==5.9.0" \
    "sentencepiece==0.2.1" \
    "numpy<2.1" \
    "huggingface-hub>=0.20" \
    "PyQt6>=6.6" \
    "pyobjc-framework-Cocoa>=10.0" \
    "pyobjc-framework-Quartz>=10.0" \
    "pyobjc-framework-ApplicationServices>=10.0" \
    "pyaudio>=0.2" \
    "onnxruntime>=1.20"

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

# Remove unused PyQt6 plugins that link to external system libraries the
# recipient won't have. Specifically the SQL drivers — libqsqlmimer needs
# libmimerapi, libqsqlodbc needs libiodbc, libqsqlpsql needs libpq from
# Postgres.app. Tellar never imports QtSql, so these are dead code that
# would otherwise trip the external-deps audit below.
echo "==> Removing unused PyQt6 plugins"
rm -rf "$SITE/PyQt6/Qt6/plugins/sqldrivers"

# Prune onnxruntime sub-packages we never import. We use only the
# inference session for silero VAD (chunking_vad.py → vad.py:
# onnxruntime.InferenceSession + SessionOptions). The shipping wheel
# also contains tooling packages (transformers helpers, quantization
# utilities, conversion tools, sample datasets, generic backend
# bindings) — none touched on Tellar's runtime path. Removing saves
# ~4 MB and avoids carrying code we don't ship behaviourally.
echo "==> Pruning unused onnxruntime sub-packages"
for sub in transformers quantization tools datasets backend; do
    rm -rf "$SITE/onnxruntime/$sub" 2>/dev/null || true
done

# --- 6c. Bundle external dylibs that pip-installed C extensions reference ---
# Some Python C extensions (notably pyaudio's _portaudio.so) are compiled
# against system libraries installed via Homebrew. The compiled .so embeds
# the absolute path of the build-machine's library as a load command. On a
# fresh recipient machine without that exact Homebrew layout, dlopen fails
# with "Library not loaded: /opt/homebrew/...". The bundle isn't actually
# self-contained until we copy those dylibs in and rewrite the load commands.
#
# Audit pattern: any .so or .dylib in Resources/ whose otool -L lists a path
# starting with /opt/homebrew/, /usr/local/, /Users/, or /Applications/ —
# i.e. a non-system, non-bundle absolute path the recipient won't have.
#
# Currently handled:
#   pyaudio/_portaudio.so → /opt/homebrew/.../libportaudio.2.dylib
#
# Not handled (because Tellar doesn't import them; they ship dead code only):
#   PyQt6/Qt6/plugins/sqldrivers/{libqsqlmimer, libqsqlodbc, libqsqlpsql}.dylib
#
# If a new external dep appears in the future, the audit at the bottom will
# fail loudly so we know to add it explicitly here.
echo "==> Bundling external dylibs"
PYAUDIO_DIR="$SITE/pyaudio"
PORTAUDIO_SO="$PYAUDIO_DIR/_portaudio.cpython-312-darwin.so"
if [ -f "$PORTAUDIO_SO" ]; then
    PORTAUDIO_SRC=/opt/homebrew/opt/portaudio/lib/libportaudio.2.dylib
    if [ ! -f "$PORTAUDIO_SRC" ]; then
        echo "ERROR: $PORTAUDIO_SRC missing. brew install portaudio first."
        exit 1
    fi
    PORTAUDIO_DST="$PYAUDIO_DIR/libportaudio.2.dylib"
    cp "$PORTAUDIO_SRC" "$PORTAUDIO_DST"
    # Homebrew installs portaudio as r--r--r-- (read-only owner). cp preserves
    # that, which later breaks `xattr -dr com.apple.quarantine` on the
    # recipient's side — xattr needs write access to update metadata. Add
    # owner write permission so quarantine cleanup works in the install path.
    chmod u+w "$PORTAUDIO_DST"
    install_name_tool -id @loader_path/libportaudio.2.dylib "$PORTAUDIO_DST" 2>&1 | grep -v "^/.*install_name_tool: warning" || true
    install_name_tool -change "$PORTAUDIO_SRC" @loader_path/libportaudio.2.dylib "$PORTAUDIO_SO" 2>&1 | grep -v "^/.*install_name_tool: warning" || true
    echo "    libportaudio.2.dylib bundled into pyaudio/"
fi

# Audit: refuse to ship a bundle that still references external paths.
# Better to fail the build than ship a non-portable .app.
#
# Note on the `|| true` after grep: with `set -euo pipefail`, a grep that
# finds zero matches exits 1 and tears down the whole pipeline silently
# (no error message, just an early script exit). We *want* zero matches —
# that's the success case. `|| true` neutralises that exit.
echo "==> Auditing remaining external dylib references"
EXTERNAL=$(find "$RES" \( -name "*.so" -o -name "*.dylib" \) -type f -print0 | \
    xargs -0 -I {} otool -L {} 2>/dev/null | \
    { grep -E "^\s+(/opt/homebrew|/usr/local|/Users|/Applications)" || true; } | \
    awk '{print $1}' | sort -u)
if [ -n "$EXTERNAL" ]; then
    echo "ERROR: Bundle still references external (non-system) dylibs:"
    echo "$EXTERNAL" | sed 's/^/    /'
    echo "    Either bundle them in step 6c, or delete the consuming package."
    exit 1
fi
echo "    clean"

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

# --- 9. Sign the bundle ad-hoc ---
# Pip and tar drop fresh dylibs into Resources/ on every build, so the bundle
# must be re-signed end-to-end. Without this, Gatekeeper rejects the app on
# the recipient's machine ("can't be opened", no override).
#
# Two non-obvious gotchas:
#
# 1. NO --options runtime. Hardened runtime enables library validation,
#    which refuses to load any dylib whose Team ID does not match the main
#    executable. Our launcher is ad-hoc (no Team ID); libpython3.12.dylib
#    from python-build-standalone has Apple's Team ID. Result: dyld blocks
#    the load with "different Team IDs". Hardened runtime is only useful
#    if we're going to notarize (Apple Developer ID + notarytool) — for
#    ad-hoc personal distribution it just breaks things.
#
# 2. Explicit sign every dylib/.so under Resources/ FIRST, then the bundle.
#    `codesign --deep` only descends into Contents/Frameworks, PlugIns, and
#    MacOS — it skips Resources/. Embedded Python lives in Resources/python,
#    its C extensions in Resources/app/site-packages. Without per-file
#    signing, those keep their original (or no) signatures and library
#    validation rejects them even with hardened runtime off, on stricter
#    systems.
#
# Ad-hoc (-) is enough for personal distribution; the recipient still
# right-clicks → Open or uses System Settings → Privacy & Security → Open
# Anyway the first time because the DMG carries com.apple.quarantine.
echo "==> Signing embedded dylibs/so files"
find "$RES" \( -name "*.dylib" -o -name "*.so" \) -type f -print0 | \
    xargs -0 -n 50 codesign --force --sign - 2>&1 | grep -v "replacing existing signature" || true

echo "==> Signing bundle (ad-hoc, no hardened runtime)"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP" && echo "    signature OK"

BUNDLE_SIZE=$(du -sh "$APP" | cut -f1)
echo
echo "==> Build complete"
echo "    Bundle size: $BUNDLE_SIZE"
echo "    Run: open $APP"
echo "    Or:  $APP/Contents/MacOS/tellar    (with stdout/stderr to terminal)"
