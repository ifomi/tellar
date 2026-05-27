# Tellar

Local push-to-talk voice dictation for Apple Silicon Mac.

Hold `⌃Space`, talk, release — your words appear in whatever app you have focused. Runs entirely on your Mac via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). No cloud, no telemetry, no network calls after the model is downloaded once.

## Requirements

- Apple Silicon Mac (M1 or newer — uses Metal GPU)
- macOS 13.0 or later
- ~600 MB free disk for the app, ~1.5 GB more for the speech model on first run
- Internet connection on first launch (to download the model)

## Install

1. Download `Tellar-VERSION.dmg` from the [Releases](../../releases) page.
2. Open the DMG, drag **Tellar** into **Applications**, eject the DMG.
3. **Bypass Gatekeeper:** open Terminal and run
   ```sh
   xattr -dr com.apple.quarantine /Applications/Tellar.app
   ```
4. Double-click **Tellar** in `/Applications`.

### Why step 3 is required

Tellar is signed ad-hoc, not with a paid Apple Developer ID. On macOS 15 Sequoia / 26 Tahoe, ad-hoc signed apps with the `com.apple.quarantine` attribute (which any download from the Internet gets) are blocked by Gatekeeper outright — there is no right-click → Open shortcut and no "Open Anyway" button in System Settings the way older macOS used to offer.

The `xattr` command removes that one attribute. After it, macOS treats Tellar as an app you placed there yourself and lets it run.

If you don't want to run Terminal commands to install software, please wait for a notarized release.

### First launch

On first launch, Tellar will:

1. Download the Whisper Large v3 Turbo model from Hugging Face (~1.5 GB, public — no account needed). Progress shown in an overlay.
2. Ask for **Microphone** permission. Click *Allow*.
3. Ask for **Input Monitoring** permission (so it can hear `⌃Space` globally). Open System Settings → Privacy & Security → Input Monitoring → enable Tellar.
4. Ask for **Accessibility** permission (so it can paste into the focused app). Same path, Privacy & Security → Accessibility → enable Tellar. Tellar will restart itself once you grant this — `AXIsProcessTrusted()` caches per process, so a fresh launch is the only way to refresh.

## Usage

- **`⌃Space`** — start / stop recording
- **`⌃Esc`** — cancel current recording
- The transcribed text gets pasted into whatever app is in focus. Toggle **Auto Paste** off (menu bar dropdown) to instead copy to clipboard without pasting.
- The menu bar icon is a waveform. Click for the menu, hold the recording hotkey to see a live timer in place of the icon.

There is no Dock icon by design — Tellar is a menu-bar accessory.

## Build from source

If you want to **modify** Tellar (fork, fix a bug, change the model, change the hotkey) or **verify** the binary by reproducing it yourself instead of trusting the released DMG, build it locally. End users do **not** need this — `Install` above is enough.

```sh
git clone https://github.com/ifomi/tellar.git
cd tellar

# 1. System dependency: portaudio (pyaudio links against it; the build script
# bundles it into the .app so recipients don't need brew themselves).
brew install portaudio

# 2. Embedded Python runtime. Download a python-build-standalone release into
# build-cache/. Pick a 3.12 build, aarch64-apple-darwin, "pgo+lto-full" flavor.
# Releases live at https://github.com/astral-sh/python-build-standalone/releases
mkdir -p build-cache
# example (replace with current release tag):
#   curl -L -o build-cache/cpython-3.12.13+20260510-aarch64-apple-darwin-pgo+lto-full.tar.zst \
#     https://github.com/astral-sh/python-build-standalone/releases/download/20260510/cpython-3.12.13+20260510-aarch64-apple-darwin-pgo+lto-full.tar.zst

# 3. Build the .app bundle (~5-10 min on first run, mostly pip install)
./build.sh    # produces Tellar.app (~600 MB)

# 4. Wrap into a distributable DMG
./package.sh  # produces Tellar-VERSION.dmg
```

`build.sh` is idempotent — rerun whenever `launcher.c`, `tellar/`, or pinned dep versions change. Output is fully self-contained: the resulting `Tellar.app` runs on any Apple Silicon Mac without Homebrew, system Python, or ffmpeg installed.

## How it works

- `launcher.c` — native Mach-O binary at `Contents/MacOS/tellar`. Embeds CPython via `Py_Initialize()` and runs `runpy.run_module('tellar.app')` in-process. Not a shell wrapper, not a subprocess fork.
- `Contents/Resources/python/` — python-build-standalone interpreter + stdlib, with unused chunks pruned.
- `Contents/Resources/app/site-packages/` — pip-installed deps. PyQt6 for overlay and dialogs, AppKit (via PyObjC) for the menu bar status item, mlx-whisper for transcription, pyaudio for capture, libportaudio bundled in.
- The bundle is fully self-contained. It does not require Homebrew, system Python, ffmpeg, or anything else on the recipient's Mac.

Pipeline:

```
PyAudio capture
  → 16 kHz mono int16 WAV in memory (numpy)
  → mlx_whisper.transcribe on Metal GPU
  → keystroke paste via CGEventPost
```

## Project layout

```
launcher.c            Native launcher source
build.sh              Bundle assembly (extract Python, pip install,
                      sign embedded dylibs, sign bundle, audit)
package.sh            DMG packaging (drag-to-Applications layout,
                      compressed UDZO image)
tellar/               Python package — entry point, recording, UI,
                      menu bar, transcription, paste, permissions UX
patches/              Source patches applied at build time
                      (mlx_whisper lazy timing import)
assets/               App icon (PNG → ICNS) and DMG background
Info.plist            Bundle metadata (LSUIElement, usage descriptions)
pyproject.toml        Python package metadata
```

## Known limitations

- **Apple Silicon only.** mlx is Metal-based; there is no Intel fallback.
- **Ad-hoc signed.** Requires the `xattr` step on first install (see above). Notarization is on the roadmap.
- **Single instance.** A second launch attempt is rejected by an `flock`-based lock; the first instance keeps running.
- **English and Russian models.** The bilingual `initial_prompt` in `tellar/transcriber.py` biases the decoder toward proper punctuation in those two languages. Other languages still transcribe but may need their own prompt.

## License

[MIT](LICENSE) © 2026 Ivan Fomichev
