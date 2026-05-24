#!/bin/bash
# package.sh — wrap latest Tellar.app build into a polished, versioned .dmg
# for distribution. Produces Tellar-<version>.dmg in repo root.
#
# What the recipient sees:
#   1. Double-clicks the .dmg → macOS mounts it
#   2. Finder window opens with Tellar.app on the left, an Applications
#      folder shortcut on the right, and a "Drag to Applications"
#      background. No toolbar, no sidebar — clearly an install dialog.
#   3. Drags Tellar onto Applications → installed. Closes window, ejects.
#
# Run after ./build.sh has produced a working Tellar.app. The script
# refuses to package a missing or stale-looking bundle.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/Tellar.app"
PLIST="$APP/Contents/Info.plist"
BG_SRC="$ROOT/assets/dmg-background.png"

if [ ! -d "$APP" ]; then
    echo "ERROR: $APP not found — run ./build.sh first"
    exit 1
fi
if [ ! -x "$APP/Contents/MacOS/tellar" ]; then
    echo "ERROR: $APP/Contents/MacOS/tellar missing — bundle looks broken, rebuild via ./build.sh"
    exit 1
fi
if [ ! -f "$BG_SRC" ]; then
    echo "ERROR: $BG_SRC not found — regenerate via the Pillow script in assets/"
    exit 1
fi

VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$PLIST")
NAME="Tellar"
DMG="$ROOT/${NAME}-${VERSION}.dmg"
STAGING="$ROOT/build-cache/dmg-staging"
TEMP_DMG="$ROOT/build-cache/dmg-temp.dmg"
MOUNT_POINT="/Volumes/$NAME"

echo "==> Packaging $NAME $VERSION"
echo "    target: $DMG"

# Defensive: if a previous run left things mounted or staged, clean up.
if [ -d "$MOUNT_POINT" ]; then
    echo "==> Detaching stale mount $MOUNT_POINT"
    hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
fi
rm -rf "$STAGING"
rm -f "$TEMP_DMG"
mkdir -p "$STAGING"

# --- 1. Stage contents that should appear in the mounted DMG ---
echo "==> Staging contents"
cp -R "$APP" "$STAGING/"
# Symlink, not copy — appears in Finder as the Applications folder, drag
# onto it copies the .app into the real /Applications.
ln -s /Applications "$STAGING/Applications"
# Hidden background folder — referenced by AppleScript below.
mkdir "$STAGING/.background"
cp "$BG_SRC" "$STAGING/.background/background.png"

# --- 2. Create a read-write DMG from the staging dir ---
# UDRW is read-write so AppleScript can manipulate Finder window state
# (icon positions, background, size). We'll convert to read-only UDZO at
# the end for distribution.
echo "==> Building writable DMG"
hdiutil create \
    -volname "$NAME" \
    -srcfolder "$STAGING" \
    -format UDRW \
    -ov \
    "$TEMP_DMG" >/dev/null

# --- 3. Mount it ---
echo "==> Mounting at $MOUNT_POINT"
hdiutil attach "$TEMP_DMG" -mountpoint "$MOUNT_POINT" -nobrowse >/dev/null

# --- 4. Configure Finder window via AppleScript ---
# Triggers a one-time "Tellar wants to control Finder" prompt on the dev
# machine. Recipients don't see it — by the time they get the DMG, the
# window layout is already baked in.
echo "==> Configuring Finder window layout"
osascript <<APPLESCRIPT
tell application "Finder"
    tell disk "$NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {400, 200, 1000, 600}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 128
        set background picture of viewOptions to file ".background:background.png"
        set position of item "${NAME}.app" of container window to {150, 200}
        set position of item "Applications" of container window to {450, 200}
        update without registering applications
        delay 1
        close
    end tell
end tell
APPLESCRIPT

# Brief sync so Finder writes its .DS_Store metadata onto the DMG.
sync
sleep 1

# --- 5. Unmount ---
echo "==> Detaching"
hdiutil detach "$MOUNT_POINT" >/dev/null

# --- 6. Convert RW → compressed read-only ---
echo "==> Compressing to UDZO"
rm -f "$DMG"
hdiutil convert "$TEMP_DMG" \
    -format UDZO \
    -imagekey zlib-level=9 \
    -o "$DMG" >/dev/null

# --- 7. Cleanup ---
rm -f "$TEMP_DMG"
rm -rf "$STAGING"

SIZE=$(du -h "$DMG" | cut -f1)
echo
echo "==> DMG ready"
echo "    $DMG ($SIZE)"
echo "    Test:  open $DMG"
