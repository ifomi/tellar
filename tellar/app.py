import sys
import os
import json
import fcntl
import threading
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF, QPoint
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QPen

from typing import Optional

from .recorder import Recorder
from .transcriber import transcribe_audio, model_exists, get_model, MODEL_DIR, MODEL_NAME
from .inserter import insert_text, get_frontmost_app
from .logging_setup import setup_logging, get_logger
from .menubar import MenuBarIcon

log = get_logger(__name__)

WAVEFORM_BARS = 5
MAX_AMPLITUDE = 1000.0
STATE_DIR = Path.home() / "Library" / "Application Support" / "Tellar"
POSITION_FILE = STATE_DIR / "overlay_position.json"


# Path to the brand silhouette used as the menubar icon. The PNG is a
# pre-rendered alpha mask (transparent background, opaque black where the
# Tellar t+wave shape is), generated from assets/icon.png — see the
# regenerate-menubar-silhouette script in repo docs. Living inside the
# Python package means build.sh's `cp -R "$ROOT/tellar"` already ships it.
_SILHOUETTE_PATH = Path(__file__).parent / "menubar_silhouette.png"
_silhouette_master: Optional[QPixmap] = None


def _make_wave_pixmap(color: str) -> QPixmap:
    """Return a 22pt-tall menubar pixmap of the Tellar silhouette tinted in `color`.

    Generated at exact retina pixel size (44px tall, width following the
    silhouette's natural aspect). No devicePixelRatio — qpixmap_to_nsimage
    in menubar.py maps pixel size → logical point size by dividing by 2.
    Earlier DPR-based approach produced half-pixel logical widths (30.5pt)
    which confused NSStatusBarButton sizing.
    """
    global _silhouette_master
    if _silhouette_master is None:
        _silhouette_master = QPixmap(str(_SILHOUETTE_PATH))
    src_w, src_h = _silhouette_master.width(), _silhouette_master.height()
    px_h = 44  # 22pt logical at retina; AppKit downscales for 1x displays
    px_w = max(px_h, int(round(src_w * px_h / src_h)))
    # Make sure px_w is even — avoids half-pixel issues when divided by 2
    # to derive logical width in qpixmap_to_nsimage.
    if px_w % 2:
        px_w += 1
    scaled = _silhouette_master.scaled(
        px_w, px_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    pm = QPixmap(px_w, px_h)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    # Centre horizontally in case scaled width came back smaller than px_w
    # (KeepAspectRatio rounding).
    p.drawPixmap((px_w - scaled.width()) // 2, 0, scaled)
    # Replace the silhouette's black with the requested colour.
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(pm.rect(), QColor(color))
    p.end()
    return pm


def _make_wave_icon(color: str) -> QIcon:
    return QIcon(_make_wave_pixmap(color))


def _make_timer_icon(text: str) -> QIcon:
    scale = 2
    w, h = 44 * scale, 22 * scale
    pm = QPixmap(w, h)
    pm.setDevicePixelRatio(scale)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setFont(QFont("Menlo", 18))
    p.setPen(QColor("#cc0000"))
    p.drawText(QRectF(0, 0, 44, 22), Qt.AlignmentFlag.AlignCenter, text)
    p.end()
    return QIcon(pm)


def _open_accessibility_settings():
    """Jump directly to System Settings → Privacy & Security → Accessibility."""
    import subprocess
    subprocess.Popen([
        "open",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    ])


def _open_input_monitoring_settings():
    """Jump directly to System Settings → Privacy & Security → Input Monitoring."""
    import subprocess
    subprocess.Popen([
        "open",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    ])


def _check_permissions():
    """Return (accessibility_ok, input_monitoring_ok).

    Tellar needs both:
    - Input Monitoring lets CGEventTap listen for global Ctrl+Space.
      Required since macOS 10.15. Without it, the event tap silently fails.
    - Accessibility lets us synthesize Cmd+V (CGEventPost) to paste the
      transcribed text into the active app. Without it, text reaches the
      clipboard but the user has to ⌘V manually.

    Both bind to the cdhash of MacOS/tellar — recompiling the launcher
    invalidates both. APIs:
    - AXIsProcessTrusted()              — synchronous bool, from
                                          pyobjc-framework-ApplicationServices
    - CGPreflightListenEventAccess()    — synchronous bool, from Quartz
    Both are non-prompting; they just report the current grant state.
    """
    try:
        from ApplicationServices import AXIsProcessTrusted
        acc_ok = bool(AXIsProcessTrusted())
    except Exception:
        log.exception("AXIsProcessTrusted unavailable; assuming Accessibility granted")
        acc_ok = True
    try:
        import Quartz
        im_ok = bool(Quartz.CGPreflightListenEventAccess())
    except Exception:
        log.exception("CGPreflightListenEventAccess unavailable; assuming Input Monitoring granted")
        im_ok = True
    return acc_ok, im_ok


class Bridge(QObject):
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    recording_cancelled = pyqtSignal()
    transcription_done = pyqtSignal(str)
    transcription_empty = pyqtSignal()
    model_ready = pyqtSignal()
    model_error = pyqtSignal(str)
    # Phase 4: download model on first launch (or after cache wipe).
    # Progress: (percent, mb_done, mb_total). Finished: switch overlay to
    # "Loading model..." while mlx_whisper.load_model runs.
    model_download_progress = pyqtSignal(int, int, int)
    model_download_finished = pyqtSignal()
    # Phase 5: Accessibility and/or Input Monitoring missing. Args are
    # (accessibility_ok, input_monitoring_ok) — true means the permission
    # IS granted. Without these, the global hotkey doesn't fire and/or
    # auto-paste doesn't work; the overlay tells the user which to grant.
    permissions_needed = pyqtSignal(bool, bool)


class OverlayWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Tool windows on macOS hide when their owning app becomes inactive
        # (e.g. user clicks away during a long model download). This attribute
        # keeps the overlay visible across focus changes — critical for the
        # download-progress UI which the user needs to monitor while working
        # in other apps.
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        # Receive mouseMove events even when no button is pressed — needed
        # for the Phase 5 hover-cursor logic over the "click to open Settings"
        # substring (so the pointer cursor only appears over that text, not
        # over the rest of the overlay which is for dragging).
        self.setMouseTracking(True)
        self.setFixedSize(150, 120)
        self._bars = [0.0] * WAVEFORM_BARS
        self._mode = "idle"
        self._elapsed = ""
        self._status_text = ""
        self._substatus_text = ""
        self._dl_pct = 0
        # Phase 5: per-permission state for the permissions_needed mode.
        # True = granted, False = missing.
        self._perm_acc = True
        self._perm_im = True
        self._spin_frame = 0
        self._spin_timer = QTimer()
        self._spin_timer.setInterval(200)
        self._spin_timer.timeout.connect(self._spin_tick)
        self._drag_pos: Optional[QPoint] = None
        # Track press position separately from drag-offset so we can
        # distinguish a click (no movement) from a drag in mouseRelease.
        # Used by the permission_needed mode to open System Settings on
        # click without breaking the existing drag-to-reposition gesture.
        self._press_screen_pos: Optional[QPoint] = None
        self._saved_position: Optional[QPoint] = self._load_position()

    def _spin_tick(self):
        self._spin_frame = (self._spin_frame + 1) % 3
        self.update()

    def _load_position(self) -> Optional[QPoint]:
        try:
            data = json.loads(POSITION_FILE.read_text())
            return QPoint(data["x"], data["y"])
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None

    def _save_position(self):
        pos = self.pos()
        POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)
        POSITION_FILE.write_text(json.dumps({"x": pos.x(), "y": pos.y()}))
        self._saved_position = pos

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_screen_pos = event.globalPosition().toPoint()
            self._drag_pos = self._press_screen_pos - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            return
        # No button pressed: update cursor based on what's under the pointer.
        # In permissions_needed mode, the substring "click to open Settings"
        # is the clickable target — pointer cursor only there. The paint
        # event draws that substring inside QRectF(0, height-28, width, 20).
        if self._mode == "permissions_needed":
            y = event.position().y()
            if self.height() - 28 <= y <= self.height() - 8:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.unsetCursor()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._drag_pos is None:
            return
        # If the mouse barely moved, treat as a click rather than a drag —
        # don't persist a position change, and let mode-specific handlers
        # react. The 5px threshold matches macOS's click-vs-drag heuristic.
        moved = (event.globalPosition().toPoint() - self._press_screen_pos).manhattanLength() if self._press_screen_pos else 0
        self._drag_pos = None
        self._press_screen_pos = None
        if moved >= 5:
            self._save_position()
            return
        if self._mode == "permissions_needed":
            # Open the pane for the first missing permission. After the user
            # grants and relaunches, if the other one is still missing the
            # next launch will open that pane. One step at a time.
            if not self._perm_im:
                _open_input_monitoring_settings()
            elif not self._perm_acc:
                _open_accessibility_settings()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(0, 0, 0, 200))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 16, 16)

        if self._mode == "recording":
            # Timer top
            p.setFont(QFont("Menlo", 13))
            p.setPen(QColor(255, 255, 255, 180))
            p.drawText(QRectF(0, 8, self.width(), 20), Qt.AlignmentFlag.AlignCenter, self._elapsed)

            # Waveform center
            p.setPen(Qt.PenStyle.NoPen)
            cx = self.width() // 2
            bar_w = 8
            gap = 8
            total_w = WAVEFORM_BARS * bar_w + (WAVEFORM_BARS - 1) * gap
            start_x = cx - total_w // 2
            y_center = 60
            for i, amp in enumerate(self._bars):
                h = max(10, int(amp * 44))
                x = start_x + i * (bar_w + gap)
                p.setBrush(QColor(255, 255, 255, 230))
                p.drawRoundedRect(x, y_center - h // 2, bar_w, h, 4, 4)

            # Hint bottom
            p.setFont(QFont("Helvetica Neue", 9))
            p.setPen(QColor(255, 255, 255, 100))
            p.drawText(QRectF(0, 98, self.width(), 16), Qt.AlignmentFlag.AlignCenter, "⌃Space · ⌃Esc")

        elif self._mode in ("loading", "transcribing", "done", "error", "downloading", "permissions_needed"):
            p.setFont(QFont("Helvetica Neue", 13, QFont.Weight.DemiBold))
            if self._mode in ("error", "permissions_needed"):
                p.setPen(QColor(255, 100, 100, 240))
            else:
                p.setPen(QColor(255, 255, 255, 240))
            text_y = 35 if self._mode == "transcribing" else 40
            if self._mode == "permissions_needed":
                # Title at top, then a 2-line checklist for the permissions,
                # then "click to fix" hint at the bottom.
                p.setFont(QFont("Helvetica Neue", 11, QFont.Weight.DemiBold))
                p.drawText(QRectF(0, 8, self.width(), 18), Qt.AlignmentFlag.AlignCenter, "Permissions required")
                p.setFont(QFont("Helvetica Neue", 10))
                line_y = 32
                for label, ok in (("Input Monitoring", self._perm_im), ("Accessibility", self._perm_acc)):
                    mark = "✓" if ok else "✗"
                    p.setPen(QColor(120, 220, 140, 230) if ok else QColor(255, 100, 100, 230))
                    p.drawText(QRectF(10, line_y, 16, 16), Qt.AlignmentFlag.AlignCenter, mark)
                    p.setPen(QColor(255, 255, 255, 220) if ok else QColor(255, 200, 200, 230))
                    p.drawText(QRectF(28, line_y, self.width() - 32, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
                    line_y += 20
            else:
                p.drawText(QRectF(0, text_y, self.width(), 20), Qt.AlignmentFlag.AlignCenter, self._status_text)

            if self._mode == "transcribing":
                cx = self.width() // 2
                dot_r = 5
                gap = 16
                y = 72
                for i in range(3):
                    alpha = 240 if i == self._spin_frame else 80
                    p.setBrush(QColor(255, 255, 255, alpha))
                    p.setPen(Qt.PenStyle.NoPen)
                    x = cx + (i - 1) * gap
                    p.drawEllipse(x - dot_r, y - dot_r, dot_r * 2, dot_r * 2)

            if self._mode == "downloading":
                bar_w = 110
                bar_h = 6
                bar_x = (self.width() - bar_w) // 2
                bar_y = 72
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, 50))
                p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)
                if self._dl_pct > 0:
                    fill_w = max(bar_h, int(bar_w * self._dl_pct / 100))
                    p.setBrush(QColor(120, 200, 255, 230))
                    p.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 3, 3)

            if self._substatus_text:
                p.setFont(QFont("Helvetica Neue", 10))
                p.setPen(QColor(255, 255, 255, 120))
                p.drawText(QRectF(0, self.height() - 28, self.width(), 20), Qt.AlignmentFlag.AlignCenter, self._substatus_text)

        p.end()

    def update_amplitude(self, amplitude: float):
        level = min(amplitude / MAX_AMPLITUDE, 1.0)
        self._bars.pop(0)
        self._bars.append(level)
        self.update()

    def _move_to_position(self):
        if self._saved_position:
            self.move(self._saved_position)
        else:
            screen = QApplication.primaryScreen().availableGeometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)

    def show_loading(self):
        # Phase 5 lock: if the user already has the permissions overlay up,
        # don't let a model_download_finished or "ready" signal silently
        # overwrite it. They have to act on the permissions overlay first.
        if self._mode == "permissions_needed":
            return
        self._mode = "loading"
        self._status_text = "Loading model..."
        self._substatus_text = "please wait"
        self._move_to_position()
        self.show()
        self.raise_()
        self.update()

    def show_downloading(self):
        if self._mode == "permissions_needed":
            return
        self._mode = "downloading"
        self._status_text = "Downloading model"
        self._substatus_text = "preparing…"
        self._dl_pct = 0
        self._move_to_position()
        self.show()
        self.raise_()
        self.update()

    def update_download_progress(self, pct: int, mb_done: int, mb_total: int):
        if self._mode != "downloading":
            return
        self._dl_pct = pct
        if mb_total > 0:
            self._substatus_text = f"{pct}%  {mb_done} / {mb_total} MB"
        else:
            self._substatus_text = f"{mb_done} MB"
        self.update()

    def show_recording(self, elapsed: str = "0:00"):
        self._mode = "recording"
        self._elapsed = elapsed
        self._bars = [0.0] * WAVEFORM_BARS
        self._move_to_position()
        self.show()
        self.raise_()
        self.update()

    def update_timer(self, elapsed: str):
        if self._mode == "recording":
            self._elapsed = elapsed
            self.update()

    def show_transcribing(self):
        self._mode = "transcribing"
        self._status_text = "Transcribing..."
        self._substatus_text = ""
        self._bars = [0.0] * WAVEFORM_BARS
        self._spin_frame = 0
        self._spin_timer.start()
        self.update()

    def show_inserted(self):
        self._mode = "done"
        self._status_text = "Inserted"
        self._substatus_text = ""
        self._spin_timer.stop()
        self.update()

    def show_copied(self):
        self._mode = "done"
        self._status_text = "Copied"
        self._substatus_text = "⌘V to paste"
        self._spin_timer.stop()
        self.update()

    def show_error(self, message: str):
        self._mode = "error"
        self._status_text = "Model failed"
        self._substatus_text = message[:40]
        self._spin_timer.stop()
        self._move_to_position()
        self.show()
        self.raise_()
        self.update()

    def show_permissions_needed(self, acc_ok: bool, im_ok: bool):
        self._mode = "permissions_needed"
        self._perm_acc = acc_ok
        self._perm_im = im_ok
        self._status_text = "Permissions required"
        self._substatus_text = "click to open Settings"
        self._spin_timer.stop()
        # Cursor is managed dynamically by mouseMoveEvent — pointer only
        # when hovering the substring text, default arrow elsewhere.
        self._move_to_position()
        self.show()
        self.raise_()
        self.update()

    def hide_overlay(self):
        self._mode = "idle"
        self._spin_timer.stop()
        self.unsetCursor()
        self.hide()


class TellarApp:
    def __init__(self, menubar):
        self.recorder = Recorder()
        self.bridge = Bridge()
        self.overlay = OverlayWidget()
        self.menubar = menubar
        self._recording = False
        self._ready = False
        self._target_app = None
        self._record_start = None
        self._timer = QTimer()
        self._timer.setInterval(100)
        # Phase 5: deferred hotkey listener — main() stores it here so we
        # can spin up the hotkey thread later if the user grants permissions
        # at runtime (poll loop notices, starts the thread without a
        # full app relaunch).
        self.listener = None
        # Polls _check_permissions while the permissions_needed overlay
        # is up. macOS doesn't notify apps about TCC changes, so we have
        # to ask repeatedly. 1.5s interval feels responsive without
        # burning CPU.
        self._perm_poll_timer = QTimer()
        self._perm_poll_timer.setInterval(1500)
        self._perm_poll_timer.timeout.connect(self._perm_recheck)
        self._timer.timeout.connect(self._on_tick)

    def attach_menubar(self):
        """Replace placeholder title with the wave icon now that QPixmap is available."""
        self.menubar.set_icon_pixmap(_make_wave_pixmap("#ff9900"))

    def _elapsed_str(self) -> str:
        if not self._record_start:
            return "0:00"
        elapsed = (datetime.now() - self._record_start).total_seconds()
        m, s = divmod(int(elapsed), 60)
        return f"{m}:{s:02d}"

    def _on_tick(self):
        if not self._recording:
            return
        elapsed = self._elapsed_str()
        self.overlay.update_timer(elapsed)
        self.overlay.update_amplitude(self.recorder.amplitude)
        self.menubar.set_icon_title(elapsed)

    def on_model_ready(self):
        self._ready = True
        self.menubar.set_icon_pixmap(_make_wave_pixmap("#ffffff"))
        # Don't override the permissions_needed overlay — model loading
        # finishing in the background doesn't fix a missing permission;
        # the user still has to act on that overlay first.
        if self.overlay._mode != "permissions_needed":
            self.overlay.hide_overlay()
        self.menubar.set_record_action_enabled(True)
        self.menubar.set_record_action_text("Start Recording  (⌃Space)")

    def on_permissions_needed(self, acc_ok: bool, im_ok: bool):
        """Show the permissions overlay and start polling so the UI
        reflects the user's grants in real time. When both are granted
        we can spin up the hotkey listener without a full app restart."""
        self.overlay.show_permissions_needed(acc_ok, im_ok)
        if not self._perm_poll_timer.isActive():
            self._perm_poll_timer.start()

    def _perm_recheck(self):
        acc_ok, im_ok = _check_permissions()
        if acc_ok and im_ok:
            log.info("Permissions granted via runtime poll; starting hotkey")
            self._perm_poll_timer.stop()
            self.overlay.hide_overlay()
            self._start_hotkey_thread()
        elif (acc_ok, im_ok) != (self.overlay._perm_acc, self.overlay._perm_im):
            # State changed but still missing one — refresh the overlay so
            # the user sees the ✗ flip to ✓ as soon as they grant.
            self.overlay.show_permissions_needed(acc_ok, im_ok)

    def _start_hotkey_thread(self):
        if self.listener is None:
            log.error("Cannot start hotkey thread: listener was never created")
            return
        threading.Thread(
            target=_run_hotkey, args=(self.listener, self.bridge), daemon=True
        ).start()
        log.info("Hotkey listener thread started (runtime)")

    def on_model_error(self, message: str):
        log.error("Model load failed: %s", message)
        self.menubar.set_icon_pixmap(_make_wave_pixmap("#cc0000"))
        self.overlay.show_error(message)
        self.menubar.set_record_action_text("Model unavailable")
        self.menubar.set_record_action_enabled(False)

    def on_hotkey_start(self):
        if not self._ready or self._recording:
            return
        log.info("Hotkey: start recording")
        self._recording = True
        self.bridge.recording_started.emit()

    def on_hotkey_stop(self):
        if not self._recording:
            return
        log.info("Hotkey: stop recording")
        self._recording = False
        self.bridge.recording_stopped.emit()

    def on_toggle(self):
        if not self._ready:
            log.debug("Toggle ignored: model not ready")
            return
        if self._recording:
            self.on_hotkey_stop()
        else:
            self.on_hotkey_start()

    def on_cancel(self):
        if not self._recording:
            return
        log.info("Hotkey: cancel recording")
        self._recording = False
        self.bridge.recording_cancelled.emit()

    def _cancel_recording(self):
        self._timer.stop()
        self.recorder.stop()
        self.recorder.cleanup()
        self._target_app = None
        self.overlay.hide_overlay()
        self.menubar.set_icon_pixmap(_make_wave_pixmap("#ffffff"))
        self.menubar.set_record_action_text("Start Recording  (⌃Space)")

    def _start_recording(self):
        self._target_app = get_frontmost_app()
        target_bundle = self._target_app.bundleIdentifier() if self._target_app else None
        log.info("Recording started, target app: %s", target_bundle)
        self._record_start = datetime.now()
        self.recorder.start()
        self.overlay.show_recording("0:00")
        self.menubar.set_icon_title("0:00")
        self.menubar.set_record_action_text("Stop Recording  (⌃Space)")
        self._timer.start()

    def _stop_recording(self):
        self._timer.stop()
        self.overlay.show_transcribing()
        self.menubar.set_icon_pixmap(_make_wave_pixmap("#ff9900"))

        elapsed = (datetime.now() - self._record_start).total_seconds() if self._record_start else 0
        log.info("Recording stopped after %.1fs", elapsed)
        wav_path = self.recorder.stop()
        if not wav_path:
            log.info("WAV empty (silence trimmed below threshold), aborting transcription")
            self.overlay.hide_overlay()
            self.menubar.set_icon_pixmap(_make_wave_pixmap("#ffffff"))
            self._recording = False
            return

        try:
            wav_size = Path(wav_path).stat().st_size
        except OSError:
            wav_size = -1
        log.info("WAV ready: %s (%d bytes)", wav_path, wav_size)

        def process():
            import time
            t0 = time.time()
            try:
                text = transcribe_audio(wav_path)
                duration = time.time() - t0
                log.info("Transcription done in %.2fs, %d chars", duration, len(text))
                self.recorder.cleanup()
                if text.strip():
                    self.bridge.transcription_done.emit(text.strip())
                else:
                    log.info("Transcription returned empty text")
                    self.bridge.transcription_empty.emit()
            except Exception:
                log.exception("Transcription failed")
                self.bridge.transcription_empty.emit()

        threading.Thread(target=process, daemon=True).start()

    def _insert_result(self, text):
        self._recording = False
        auto_paste = self.menubar.is_auto_paste_enabled()
        try:
            pasted = insert_text(text, self._target_app, auto_paste)
            log.info("Insert: %s (%d chars, auto_paste=%s)",
                     "pasted" if pasted else "clipboard-only", len(text), auto_paste)
            if pasted:
                self.overlay.show_inserted()
            else:
                self.overlay.show_copied()
        except Exception:
            log.exception("Insert error")
            self.overlay.show_copied()
        self._target_app = None
        self.menubar.set_icon_pixmap(_make_wave_pixmap("#00cc66"))
        QTimer.singleShot(1500, self._finish)

    def _finish(self):
        self.overlay.hide_overlay()
        self.menubar.set_icon_pixmap(_make_wave_pixmap("#ffffff"))
        self._recording = False
        self.menubar.set_record_action_text("Start Recording  (⌃Space)")


LOCK_FILE = STATE_DIR / "tellar.lock"
PID_FILE = STATE_DIR / "tellar.pid"

# Module-level so the descriptor lives for the lifetime of the process; closing
# the file would release the advisory lock.
_lock_fd = None


def _acquire_lock() -> bool:
    global _lock_fd
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    _lock_fd = fd
    # Pidfile is informational only — useful for diag/debug, not for locking.
    try:
        PID_FILE.write_text(str(os.getpid()))
    except OSError:
        pass
    return True


def _release_lock():
    global _lock_fd
    if _lock_fd is not None:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            os.close(_lock_fd)
        except OSError:
            pass
        _lock_fd = None
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _set_accessory_policy():
    """Hide Dock icon for menu-bar-only behavior.

    LSUIElement in Info.plist is honored only when the .app's main executable
    is itself the running process. Our bash launcher exec's python3.12, after
    which macOS associates the process with python, not the bundle, and
    LSUIElement is ignored. Setting the activation policy at runtime achieves
    the same effect regardless of how Python was launched.
    """
    try:
        from AppKit import NSApplication
        NSApplicationActivationPolicyAccessory = 1
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )
        log.info("Activation policy set to accessory (no Dock icon)")
    except Exception:
        log.exception("Could not set accessory activation policy")


def main():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if "--diag" in sys.argv:
        from .diag import print_diag
        print_diag()
        sys.exit(0)

    setup_logging()

    # Diagnostic: what does macOS think this process is?
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        log.info("mainBundle path: %s", bundle.bundlePath())
        log.info("mainBundle ID: %s", bundle.bundleIdentifier())
        info = bundle.infoDictionary() or {}
        log.info("LSUIElement from Info.plist: %s", info.get("LSUIElement"))
    except Exception:
        log.exception("NSBundle diagnostic failed")

    # Re-enable accessory policy now that QSystemTrayIcon is gone.
    # NSStatusBar item is what we want visible; accessory mode is the right
    # activation policy for a menu-bar-only app.
    _set_accessory_policy()

    if not _acquire_lock():
        log.warning("Tellar is already running. Exiting.")
        print("Tellar is already running. Exiting.", flush=True)
        sys.exit(0)

    # CRITICAL: create the NSStatusBar item BEFORE QApplication.
    # PyQt6 installs its own NSApplication subclass during QApplication init,
    # which interferes with NSStatusItem visibility if the item is created
    # afterwards. Creating it first attaches it to the shared NSApp instance
    # before Qt takes over. The on_toggle callback is wired up via a holder
    # because TellarApp doesn't exist yet.
    _toggle_holder = [None]
    menubar = MenuBarIcon(on_toggle=lambda: _toggle_holder[0]() if _toggle_holder[0] else None)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.aboutToQuit.connect(_release_lock)
    app.aboutToQuit.connect(lambda: log.info("Tellar exiting"))

    tellar = TellarApp(menubar)
    _toggle_holder[0] = tellar.on_toggle
    tellar.attach_menubar()

    tellar.bridge.recording_started.connect(tellar._start_recording)
    tellar.bridge.recording_stopped.connect(tellar._stop_recording)
    tellar.bridge.recording_cancelled.connect(tellar._cancel_recording)
    tellar.bridge.transcription_done.connect(tellar._insert_result)
    tellar.bridge.transcription_empty.connect(tellar._finish)
    tellar.bridge.model_ready.connect(tellar.on_model_ready)
    tellar.bridge.model_error.connect(tellar.on_model_error)
    tellar.bridge.model_download_progress.connect(tellar.overlay.update_download_progress)
    # When download finishes, transition the overlay to "Loading model..." while
    # mlx_whisper imports and load_model run.
    tellar.bridge.model_download_finished.connect(tellar.overlay.show_loading)
    tellar.bridge.permissions_needed.connect(tellar.on_permissions_needed)

    if not model_exists():
        log.info("Whisper model not found locally, will download on preload")
        tellar.overlay.show_downloading()
    else:
        log.info("Whisper model already cached")
        tellar.overlay.show_loading()

    def preload_model():
        import time
        t0 = time.time()
        log.info("Preloading whisper model...")

        download_started = False

        def on_progress(pct, mb_done, mb_total):
            nonlocal download_started
            download_started = True
            tellar.bridge.model_download_progress.emit(pct, mb_done, mb_total)

        try:
            get_model(on_download_progress=on_progress)
            # Only nudge the overlay from "downloading" → "loading" if a
            # download actually occurred. When the model was already cached,
            # the overlay never went into downloading mode in the first
            # place, and emitting here would clobber whatever state it's
            # in (e.g. permissions_needed shown by Phase 5).
            if download_started:
                tellar.bridge.model_download_finished.emit()
        except Exception as e:
            log.exception("Model preload failed")
            tellar.bridge.model_error.emit(str(e))
            return
        log.info("Model preload done in %.2fs", time.time() - t0)
        tellar.bridge.model_ready.emit()

    threading.Thread(target=preload_model, daemon=True).start()

    from .hotkey import HotkeyListener
    listener = HotkeyListener(on_toggle=tellar.on_toggle, on_cancel=tellar.on_cancel)
    # Store on TellarApp so the runtime-grant poll loop can spin up the
    # hotkey thread later if permissions are missing at startup.
    tellar.listener = listener

    # Phase 5: check permissions BEFORE spawning the hotkey thread. If
    # either Input Monitoring or Accessibility is missing, surface the
    # overlay immediately and skip the hotkey thread (it would just fail
    # silently in CGEventTapCreate). The poll loop in TellarApp will
    # start the thread the moment the user grants both.
    acc_ok, im_ok = _check_permissions()
    log.info("Permissions check: accessibility=%s input_monitoring=%s", acc_ok, im_ok)
    if not (acc_ok and im_ok):
        tellar.bridge.permissions_needed.emit(acc_ok, im_ok)
    else:
        tellar._start_hotkey_thread()

    log.info("Entering Qt event loop")
    sys.exit(app.exec())


def _run_hotkey(listener, bridge):
    import Quartz
    try:
        listener.start()
    except PermissionError as e:
        # Belt-and-suspenders: even after the proactive _check_permissions
        # gate, a race or API discrepancy could let us through and have
        # CGEventTapCreate still fail. Treat any failure here as a missing
        # permission and re-surface the overlay. We don't know which one
        # is actually missing at this layer, so flag both as suspect.
        log.error("Hotkey unavailable: %s", e)
        bridge.permissions_needed.emit(False, False)
        return
    Quartz.CFRunLoopRun()


if __name__ == "__main__":
    main()
