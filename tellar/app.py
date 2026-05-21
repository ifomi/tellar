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


def _make_wave_pixmap(color: str) -> QPixmap:
    scale = 2
    size = 22 * scale
    pm = QPixmap(size, size)
    pm.setDevicePixelRatio(scale)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # A single arc above a source dot — minimal "signal/tell" mark.
    cx, cy = 11.0, 14.0

    pen = QPen(QColor(color))
    pen.setWidthF(2.2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    r = 9.0
    arc_rect = QRectF(cx - r, cy - r, r * 2, r * 2)
    p.drawArc(arc_rect, 0, 180 * 16)

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    dot_d = 4.0
    p.drawEllipse(QRectF(cx - dot_d / 2, cy - dot_d / 2, dot_d, dot_d))

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


class Bridge(QObject):
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    recording_cancelled = pyqtSignal()
    transcription_done = pyqtSignal(str)
    transcription_empty = pyqtSignal()
    model_ready = pyqtSignal()
    model_error = pyqtSignal(str)


class OverlayWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(150, 120)
        self._bars = [0.0] * WAVEFORM_BARS
        self._mode = "idle"
        self._elapsed = ""
        self._spin_frame = 0
        self._spin_timer = QTimer()
        self._spin_timer.setInterval(200)
        self._spin_timer.timeout.connect(self._spin_tick)
        self._drag_pos: Optional[QPoint] = None
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
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_pos:
            self._drag_pos = None
            self._save_position()

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

        elif self._mode in ("loading", "transcribing", "done", "error"):
            p.setFont(QFont("Helvetica Neue", 13, QFont.Weight.DemiBold))
            if self._mode == "error":
                p.setPen(QColor(255, 100, 100, 240))
            else:
                p.setPen(QColor(255, 255, 255, 240))
            text_y = 35 if self._mode == "transcribing" else 40
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
        self._mode = "loading"
        self._status_text = "Loading model..."
        self._substatus_text = "please wait"
        self._move_to_position()
        self.show()
        self.raise_()
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

    def hide_overlay(self):
        self._mode = "idle"
        self._spin_timer.stop()
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
        self.overlay.hide_overlay()
        self.menubar.set_record_action_enabled(True)
        self.menubar.set_record_action_text("Start Recording  (⌃Space)")

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

    if not model_exists():
        log.info("Whisper model not found locally, will download on preload")
        print(f"Downloading whisper model to {MODEL_DIR}...")
        tellar.overlay.show_loading()
    else:
        log.info("Whisper model already cached")

    def preload_model():
        import time
        t0 = time.time()
        log.info("Preloading whisper model...")
        try:
            get_model()
        except Exception as e:
            log.exception("Model preload failed")
            tellar.bridge.model_error.emit(str(e))
            return
        log.info("Model preload done in %.2fs", time.time() - t0)
        tellar.bridge.model_ready.emit()

    threading.Thread(target=preload_model, daemon=True).start()

    from .hotkey import HotkeyListener
    listener = HotkeyListener(on_toggle=tellar.on_toggle, on_cancel=tellar.on_cancel)
    try:
        hotkey_thread = threading.Thread(target=_run_hotkey, args=(listener,), daemon=True)
        hotkey_thread.start()
        log.info("Hotkey listener thread started")
    except PermissionError as e:
        log.error("Hotkey permission error: %s", e)
        print(f"⚠️  {e}")

    log.info("Entering Qt event loop")
    sys.exit(app.exec())


def _run_hotkey(listener):
    import Quartz
    listener.start()
    Quartz.CFRunLoopRun()


if __name__ == "__main__":
    main()
