"""Application bootstrap."""

from __future__ import annotations

import sys
import threading
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from . import APP_NAME, version_string
from .ffmpeg import ffmpeg_bin, ffprobe_bin


def _set_app_id() -> None:
    """Windows: group the taskbar button under KangEdit's own icon, not python.exe's, when run from source."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{APP_NAME}.{APP_NAME}")
        except (AttributeError, OSError):
            pass


def _front(w) -> None:
    """Show a window (there is no main window yet) in front: behind the Explorer window it was started from it
    looks as if nothing happens."""
    w.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    w.show()
    w.raise_()
    w.activateWindow()


def _download_ffmpeg() -> bool:
    """Offer to download FFmpeg next to the app; True once it is there."""
    from . import crashlog, ffmpeg_setup
    note = ffmpeg_setup.missing_note()
    crashlog.log.warning("No usable FFmpeg: %s (%s)", note, ffmpeg_setup.installed())
    ask = QMessageBox(QMessageBox.Question, APP_NAME,
                      f"{note}\n\nDownload FFmpeg {ffmpeg_setup.MIN_VERSION}+ now (about 80-200 MB, into "
                      f"{ffmpeg_setup.bin_dir()})?", QMessageBox.Yes | QMessageBox.No)
    _front(ask)
    if ask.exec() != QMessageBox.Yes:
        QMessageBox.information(None, APP_NAME, f"Install FFmpeg {ffmpeg_setup.MIN_VERSION} or newer and make "
                                                "sure 'ffmpeg' and 'ffprobe' are on your PATH, or start KangEdit "
                                                "again to download it.")
        return False
    dlg = QProgressDialog("Downloading FFmpeg…", "Cancel", 0, 1000)
    dlg.setWindowTitle(APP_NAME)
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setMinimumWidth(460)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    cancel = threading.Event()
    dlg.canceled.connect(cancel.set)
    state = {"p": 0.0, "msg": "Downloading FFmpeg…", "error": None, "done": False}

    def say(p, m):
        state.update(p=p, msg=m)
        if p == 0.0:
            crashlog.log.info("FFmpeg download: %s", m)

    def work():
        try:
            path = ffmpeg_setup.install(say, cancel)
            crashlog.log.info("FFmpeg downloaded into %s", path)
        except Exception as e:  # noqa: BLE001
            state["error"] = str(e)
            crashlog.log.error("FFmpeg download failed: %s", e)
        state["done"] = True

    _front(dlg)
    threading.Thread(target=work, daemon=True, name="ffmpeg download").start()
    t0 = time.monotonic()
    last = (None, 0.0, t0)
    while not state["done"]:
        dlg.setValue(int(state["p"] * 1000))
        msg = state["msg"]
        if (msg, state["p"]) != last[:2]:          # (the seconds count while nothing changes: it isn't stuck)
            last = (msg, state["p"], time.monotonic())
        waited = time.monotonic() - last[2]
        dlg.setLabelText(msg + (f" ({waited:.0f} s)" if waited >= 2 else ""))
        QApplication.processEvents()
        time.sleep(0.03)
    crashlog.log.info("FFmpeg download ended after %.0f s", time.monotonic() - t0)
    dlg.close()
    if state["error"] or not (ffmpeg_bin() and ffprobe_bin()):
        if not cancel.is_set():
            box = QMessageBox(QMessageBox.Critical, APP_NAME,
                              f"FFmpeg could not be downloaded:\n{state['error']}\n\nDownload it yourself "
                              "(Windows: gyan.dev/ffmpeg/builds, \"release essentials\") and put ffmpeg and ffprobe "
                              f"into\n{ffmpeg_setup.bin_dir()}\nor on your PATH, then start KangEdit again.")
            box.setTextInteractionFlags(Qt.TextSelectableByMouse)
            _front(box)
            box.exec()
        return False
    return True


def show_maximized_on_current_screen(win) -> None:
    """Open maximized on the display the mouse pointer is on (where the app was started from), not always on
    the primary one."""
    from PySide6.QtGui import QCursor, QGuiApplication
    screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
    if screen is not None:
        win.setScreen(screen)
        win.move(screen.availableGeometry().topLeft())   # Windows maximizes on the screen the window is on
    win.showMaximized()


def self_test() -> int:
    """CI / packaging check: every module and native library loads, and the main window builds and closes
    offscreen with throwaway settings (no network, no dialogs). Exit code 0 on success."""
    import importlib
    import os
    import pkgutil
    import shutil
    import tempfile
    import traceback
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    tmp = tempfile.mkdtemp(prefix="kangedit-selftest-")
    os.environ["KANGEDIT_SETTINGS_DIR"] = os.environ["KANGEDIT_DATA_DIR"] = tmp
    try:
        import kangedit
        for mod in pkgutil.walk_packages(kangedit.__path__, "kangedit."):
            importlib.import_module(mod.name)
        for lib in ("av", "cv2", "numpy", "ctranslate2", "faster_whisper", "onnxruntime"):
            importlib.import_module(lib)
        # The smarter analysis' models are bundled and run (a build that left them out fails here).
        import numpy as np
        from . import smart
        if not smart.available():
            raise RuntimeError("smarter analysis models missing: " + smart.EMBED_MODEL + ", " + smart.SOUND_MODEL)
        smart.describe([np.zeros((90, 160, 3), np.uint8)])
        smart.hear(np.zeros(16000, np.float32))
        app = QApplication.instance() or QApplication([sys.argv[0]])
        from .ui.main_window import MainWindow
        win = MainWindow()
        show_maximized_on_current_screen(win)
        app.processEvents()
        win.project.dirty = False
        win.close()
        app.processEvents()
        _report(f"{APP_NAME} {version_string()} self-test OK")
        return 0
    except Exception:  # noqa: BLE001
        _report(traceback.format_exc())
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _report(text: str) -> None:
    """Print, and append to $KANGEDIT_SELFTEST_LOG: a windowed build has no console (sys.stdout is None)."""
    import os
    log = os.environ.get("KANGEDIT_SELFTEST_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n")
    if sys.stdout is not None:
        print(text, flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    from .asr_qwen import WORKER_FLAG
    if WORKER_FLAG in argv[1:]:                # the Qwen3-ASR helper process (see asr_qwen.py): no window
        from .asr_qwen import worker_main
        return worker_main(argv[argv.index(WORKER_FLAG) + 1:])
    if "--version" in argv[1:]:
        _report(f"{APP_NAME} {version_string()}")
        return 0
    if "--self-test" in argv[1:]:
        return self_test()
    from . import crashlog
    crashlog.install()
    _set_app_id()
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")
    from .ui.theme import STYLE
    app.setStyleSheet(STYLE)
    from .ui.icons import app_icon
    app.setWindowIcon(app_icon())

    if (not ffmpeg_bin() or not ffprobe_bin()) and not _download_ffmpeg():
        return 1

    from .ui.main_window import MainWindow
    win = MainWindow()
    show_maximized_on_current_screen(win)
    crashlog.log.info("Main window shown")
    args = [a for a in argv[1:] if not a.startswith("-")]
    if args and args[0].endswith(".kgproj"):
        win.load_project_file(args[0])
    elif args:
        win.import_files(args)
    code = app.exec()
    crashlog.log.info("Exiting with code %s", code)
    return code
