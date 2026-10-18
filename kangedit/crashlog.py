"""Crash and error logging: everything that would otherwise vanish (a windowed build has no console) goes to
<app data>/logs/kangedit.log, with native crashes (segfaults in PyAV, OpenCV, Qt...) dumped by faulthandler."""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import threading
from pathlib import Path

log = logging.getLogger("kangedit")

_fault_file = None   # kept open for the whole run: faulthandler writes to it from the crash itself


_quiet = False       # the app has no console: stdout / stderr go nowhere


def quiet_missing_streams() -> None:
    """Without a console (pythonw.exe, the windowed build) sys.stdout and sys.stderr are None, and any library
    printing to them fails ('NoneType' object has no attribute 'write'): a download's progress bar, a
    traceback printed while handling an error. They are given a stream that takes everything and drops it
    (what matters goes to the log)."""
    global _quiet
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
            _quiet = True


def log_dir() -> Path:
    from .config import app_data_dir
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    return log_dir() / "kangedit.log"


def install() -> Path:
    """Start logging to the log file and hook every way the app can die. Returns the log file's path."""
    global _fault_file
    from . import APP_NAME, version_string
    quiet_missing_streams()
    path = log_path()
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if os.environ.get("KANGEDIT_DEBUG") else logging.INFO)
    root.addHandler(handler)
    if not _quiet:
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        root.addHandler(console)

    # Native crashes: Python can't catch these, so faulthandler writes every thread's stack to a side file
    # (it must be a raw, always-open file), which the next start copies into the log.
    fault = log_dir() / "fault.log"
    _collect_previous_fault(fault)
    _fault_file = open(fault, "w", encoding="utf-8")
    faulthandler.enable(_fault_file, all_threads=True)

    sys.excepthook = _excepthook
    threading.excepthook = lambda a: log.critical("Unhandled exception in thread %s", a.thread.name if a.thread
                                                   else "?", exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    _install_qt_handler()
    log.info("==== %s %s starting | Python %s | %s | argv=%s", APP_NAME, version_string(),
             platform.python_version(), platform.platform(), sys.argv)
    return path


def _collect_previous_fault(fault: Path) -> None:
    try:
        text = fault.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return
    if text:
        log.critical("The previous run crashed (native fault):\n%s", text)


def _excepthook(exc_type, exc, tb):
    # Also reached for exceptions raised inside Qt slots/event handlers, which PySide routes here.
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))


def _install_qt_handler() -> None:
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:
        return
    levels = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
              QtMsgType.QtWarningMsg: logging.WARNING, QtMsgType.QtCriticalMsg: logging.ERROR,
              QtMsgType.QtFatalMsg: logging.CRITICAL}
    qt_log = logging.getLogger("qt")

    def handler(mode, context, message):
        where = f" ({context.file}:{context.line})" if context.file else ""
        qt_log.log(levels.get(mode, logging.WARNING), "%s%s", message, where)

    qInstallMessageHandler(handler)
