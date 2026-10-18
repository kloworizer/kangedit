"""The background task queue without a console (pythonw.exe, the windowed build: sys.stderr is None)."""

import os
import sys
import time

from kangedit import crashlog
from kangedit.tasks import TaskQueue


def test_failed_task_says_why_without_a_console(qapp, monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    q = TaskQueue()
    errors, done = [], []

    def fail(progress, cancel):
        raise RuntimeError("download failed")

    q.submit("Downloading", fail, done.append, errors.append)
    end = time.monotonic() + 5
    while not errors and not done and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    q.shutdown()
    assert errors == ["download failed"] and not done          # said in the app (and the log), not lost


def test_missing_console_streams_take_writes(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(crashlog, "_quiet", False)
    crashlog.quiet_missing_streams()
    print("progress 10%", file=sys.stderr)                     # what a library's progress bar does
    sys.stdout.write("x")
    assert crashlog._quiet and sys.stderr.name == os.devnull
