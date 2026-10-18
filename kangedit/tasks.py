"""Sequential background task queue with progress and cancellation."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Signal

from .ffmpeg import Cancelled

log = logging.getLogger("kangedit.tasks")

CANCEL_GRACE = 0.5   # seconds a cancelled task gets to stop before the queue moves on

TaskFn = Callable[[Callable[[float, str], None], threading.Event], Any]


@dataclass
class Task:
    name: str
    fn: TaskFn
    on_done: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[str], None]] = None


class TaskQueue(QObject):
    progress = Signal(float, str)
    started = Signal(str)
    idle = Signal()
    pending_changed = Signal(int)
    _done = Signal(object, object)    # task, result
    _error = Signal(object, str)      # task, message

    def __init__(self):
        super().__init__()
        self._q: "queue.Queue[Optional[Task]]" = queue.Queue()
        self._cancel = threading.Event()
        self._pending = 0
        self._lock = threading.Lock()
        self.current: Optional[str] = None
        self._done.connect(self._on_done)
        self._error.connect(self._on_error)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._pending > 0

    def submit(self, name: str, fn: TaskFn, on_done=None, on_error=None) -> None:
        with self._lock:
            self._pending += 1
        self.pending_changed.emit(self._pending)
        self._q.put(Task(name, fn, on_done, on_error))

    def cancel_all(self) -> None:
        self._cancel.set()
        try:
            while True:
                t = self._q.get_nowait()
                if t is not None:
                    with self._lock:
                        self._pending -= 1
        except queue.Empty:
            pass
        self.pending_changed.emit(self._pending)

    def shutdown(self) -> None:
        self.cancel_all()
        self._q.put(None)

    def _loop(self):
        while True:
            task = self._q.get()
            if task is None:
                return
            # Each task gets its own event: a cancelled task that is still unwinding in the background keeps
            # seeing it set, while the next task starts with a clear one.
            cancel = self._cancel = threading.Event()
            self.current = task.name
            self.started.emit(task.name)
            out: dict = {}

            def run(task=task, cancel=cancel, out=out):
                try:
                    out["res"] = task.fn(lambda p, m="": None if cancel.is_set() else
                                         self.progress.emit(float(p), m or task.name), cancel)
                except Cancelled:
                    out["cancelled"] = True
                except Exception as e:  # noqa: BLE001
                    out["error"] = f"{e}"
                    log.exception("Task %r failed", task.name)

            # The task runs in its own thread so a cancel is answered at once, even while the task is stuck in
            # a call that cannot be interrupted (native code, a subprocess); it then finishes or bails out on
            # its own, and its result is dropped.
            worker = threading.Thread(target=run, daemon=True, name=f"task: {task.name}")
            worker.start()
            while worker.is_alive():
                worker.join(0.1)
                if cancel.is_set():
                    worker.join(CANCEL_GRACE)
                    break
            if cancel.is_set() or out.get("cancelled"):
                self._error.emit(task, "")
            elif "error" in out:
                self._error.emit(task, out["error"])
            else:
                self._done.emit(task, out.get("res"))

    def _finish(self):
        with self._lock:
            self._pending = max(0, self._pending - 1)
        self.current = None
        self.pending_changed.emit(self._pending)
        if self._pending == 0:
            self.idle.emit()

    def _on_done(self, task: Task, res):
        try:
            if task.on_done:
                task.on_done(res)
        finally:
            self._finish()

    def _on_error(self, task: Task, msg: str):
        try:
            if task.on_error:
                task.on_error(msg)
        finally:
            self._finish()
