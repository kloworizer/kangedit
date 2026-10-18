"""Choosing the hook: the strongest moments of the whole video as thumbnails (its start, middle and end), the
strongest ticked, as many as the hook has clips. Tick others, or ask for more suggestions."""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QToolButton, QVBoxLayout, QWidget)

from ..autoedit import MAX_HOOK_CLIPS
from ..models import TimelineClip
from .theme import fmt_time

FRAME_W = 112          # each of a card's three frames
PREVIEW_FPS = 12       # a moment's preview plays this many frames a second (decoded small, in the background)
PREVIEW_W = 360


def _pixmap(rgb: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(rgb)
    img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img)


class MomentPreview(QLabel):
    """Plays a moment over and over: its frames decoded (decode(clip, cancel) -> RGB frames at PREVIEW_FPS)
    on a background thread, shown as they are ready."""

    _ready = Signal(object, object)          # (clip id, frames) from the decoding thread

    def __init__(self, decode: Callable, parent=None):
        super().__init__(parent)
        self.decode = decode
        self.setFixedSize(PREVIEW_W, PREVIEW_W * 9 // 16)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #111; color: #9aa0ab;")
        self.setText("Point at a moment to play it")
        self.clip_id = ""
        self.frames: list = []
        self._k = 0
        self._cancel = threading.Event()
        self._timer = QTimer(self, interval=int(1000 / PREVIEW_FPS))
        self._timer.timeout.connect(self._next)
        self._ready.connect(self._got)

    def play(self, clip: TimelineClip) -> None:
        if clip.id == self.clip_id:
            return
        self.stop()
        self.clip_id = clip.id
        self.setText("Loading…")
        cancel = self._cancel = threading.Event()
        c = TimelineClip.from_dict(clip.to_dict())      # a private copy for the decoding thread

        def work():
            try:
                frames = self.decode(c, cancel)
            except Exception:  # noqa: BLE001 - a moment that can't be decoded just isn't previewed
                frames = []
            if not cancel.is_set():
                self._ready.emit(c.id, frames)

        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        self._cancel.set()
        self._timer.stop()
        self.frames, self.clip_id = [], ""

    def _got(self, clip_id: str, frames: list) -> None:
        if clip_id != self.clip_id:
            return
        self.frames = [_pixmap(f).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) for f in frames]
        self._k = 0
        if self.frames:
            self._next()
            self._timer.start()
        else:
            self.setText("Can't play this moment")

    def _next(self) -> None:
        if self.frames:
            self.setPixmap(self.frames[self._k % len(self.frames)])
            self._k += 1


class _Hover(QObject):
    """Plays a card's moment in the preview when the pointer comes onto it."""

    def __init__(self, picker: "HookPicker"):
        super().__init__(picker)
        self.picker = picker

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Enter:
            self.picker.preview_card(obj)
        return False


def strip(analysis, a: float, b: float, width: int = FRAME_W) -> Optional[QPixmap]:
    """Three frames of a moment a..b side by side (start, middle, end), from the analysis's thumbnails."""
    if analysis is None or not len(analysis.thumbs):
        return None
    tt = np.asarray(analysis.thumb_times, np.float32)
    frames = []
    for t in (a + 0.15, (a + b) / 2, b - 0.15):
        k = int(np.argmin(np.abs(tt - t))) if len(tt) else 0
        frames.append(np.asarray(analysis.thumbs[min(k, len(analysis.thumbs) - 1)]))
    h = min(f.shape[0] for f in frames)
    arr = np.ascontiguousarray(np.hstack([f[:h] for f in frames]))
    img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaledToWidth(3 * width, Qt.SmoothTransformation)


class HookPicker(QDialog):
    """fetch(count, avoid) -> candidate moments for a `count`-clip hook not overlapping `avoid` (moments
    already offered), strongest first. names(clip) -> a card's text; thumbs(clip) -> its picture;
    length_for(count) -> how long each moment of a `count`-clip hook is; decode(clip, cancel) -> its frames,
    played in the preview when the pointer is on its card (None: no preview)."""

    def __init__(self, parent, count: int, fetch: Callable[[int, list], list],
                 thumbs: Callable[[TimelineClip], Optional[QPixmap]], names: Callable[[TimelineClip], str],
                 length_for: Callable[[int], float], decode: Optional[Callable] = None):
        super().__init__(parent)
        self.setWindowTitle("🎣 Choose the hook")
        self.fetch, self.thumbs, self.names, self.length_for = fetch, thumbs, names, length_for
        # decode(clip, cancel) -> frames: point at a card to play its moment (None: no preview)
        self.preview = MomentPreview(decode) if decode is not None else None
        self._hover = _Hover(self)
        self.shown: list[TimelineClip] = []      # every moment offered (More suggestions offers others)
        self.cards: list[tuple[QToolButton, TimelineClip]] = []
        self.order: list[QToolButton] = []       # the ticked cards, in the order they were ticked
        lay = QVBoxLayout(self)
        intro = QLabel("The strongest moments of the whole video (movement, faces, people laughing or cheering). "
                       "The ticked ones open the video in the order you tick them (the number on each). "
                       "Tick the ones you like.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Hook clips"))
        self.sp_count = QSpinBox()
        self.sp_count.setRange(1, MAX_HOOK_CLIPS)
        self.sp_count.setValue(max(1, min(MAX_HOOK_CLIPS, count)))
        self.sp_count.setToolTip("How many moments open the video: 1 quick hook, or a teaser of 2 s moments")
        self.sp_count.valueChanged.connect(self._count_changed)
        row.addWidget(self.sp_count)
        row.addStretch(1)
        self.btn_more = QPushButton("↻ More suggestions")
        self.btn_more.setToolTip("Other strong moments, not shown yet")
        self.btn_more.clicked.connect(self.more)
        row.addWidget(self.btn_more)
        lay.addLayout(row)
        self.grid_box = QWidget()
        self.grid = QGridLayout(self.grid_box)
        self.grid.setContentsMargins(0, 0, 0, 0)
        body = QHBoxLayout()
        body.addWidget(self.grid_box, 1)
        if self.preview is not None:
            side = QVBoxLayout()
            side.addWidget(self.preview)
            self.lbl_preview = QLabel("")
            self.lbl_preview.setWordWrap(True)
            self.lbl_preview.setMaximumWidth(PREVIEW_W)
            side.addWidget(self.lbl_preview)
            side.addStretch(1)
            body.addLayout(side)
        lay.addLayout(body, 1)
        self.lbl = QLabel("")
        self.lbl.setStyleSheet("color: #9aa0ab;")
        lay.addWidget(self.lbl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_ok = bb.button(QDialogButtonBox.Ok)
        self.btn_ok.setText("Use as the hook")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._length = None
        self._load(fresh=True)

    @property
    def count(self) -> int:
        return self.sp_count.value()

    def _load(self, fresh: bool = False) -> None:
        """Offer the strongest moments (fresh), or others not offered yet: the ticked ones stay, first, in the
        order they were ticked."""
        kept = [] if fresh else self.chosen()
        avoid = [] if fresh else list(self.shown)
        got = self.fetch(self.count, avoid)
        if not got and not fresh:                # every strong moment was offered: start the round again
            self.shown = list(kept)
            got = self.fetch(self.count, kept)
        if kept:                                 # as many cards as before: the ticked ones take some places
            got = got[:max(1, len(got) - len(kept))]
        # (only what is shown counts as offered: the rest comes with More suggestions)
        self.shown += [TimelineClip.from_dict(c.to_dict()) for c in got]
        self._length = self.length_for(self.count)
        self._show(kept + got, len(kept))

    def _show(self, clips: list, ticked: int = 0) -> None:
        """Cards for `clips`; the first `ticked` stay ticked in that order, else the strongest are ticked."""
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.cards = []
        self.order = []
        cols = 3
        for k, c in enumerate(clips):
            b = QToolButton()
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            pix = self.thumbs(c)
            if pix is not None:
                b.setIcon(pix)
                b.setIconSize(QSize(pix.width(), pix.height()))
            b.setText(self.names(c))
            b.setToolTip("Start · middle · end of the moment. Click to tick or untick it.")
            b.toggled.connect(lambda on, b=b: self._toggled(b, on))
            b.installEventFilter(self._hover)
            self.grid.addWidget(b, k // cols, k % cols)
            self.cards.append((b, c))
        if ticked:
            self._tick([b for b, _ in self.cards[:ticked]])
        else:
            self._tick_best()
        if self.cards:
            self.preview_card(self.cards[0][0])
        if not clips:
            self.lbl.setText("No strong moment left that the edit doesn't already show.")

    def preview_card(self, button) -> None:
        """Play a card's moment in the preview."""
        if self.preview is None:
            return
        for b, c in self.cards:
            if b is button:
                self.preview.play(c)
                self.lbl_preview.setText(self.names(c).replace("\n", " · "))
                return

    def done(self, r: int) -> None:
        if self.preview is not None:
            self.preview.stop()
        super().done(r)

    def _tick_best(self) -> None:
        """Tick the strongest moments, as many as the hook has clips (strongest first)."""
        self._tick([b for b, _ in self.cards[:self.count]])

    def _tick(self, buttons: list) -> None:
        """Tick exactly `buttons`, in that order."""
        for b, _ in self.cards:
            b.blockSignals(True)
            b.setChecked(b in buttons)
            b.blockSignals(False)
        self.order = list(buttons)
        self._update()

    def _toggled(self, b: QToolButton, on: bool) -> None:
        """A card ticked goes last in the hook; one unticked leaves it (the ones after move up)."""
        if on and b not in self.order:
            self.order.append(b)
        elif not on and b in self.order:
            self.order.remove(b)
        self._update()

    def _count_changed(self) -> None:
        # One hook clip is longer than a teaser's moments: other moments then fit better.
        if self._length is not None and abs(self.length_for(self.count) - self._length) > 0.05:
            self.shown = []
            self._load(fresh=True)
            return
        # Same moments: keep the ones ticked (in their order), add the strongest others or drop the last ones.
        order = self.order[:self.count]
        order += [b for b, _ in self.cards if b not in order][:self.count - len(order)]
        self._tick(order)

    def more(self) -> None:
        self._load()

    def _update(self) -> None:
        n = len(self.order)
        for b, c in self.cards:
            b.setText((f"✔ {self.order.index(b) + 1}  " if b in self.order else "") + self.names(c))
        self.btn_ok.setEnabled(n > 0)
        if self.cards:
            self.lbl.setText(f"{n} ticked: the hook is {n} clip(s), "
                             f"{sum(c.duration for c in self.chosen()):.1f} s" if n else "Tick at least one moment.")

    def chosen(self) -> list[TimelineClip]:
        """The ticked moments, in the order they were ticked (the hook plays them in that order)."""
        clip_of = {id(b): c for b, c in self.cards}
        return [clip_of[id(b)] for b in self.order if id(b) in clip_of]


def card_name(c: TimelineClip, footage_name: str) -> str:
    score = f"  ★{max(0.0, c.score) * 100:.0f}" if c.score is not None else ""
    return f"{footage_name}\n{fmt_time(c.src_in, frames=False)}, {c.duration:.1f} s{score}"
