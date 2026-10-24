"""The transcript beside a clip being slipped: what is said in its footage, a sentence per row, the ones the
clip shows highlighted and following the drag. Clicking a sentence slips the clip to start on it."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                               QVBoxLayout, QWidget)

from ..polish import sentence_spans
from .theme import fmt_time

SHOWN = QColor(70, 120, 200, 90)     # a sentence the clip shows
SLIM_HEIGHT = 50                     # the panel's height: about two sentences (the timeline's lanes stay in view)


def sentence_rows(words: list[dict]) -> list[tuple[float, float, str]]:
    """(start, end, text) of each sentence of a transcript."""
    out = []
    for i, j in sentence_spans(words):
        text = " ".join(str(w.get("w", "")).strip() for w in words[i:j + 1]).strip()
        if text:
            out.append((float(words[i]["s"]), float(words[j]["e"]), text))
    return out


class SlipTranscript(QWidget):
    slip_to = Signal(str, float)      # clip id, new source in point (a sentence's start)
    transcribe = Signal(str)          # footage id to transcribe

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clip_id = ""
        self.footage_id = ""
        self.rows: list[tuple[float, float, str]] = []
        self._range = (0.0, 0.0)
        # One slim row (the lanes keep their room): what it is, the clip's length while it is trimmed and the
        # search on the left; two sentences at a time on the right.
        self.title = QLabel("")
        self.title.setStyleSheet("font-weight: 600;")
        self.length = QLabel("")
        self.length.setStyleSheet("color: #ffd34d; font-weight: 600;")
        self.length.hide()
        self.search = QLineEdit(placeholderText="Find words…")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedHeight(22)
        self.search.textChanged.connect(self._filter)
        self.btn_transcribe = QPushButton("Transcribe this file")
        self.btn_transcribe.clicked.connect(lambda: self.transcribe.emit(self.footage_id))
        self.btn_transcribe.hide()
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(1)
        side.addWidget(self.title)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.length)
        row.addWidget(self.btn_transcribe)
        row.addWidget(self.search, 1)
        side.addLayout(row)
        side.addStretch(1)
        left = QWidget()
        left.setLayout(side)
        left.setFixedWidth(250)
        f = left.font()
        f.setPointSizeF(max(7.0, f.pointSizeF() * 0.9))
        left.setFont(f)
        self.list = QListWidget()
        self.list.setToolTip("What is said in the clip's footage. Highlighted: what the clip shows now.\n"
                             "Click a sentence to slip the clip to start on it (its length stays).")
        self.list.setAlternatingRowColors(False)
        self.list.setFixedHeight(SLIM_HEIGHT)
        self.list.itemClicked.connect(self._clicked)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 1, 4, 1)
        lay.setSpacing(6)
        lay.addWidget(left)
        lay.addWidget(self.list, 1)
        self.setMaximumHeight(SLIM_HEIGHT + 4)

    def show_length(self, duration: Optional[float], before: Optional[float] = None) -> None:
        """The clip's length while it is trimmed, and how much it changed (None: not trimming)."""
        if duration is None:
            self.length.hide()
            return
        text = f"⏱ {duration:.2f} s"
        if before is not None and abs(duration - before) >= 0.005:
            text += f" ({duration - before:+.2f})"
            self.length.setToolTip(f"The clip is {duration:.2f} s long, was {before:.2f} s")
        self.length.setText(text)
        self.length.show()

    def _set_title(self, text: str) -> None:
        """One line (shortened to fit; the whole text in its tooltip)."""
        self.title.setToolTip(text)
        self.title.setText(self.title.fontMetrics().elidedText(text, Qt.ElideRight, 244))

    def show_clip(self, clip, footage) -> None:
        """Show a main-track clip's transcript (None: nothing selected)."""
        self.clip_id = clip.id if clip is not None else ""
        self.footage_id = footage.id if footage is not None else ""
        words = (footage.transcript or []) if footage is not None else []
        self.rows = sentence_rows(words)
        self.list.clear()
        if clip is None or footage is None:
            self._set_title("🗣 Select a clip to see what is said in it")
            self.btn_transcribe.hide()
            return
        if footage.transcript is None or not footage.has_audio:
            self._set_title(f"🗣 {footage.name}: no transcript yet" if footage.has_audio
                               else f"🗣 {footage.name} has no sound")
            self.btn_transcribe.setVisible(footage.has_audio and not footage.is_image)
            return
        self.btn_transcribe.hide()
        if not self.rows:
            self._set_title(f"🗣 {footage.name}: nobody speaks in it")
            return
        self._set_title(f"🗣 {footage.name}: click a sentence to slip the clip to it")
        for a, b, text in self.rows:
            it = QListWidgetItem(f"{fmt_time(a, frames=False)}   {text}")
            it.setData(Qt.UserRole, a)
            self.list.addItem(it)
        self._filter(self.search.text())
        self.follow(clip.id, clip.src_in, clip.src_out)

    def follow(self, clip_id: str, src_in: float, src_out: Optional[float] = None) -> None:
        """Highlight the sentences source range src_in..src_out shows (slipping: as it is dragged)."""
        if clip_id != self.clip_id or not self.rows:
            return
        if src_out is None:
            src_out = src_in + (self._range[1] - self._range[0])
        self._range = (src_in, src_out)
        first = None
        bold = QFont()
        bold.setBold(True)
        for k, (a, b, _) in enumerate(self.rows):
            it = self.list.item(k)
            shown = a < src_out and b > src_in
            it.setBackground(QBrush(SHOWN) if shown else QBrush())
            it.setFont(bold if shown else QFont())
            if shown and first is None:
                first = it
        if first is not None:
            self.list.scrollToItem(first, QListWidget.PositionAtTop)

    def _filter(self, text: str) -> None:
        q = text.strip().lower()
        for k in range(self.list.count()):
            it = self.list.item(k)
            it.setHidden(bool(q) and q not in it.text().lower())

    def _clicked(self, it: QListWidgetItem) -> None:
        if self.clip_id:
            self.slip_to.emit(self.clip_id, float(it.data(Qt.UserRole)))
