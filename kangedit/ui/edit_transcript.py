"""The transcript of the whole edit: what is said in each main-track clip, in timeline order. Click a clip's
row to go to it; drag rows to put the clips in another order (the edit follows); find a clip by its words."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout,
                               QWidget)

from ..models import Project
from ..polish import clip_words
from .theme import fmt_time

PLAYING = QColor(70, 120, 200, 70)


class EditTranscript(QWidget):
    seek = Signal(float)              # timeline time to go to
    reordered = Signal(list)          # the main-track clip ids in their new order
    transcribe = Signal()             # transcribe the edit's footage that has no transcript yet

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self._ids: list[str] = []
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: #9aa0ab;")
        self.info.linkActivated.connect(lambda _: self.transcribe.emit())
        self.search = QLineEdit(placeholderText="Find words…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setToolTip("What is said in each clip, in the order of the edit.\nClick a clip to go to it; drag "
                             "it up or down to move the clip in the edit.")
        self.list.itemClicked.connect(self._clicked)
        self.list.model().rowsMoved.connect(self._moved)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.info)
        lay.addWidget(self.search)
        lay.addWidget(self.list, 1)

    def set_project(self, project: Project) -> None:
        self.project = project
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the rows from the timeline (only when what they show changed)."""
        p = self.project
        rows = []
        missing = 0
        for k, (c, start) in enumerate(zip(p.clips, p.clip_starts())):
            f = p.footage.get(c.footage_id)
            if f is not None and f.has_audio and not f.is_image and f.transcript is None:
                missing += 1
            said = clip_words(c, f)
            name = f.name if f is not None else "?"
            text = said if said else ("(no transcript yet)" if f is not None and f.has_audio and f.transcript is None
                                      else "(nobody speaks)")
            rows.append((c.id, f"{k + 1}.  {fmt_time(start, frames=False)}  ·  {name}\n{text}", bool(said)))
        if [r[0] for r in rows] != self._ids or any(
                self.list.item(i).text() != r[1] for i, r in enumerate(rows) if i < self.list.count()):
            self._ids = [r[0] for r in rows]
            self.list.blockSignals(True)
            self.list.clear()
            for cid, text, spoken in rows:
                it = QListWidgetItem(text)
                it.setData(Qt.UserRole, cid)
                if not spoken:
                    it.setForeground(QBrush(QColor("#8b909a")))
                self.list.addItem(it)
            self.list.blockSignals(False)
            self._filter(self.search.text())
        if not p.clips:
            self.info.setText("The edit's transcript shows here once there are clips on the timeline.")
        elif missing:
            self.info.setText(f"{missing} clip(s) have no transcript yet: <a href='t'>transcribe them</a>.")
        else:
            self.info.setText("What is said in the edit, clip by clip. Click to go there, drag to reorder.")

    def show_time(self, t: float) -> None:
        """Mark the clip playing at timeline time t."""
        idx, _ = self.project.clip_at(t)
        for k in range(self.list.count()):
            it = self.list.item(k)
            it.setBackground(QBrush(PLAYING) if k == idx else QBrush())

    def _filter(self, text: str) -> None:
        q = text.strip().lower()
        for k in range(self.list.count()):
            it = self.list.item(k)
            it.setHidden(bool(q) and q not in it.text().lower())

    def _clicked(self, it: QListWidgetItem) -> None:
        i = self.project.index_of(it.data(Qt.UserRole))
        if i >= 0:
            self.seek.emit(self.project.clip_starts()[i])

    def _moved(self, *_args) -> None:
        ids = [self.list.item(k).data(Qt.UserRole) for k in range(self.list.count())]
        if ids != self._ids:
            self._ids = ids
            self.reordered.emit(ids)
