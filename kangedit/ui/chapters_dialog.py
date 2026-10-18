"""YouTube chapters: the edit's chapters as description lines to copy (editable), and title cards for them."""

from __future__ import annotations

from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QPushButton,
                               QVBoxLayout)

from .. import chapters as chap
from ..models import Project

CARD_SECONDS = 3.0       # how long a chapter's title card shows


class ChaptersDialog(QDialog):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YouTube chapters")
        self.project = project
        self.added = 0                      # title cards added
        self.changed = False                # the titles changed (cards added or taken off): the timeline refreshes
        found, how = chap.make(project)
        lay = QVBoxLayout(self)
        if not found:
            note = ("No chapters for this edit: YouTube needs at least 3 of 10 s or more, each starting on a "
                    "clip (too short, or too few clips).")
        elif how == "scenes":
            note = ("A chapter for each scene of the edit (another day or place), named by its day and what is "
                    "said first. Edit the names, then paste the lines into the video's description.")
        else:
            note = ("The edit has fewer than 3 scenes, so it is split into parts at its clips, named by what is "
                    "said first. Edit the names, then paste the lines into the video's description.")
        info = QLabel(note + "\nYouTube shows chapters when the first is at 0:00, there are at least three and "
                             "each is at least 10 s long.")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.text = QPlainTextEdit(chap.text(found))
        self.text.setMinimumSize(460, 220)
        lay.addWidget(self.text, 1)
        self.lbl = QLabel("")
        self.lbl.setStyleSheet("color: #9aa0ab;")
        lay.addWidget(self.lbl)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        self.btn_copy = QPushButton("📋 Copy")
        self.btn_copy.clicked.connect(self.copy)
        self.btn_cards = QPushButton("Add title cards")
        self.btn_cards.setToolTip("Show each chapter's name at its start (the Chapter title template, "
                                  f"{CARD_SECONDS:g} s); the first chapter at 0:00 gets none")
        self.btn_cards.clicked.connect(self.add_cards)
        for b in (self.btn_copy, self.btn_cards):
            b.setEnabled(bool(found))
            bb.addButton(b, QDialogButtonBox.ActionRole)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def chapters(self) -> list:
        return chap.parse(self.text.toPlainText())

    def copy(self) -> None:
        QApplication.clipboard().setText(self.text.toPlainText().strip())
        self.lbl.setText("Copied: paste it into the video's description")

    def add_cards(self) -> None:
        """A title card at each chapter's start (but the first). Cards added before are replaced."""
        from ..titles import make_text
        found = [c for c in self.chapters() if c.start > 0.5 and c.start < self.project.duration - 1.0]
        self.project.checkpoint()
        self.project.texts = [x for x in self.project.texts if x.preset != "chapter_auto"]
        for c in found:
            tc = make_text("chapter", c.start, self.project.duration)
            tc.text, tc.preset = c.title, "chapter_auto"
            tc.duration = min(CARD_SECONDS, self.project.duration - c.start)
            self.project.texts.append(tc)
        self.project.dirty = True
        self.added, self.changed = len(found), True
        self.lbl.setText(f"{len(found)} title card(s) added (Undo takes them off)")
