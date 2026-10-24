"""File → Version history: the kept versions of the project, what changed since each, Restore / Open as copy."""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QLabel, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout)

from .. import history
from ..models import Project
from .theme import fmt_time


class HistoryDialog(QDialog):
    """Lists the versions of `project` (newest first). After exec(), `choice` is ("restore" | "copy", entry)
    or None when closed."""

    RESTORE, COPY = "restore", "copy"

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Version history")
        self.project = project
        self.choice: Optional[tuple[str, dict]] = None
        self.entries = history.versions(project.path)
        lay = QVBoxLayout(self)
        self.head = QLabel("")
        self.head.setWordWrap(True)
        lay.addWidget(self.head)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Saved", "", "Clips", "Length", "Note"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setMinimumSize(620, 260)
        now = time.time()
        for v in self.entries:
            it = QTreeWidgetItem([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v.get("time", 0.0))),
                                  history.ago(v.get("time", 0.0), now), str(v.get("clips", 0)),
                                  fmt_time(v.get("duration", 0.0), frames=False), v.get("label", "")])
            it.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
            self.tree.addTopLevelItem(it)
        for col in range(4):
            self.tree.resizeColumnToContents(col)
        self.tree.currentItemChanged.connect(lambda *_: self._show_diff())
        self.tree.itemActivated.connect(lambda *_: self._choose(self.RESTORE))   # Enter / double-click
        lay.addWidget(self.tree, 1)
        self.diff = QLabel("")
        self.diff.setWordWrap(True)
        self.diff.setStyleSheet("color: #9aa0ab;")
        lay.addWidget(self.diff)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        self.btn_restore = bb.addButton("Restore", QDialogButtonBox.AcceptRole)
        self.btn_restore.setToolTip("Bring this version back into the window (the current state is kept as a "
                                    "version first, so this can be undone). Save to keep it in the file.")
        self.btn_copy = bb.addButton("Open as copy", QDialogButtonBox.ActionRole)
        self.btn_copy.setToolTip("Open this version as a new untitled project, leaving the project file alone")
        self.btn_restore.clicked.connect(lambda: self._choose(self.RESTORE))
        self.btn_copy.clicked.connect(lambda: self._choose(self.COPY))
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if not project.path:
            self.head.setText("This project hasn't been saved yet. Once it is, a version is kept each time you "
                              "save (and every few minutes of auto-saving), so you can go back to it here.")
        elif not self.entries:
            self.head.setText("No earlier versions yet. A version is kept each time you save (and every few "
                              "minutes of auto-saving), and before big changes such as a new auto edit.")
        else:
            self.head.setText(f"{len(self.entries)} version(s) of this project, newest first. Restore brings one "
                              "back into this window; Open as copy opens it as a new untitled project.")
        for b in (self.btn_restore, self.btn_copy):
            b.setEnabled(bool(self.entries))
        if self.entries:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
            self.btn_restore.setDefault(True)
            self.tree.setFocus()

    def selected(self) -> Optional[dict]:
        k = self.tree.indexOfTopLevelItem(self.tree.currentItem()) if self.tree.currentItem() else -1
        return self.entries[k] if 0 <= k < len(self.entries) else None

    def _show_diff(self) -> None:
        v = self.selected()
        if v is None:
            self.diff.setText("")
            return
        try:
            old = history.load(self.project.path, v)
        except (OSError, ValueError, KeyError) as e:
            self.diff.setText(f"This version can't be read: {e}")
            return
        self.diff.setText(history.diff_summary(old, self.project))

    def _choose(self, what: str) -> None:
        v = self.selected()
        if v is not None:
            self.choice = (what, v)
            self.accept()
