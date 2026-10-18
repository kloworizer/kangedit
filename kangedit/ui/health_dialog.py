"""The health check before an export: each issue with its fix, and Export (not while an error is left)."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFrame, QGridLayout, QLabel, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from ..health import ERROR, INFO, Issue

ICONS = {ERROR: "⛔", "warning": "⚠️", INFO: "ℹ️"}


class HealthDialog(QDialog):
    """check() -> the issues now; fix(key) -> what it did (the issues are checked again after)."""

    def __init__(self, check: Callable[[], list], fix: Callable[[str], str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Before the export")
        self.check, self.fix = check, fix
        self.issues: list[Issue] = []
        lay = QVBoxLayout(self)
        self.head = QLabel("")
        self.head.setWordWrap(True)
        lay.addWidget(self.head)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.NoFrame)
        self.area.setMinimumSize(560, 200)
        lay.addWidget(self.area, 1)
        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color: #9aa0ab;")
        lay.addWidget(self.lbl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_export = bb.button(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.refresh()

    def refresh(self) -> None:
        self.issues = self.check()
        box = QWidget()
        grid = QGridLayout(box)
        for k, it in enumerate(self.issues):
            icon = QLabel(ICONS.get(it.level, ""))
            text = QLabel(it.text)
            text.setWordWrap(True)
            grid.addWidget(icon, k, 0)
            grid.addWidget(text, k, 1)
            if it.fix:
                b = QPushButton(it.fix_label or "Fix")
                b.clicked.connect(lambda _=False, key=it.fix: self.apply(key))
                grid.addWidget(b, k, 2)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(len(self.issues), 1)
        self.area.setWidget(box)
        errors = sum(1 for i in self.issues if i.level == ERROR)
        self.btn_export.setEnabled(errors == 0)
        self.btn_export.setText("Export anyway" if self.issues and not errors else "Export")
        if errors:
            self.head.setText(f"{errors} problem(s) stop the export: fix them first.")
        elif self.issues:
            self.head.setText("The export can go ahead, but have a look at these first.")
        else:
            self.head.setText("Everything is ready for the export.")

    def apply(self, key: str) -> None:
        msg = self.fix(key)
        self.refresh()
        if msg:
            self.lbl.setText(msg)
