"""Auto edit from a script: the dialog (script, footage, matching settings and a live preview)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QSlider, QSplitter,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import scriptedit
from ..models import Project
from .theme import fmt_time

PLACEHOLDER = ("Paste or type the script here, one line or sentence after another:\n\n"
               "HOST: Welcome back to the channel!\n"
               "Today we're looking at the new camera. [beat]\n"
               "The battery lasts all day, which is great.\n\n"
               "Speaker labels (HOST:) and [directions] are ignored.")


class ScriptEditDialog(QDialog):
    """settings() -> {"script", "footage", "threshold", "prefer", "pad_in", "pad_out", "merge", "replace"}."""

    def __init__(self, project: Project, selected: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Edit from script")
        self.project = project
        last = dict(project.settings.script or {})
        self.resize(980, 640)

        self.ed_script = QPlainTextEdit()
        self.ed_script.setPlaceholderText(PLACEHOLDER)
        self.ed_script.setPlainText(last.get("script", ""))

        self.lst_footage = QListWidget()
        videos = [f for f in project.footage.values() if not f.is_image and f.has_audio]
        chosen = set(selected) & {f.id for f in videos} or set(last.get("footage") or []) & {f.id for f in videos}
        for f in videos:
            it = QListWidgetItem(f"{f.name}  " + ("· transcribed ✓" if f.transcript is not None else
                                                   "· needs a transcript"))
            it.setData(Qt.UserRole, f.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if (not chosen or f.id in chosen) else Qt.Unchecked)
            self.lst_footage.addItem(it)

        self.sl_threshold = QSlider(Qt.Horizontal)
        self.sl_threshold.setRange(30, 100)
        self.sl_threshold.setValue(int(round(100 * last.get("threshold", scriptedit.DEFAULT_THRESHOLD))))
        self.lbl_threshold = QLabel()
        self.cb_prefer = QComboBox()
        self.cb_prefer.addItem("Best match", "best")
        self.cb_prefer.addItem("Last good take", "last")
        self.cb_prefer.setCurrentIndex(max(0, self.cb_prefer.findData(last.get("prefer", "best"))))
        self.cb_prefer.setToolTip("A line said in several takes: the one that matches the script best, or the last "
                                  f"take that matches nearly as well (within {scriptedit.LAST_TAKE_MARGIN:.0%})")
        self.sp_pad_in = QDoubleSpinBox(decimals=2, singleStep=0.05, suffix=" s")
        self.sp_pad_out = QDoubleSpinBox(decimals=2, singleStep=0.05, suffix=" s")
        for sp, key, default in ((self.sp_pad_in, "pad_in", 0.15), (self.sp_pad_out, "pad_out", 0.25)):
            sp.setRange(0.0, 2.0)
            sp.setValue(float(last.get(key, default)))
            sp.setToolTip("Room kept before / after the words (never into the next words)")
        self.chk_merge = QCheckBox("Join lines said one after the other in the same take")
        self.chk_merge.setChecked(bool(last.get("merge", True)))
        self.chk_replace = QCheckBox("Replace the timeline (else add after it)")
        self.chk_replace.setChecked(bool(last.get("replace", True)))

        form = QFormLayout()
        th = QHBoxLayout()
        th.addWidget(self.sl_threshold, 1)
        th.addWidget(self.lbl_threshold)
        form.addRow("Similarity threshold", th)
        form.addRow("Several takes", self.cb_prefer)
        pads = QHBoxLayout()
        pads.addWidget(QLabel("before"))
        pads.addWidget(self.sp_pad_in)
        pads.addWidget(QLabel("after"))
        pads.addWidget(self.sp_pad_out)
        pads.addStretch(1)
        form.addRow("Padding", pads)
        form.addRow(self.chk_merge)
        form.addRow(self.chk_replace)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("<b>Script</b>"))
        lv.addWidget(self.ed_script, 3)
        lv.addWidget(QLabel("<b>Footage to search</b>"))
        lv.addWidget(self.lst_footage, 1)
        lv.addLayout(form)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["#", "Line", "Match", "Footage", "At", "Heard"])
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 30)
        self.tree.setColumnWidth(1, 220)
        self.tree.setColumnWidth(2, 50)
        self.tree.setColumnWidth(3, 110)
        self.tree.setColumnWidth(4, 60)
        self.lbl_summary = QLabel()
        self.lbl_summary.setWordWrap(True)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("<b>Preview</b>"))
        rv.addWidget(self.tree, 1)
        rv.addWidget(self.lbl_summary)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 560])
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Create edit")
        bb.button(QDialogButtonBox.Ok).setObjectName("primary")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        lay.addWidget(bb)

        self._timer = QTimer(self, singleShot=True, interval=350)
        self._timer.timeout.connect(self.update_preview)
        self.ed_script.textChanged.connect(self._timer.start)
        self.sl_threshold.valueChanged.connect(self._threshold_changed)
        for w in (self.cb_prefer,):
            w.currentIndexChanged.connect(lambda _: self._timer.start())
        self.lst_footage.itemChanged.connect(lambda _: self._timer.start())
        self._threshold_changed()

    def _threshold_changed(self):
        self.lbl_threshold.setText(f"{self.sl_threshold.value()} %")
        self._timer.start()

    def footage_ids(self) -> list[str]:
        out = []
        for i in range(self.lst_footage.count()):
            it = self.lst_footage.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def settings(self) -> dict:
        return {"script": self.ed_script.toPlainText(), "footage": self.footage_ids(),
                "threshold": self.sl_threshold.value() / 100.0, "prefer": self.cb_prefer.currentData(),
                "pad_in": self.sp_pad_in.value(), "pad_out": self.sp_pad_out.value(),
                "merge": self.chk_merge.isChecked(), "replace": self.chk_replace.isChecked()}

    def update_preview(self):
        """Match the script now when every chosen file has its transcript (else say what is still needed)."""
        s = self.settings()
        self.tree.clear()
        lines = scriptedit.split_script(s["script"])
        foot = [self.project.footage[i] for i in s["footage"] if i in self.project.footage]
        todo = [f for f in foot if f.transcript is None]
        if not lines:
            self.lbl_summary.setText("Type or paste the script on the left.")
            return
        if todo:
            for k, line in enumerate(lines):
                self.tree.addTopLevelItem(QTreeWidgetItem([str(k + 1), line, "", "", "", ""]))
            self.lbl_summary.setText(f"{len(lines)} lines. {len(todo)} file(s) need a transcript first: they are "
                                     "transcribed when you click Create edit, then the script is matched.")
            return
        result = scriptedit.match_script(lines, foot, s["threshold"], s["prefer"])
        names = {f.id: f.name for f in foot}
        for m in result:
            it = QTreeWidgetItem([str(m.index + 1), m.line, f"{m.score:.0%}",
                                  names.get(m.footage_id, "—"), fmt_time(m.src_in, frames=False) if m.found else "",
                                  m.heard])
            if not m.found:
                for col in range(6):
                    it.setForeground(col, QBrush(QColor("#e07a5f")))
                it.setToolTip(1, f"Not found: the closest is {m.score:.0%} similar"
                                 + (f" (“{m.heard}”)" if m.heard else ""))
            elif m.takes > 1:
                it.setToolTip(2, f"{m.takes} takes match")
            self.tree.addTopLevelItem(it)
        found = sum(m.found for m in result)
        self.lbl_summary.setText(f"{found} of {len(result)} lines found"
                                 + (f"; {len(result) - found} not found (in red): lower the threshold, or check the "
                                    "script and the footage" if found < len(result) else "."))
