"""Text tab: add animated titles from templates, list them and edit the selected one."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
                               QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from ..models import Project, TextClip
from ..titles import ALIGNS, ANIM_IN, ANIM_OUT, TEXT_PRESETS
from .dialogs import font_combo, select_font
from .theme import fmt_time

# 3 x 3 position grid: (x, y, align)
POSITIONS = [[(0.06, 0.12, "left"), (0.5, 0.12, "center"), (0.94, 0.12, "right")],
             [(0.06, 0.5, "left"), (0.5, 0.5, "center"), (0.94, 0.5, "right")],
             [(0.06, 0.82, "left"), (0.5, 0.82, "center"), (0.94, 0.82, "right")]]


class TextPanel(QScrollArea):
    add_requested = Signal(str)        # template key: add at the playhead
    selected = Signal(str)             # title id ("" = none)
    before_change = Signal(str)        # undo key, before a field changes
    changed = Signal()                 # a field of the selected title changed
    delete_requested = Signal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.text: Optional[TextClip] = None
        self.playhead: Callable[[], float] = lambda: 0.0   # timeline time (set by the main window)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)    # dragged narrower than it: scroll sideways
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 14, 6)    # room for the scroll bar on the right

        g = QGroupBox("Add a title (at the playhead)")
        grid = QGridLayout(g)
        for i, (key, cfg) in enumerate(TEXT_PRESETS.items()):
            b = QPushButton(cfg["label"])
            b.setToolTip(f"{cfg['text']!r} · {ANIM_IN[cfg.get('anim_in', 'fade')]} / "
                         f"{ANIM_OUT[cfg.get('anim_out', 'fade')]}")
            b.clicked.connect(lambda _=False, k=key: self.add_requested.emit(k))
            grid.addWidget(b, i // 2, i % 2)
        lay.addWidget(g)

        self.list = QListWidget()
        self.list.setMaximumHeight(130)
        self.list.currentItemChanged.connect(self._list_changed)
        self.lbl_empty = QLabel("No titles yet. Pick a template above, or right-click the timeline's text track.")
        self.lbl_empty.setWordWrap(True)
        self.lbl_empty.setStyleSheet("color: #8b909a;")
        lay.addWidget(self.list)
        lay.addWidget(self.lbl_empty)

        self.editor = QGroupBox("Selected title")
        f = QFormLayout(self.editor)
        # The hint wraps here instead of being the box's title, which can't wrap and would widen the panel.
        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        f.addRow(self.lbl_hint)
        self.ed_text = QPlainTextEdit()
        self.ed_text.setMaximumHeight(64)
        self.ed_text.setPlaceholderText("Type the text (Enter for a new line)")
        self._text_timer = QTimer(self, singleShot=True, interval=250)
        self._text_timer.timeout.connect(lambda: self._set("text", self.ed_text.toPlainText()))
        self.ed_text.textChanged.connect(self._text_timer.start)
        f.addRow(self.ed_text)

        self.cb_font = font_combo("Arial")
        self.cb_font.currentTextChanged.connect(lambda v: v.strip() and self._set("font", v.strip()))
        f.addRow("Font", self.cb_font)
        self.sp_size = QDoubleSpinBox(decimals=1, singleStep=0.5, suffix=" % of height")
        self.sp_size.setRange(1.5, 30)
        self.sp_size.valueChanged.connect(lambda v: self._set("size_pct", v))
        f.addRow("Size", self.sp_size)

        row = QHBoxLayout()
        self.b_color = QPushButton()
        self.b_color.clicked.connect(lambda: self._pick("color"))
        self.chk_bold, self.chk_italic, self.chk_outline = QCheckBox("Bold"), QCheckBox("Italic"), QCheckBox("Outline")
        for chk, key in ((self.chk_bold, "bold"), (self.chk_italic, "italic"), (self.chk_outline, "outline")):
            chk.toggled.connect(lambda v, k=key: self._set(k, v))
        row.addWidget(self.chk_bold)
        row.addWidget(self.chk_italic)
        row.addWidget(self.chk_outline)
        row.addStretch(1)
        f.addRow("Colour", self.b_color)
        f.addRow("Style", row)

        row = QHBoxLayout()
        self.chk_box = QCheckBox("Background box")
        self.chk_box.toggled.connect(lambda v: self._set("box", v))
        self.b_box_color = QPushButton()
        self.b_box_color.clicked.connect(lambda: self._pick("box_color"))
        self.sp_box_op = QDoubleSpinBox(decimals=0, suffix=" %")
        self.sp_box_op.setRange(0, 100)
        self.sp_box_op.setToolTip("Box opacity")
        self.sp_box_op.valueChanged.connect(lambda v: self._set("box_opacity", v))
        row.addWidget(self.chk_box)
        row.addWidget(self.b_box_color)
        row.addWidget(self.sp_box_op)
        f.addRow("Box", row)

        pos = QGridLayout()
        pos.setSpacing(2)
        for r, line in enumerate(POSITIONS):
            for c, (x, y, align) in enumerate(line):
                b = QToolButton(text="●")
                b.setFixedSize(26, 20)
                b.setToolTip(f"{['Top', 'Middle', 'Bottom'][r]} {['left', 'centre', 'right'][c]}")
                b.clicked.connect(lambda _=False, x=x, y=y, a=align: self._set_pos(x, y, a))
                pos.addWidget(b, r, c)
        pos_row = QVBoxLayout()     # the grid above X / Y: fits a narrow panel
        pos.setColumnStretch(3, 1)
        pos_row.addLayout(pos)
        xy = QFormLayout()
        self.sp_x = QDoubleSpinBox(decimals=0, suffix=" %")
        self.sp_y = QDoubleSpinBox(decimals=0, suffix=" %")
        for sp, key in ((self.sp_x, "x"), (self.sp_y, "y")):
            sp.setRange(0, 100)
            sp.valueChanged.connect(lambda v, k=key: self._set(k, v / 100.0))
        self.cb_align = QComboBox()
        for k, label in ALIGNS.items():
            self.cb_align.addItem(label, k)
        self.cb_align.currentIndexChanged.connect(lambda _: self._set("align", self.cb_align.currentData()))
        xy.addRow("X", self.sp_x)
        xy.addRow("Y", self.sp_y)
        xy.addRow("Align", self.cb_align)
        pos_row.addLayout(xy)
        f.addRow("Position", pos_row)
        hint = QLabel("Tip: drag the text in the viewer to place it.", wordWrap=True)
        hint.setStyleSheet("color: #8b909a;")
        f.addRow("", hint)

        self.cb_in, self.cb_out = QComboBox(), QComboBox()
        for cb, table, key in ((self.cb_in, ANIM_IN, "anim_in"), (self.cb_out, ANIM_OUT, "anim_out")):
            for k, label in table.items():
                cb.addItem(label, k)
            cb.currentIndexChanged.connect(lambda _, cb=cb, k=key: self._set(k, cb.currentData()))
        self.sp_anim = QDoubleSpinBox(decimals=2, singleStep=0.1, suffix=" s")
        self.sp_anim.setRange(0.1, 5)
        self.sp_anim.valueChanged.connect(lambda v: self._set("anim_dur", v))
        f.addRow("Animate in", self.cb_in)
        f.addRow("Animate out", self.cb_out)
        f.addRow("Animation length", self.sp_anim)

        self.sp_start = QDoubleSpinBox(decimals=2, singleStep=0.1, suffix=" s")
        self.sp_start.setRange(0, 1e5)
        self.sp_start.valueChanged.connect(lambda v: self._set("start", v))
        self.sp_dur = QDoubleSpinBox(decimals=2, singleStep=0.1, suffix=" s")
        self.sp_dur.setRange(0.25, 3600)
        self.sp_dur.valueChanged.connect(lambda v: self._set("duration", v))
        f.addRow("Starts at", self.sp_start)
        f.addRow("Lasts", self.sp_dur)
        b_del = QPushButton("🗑 Delete title")
        b_del.clicked.connect(lambda: self.text and self.delete_requested.emit(self.text.id))
        f.addRow(b_del)
        lay.addWidget(self.editor)
        lay.addStretch(1)
        self._fields = [self.ed_text, self.cb_font, self.sp_size, self.chk_bold, self.chk_italic, self.chk_outline,
                        self.chk_box, self.sp_box_op, self.sp_x, self.sp_y, self.cb_align, self.cb_in, self.cb_out,
                        self.sp_anim, self.sp_start, self.sp_dur]
        self.refresh()

    # ------------------------------------------------------------------ public
    def set_project(self, project: Project):
        self.project = project
        self.text = None
        self.refresh()

    def refresh(self, select: Optional[str] = None):
        """Rebuild the list (keeping or changing the selection) and show the selected title's fields."""
        if select is not None:
            self.text = next((x for x in self.project.texts if x.id == select), None)
        elif self.text is not None and self.text not in self.project.texts:
            self.text = next((x for x in self.project.texts if x.id == self.text.id), None)
        self.list.blockSignals(True)
        self.list.clear()
        for x in sorted(self.project.texts, key=lambda x: x.start):
            it = QListWidgetItem(f"{fmt_time(x.start, frames=False)}   {x.text.splitlines()[0] if x.text else '(empty)'}")
            it.setData(Qt.UserRole, x.id)
            self.list.addItem(it)
            if self.text is not None and x.id == self.text.id:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)
        self.lbl_empty.setVisible(not self.project.texts)
        self.list.setVisible(bool(self.project.texts))
        self._load()

    def showEvent(self, e):
        """Opening the tab with no title selected selects the one at the playhead (else the nearest), so
        its font, size and style can be changed straight away."""
        super().showEvent(e)
        if self.text is None and self.project.texts:
            t = self.playhead()
            here = [x for x in self.project.texts if x.start <= t < x.start + x.duration]
            tc = here[0] if here else min(self.project.texts, key=lambda x: abs(x.start - t))
            self.refresh(select=tc.id)
            self.selected.emit(tc.id)

    def focus_text(self):
        self.ed_text.setFocus()
        self.ed_text.selectAll()

    # ------------------------------------------------------------------ internals
    def _list_changed(self, item, _prev=None):
        tid = item.data(Qt.UserRole) if item is not None else ""
        self.text = next((x for x in self.project.texts if x.id == tid), None)
        self._load()
        self.selected.emit(tid or "")

    def _load(self):
        tc = self.text
        self.editor.setEnabled(tc is not None)
        self.lbl_hint.setText("" if tc else "Select a title on the timeline or above." if self.project.texts else
                              "Add a title above, then change its font, size and style here.")
        self.lbl_hint.setVisible(tc is None)
        if tc is None:
            return
        for w in self._fields:
            w.blockSignals(True)
        if self.ed_text.toPlainText() != tc.text:
            self.ed_text.setPlainText(tc.text)
        select_font(self.cb_font, tc.font)
        self.sp_size.setValue(tc.size_pct)
        self.chk_bold.setChecked(tc.bold)
        self.chk_italic.setChecked(tc.italic)
        self.chk_outline.setChecked(tc.outline)
        self.chk_outline.setEnabled(not tc.box)
        self.chk_box.setChecked(tc.box)
        self.sp_box_op.setValue(tc.box_opacity)
        self.sp_box_op.setEnabled(tc.box)
        self.b_box_color.setEnabled(tc.box)
        self.sp_x.setValue(tc.x * 100)
        self.sp_y.setValue(tc.y * 100)
        self.cb_align.setCurrentIndex(max(0, self.cb_align.findData(tc.align)))
        self.cb_in.setCurrentIndex(max(0, self.cb_in.findData(tc.anim_in)))
        self.cb_out.setCurrentIndex(max(0, self.cb_out.findData(tc.anim_out)))
        self.sp_anim.setValue(tc.anim_dur)
        self.sp_start.setValue(tc.start)
        self.sp_dur.setValue(tc.duration)
        for w in self._fields:
            w.blockSignals(False)
        for b, col in ((self.b_color, tc.color), (self.b_box_color, tc.box_color)):
            b.setText(col)
            b.setStyleSheet(f"background: {col}; color: {'black' if QColor(col).lightness() > 128 else 'white'};")

    def _set(self, key: str, value):
        tc = self.text
        if tc is None or getattr(tc, key) == value:
            return
        self.before_change.emit(f"text:{tc.id}:{key}")
        setattr(tc, key, value)
        if key in ("box", "color", "box_color"):
            self._load()
        if key in ("text", "start"):
            self.refresh()
        self.changed.emit()

    def _set_pos(self, x: float, y: float, align: str):
        tc = self.text
        if tc is None:
            return
        self.before_change.emit(f"text:{tc.id}:pos")
        tc.x, tc.y, tc.align = x, y, align
        self._load()
        self.changed.emit()

    def _pick(self, key: str):
        tc = self.text
        if tc is None:
            return
        c = QColorDialog.getColor(QColor(getattr(tc, key)), self, "Text colour" if key == "color" else "Box colour")
        if c.isValid():
            self._set(key, c.name())
