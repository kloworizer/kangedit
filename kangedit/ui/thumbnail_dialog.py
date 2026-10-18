"""Make a thumbnail: pick one of the edit's best stills (or the frame at the playhead), add big outlined text,
save a JPG."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
                               QSlider, QVBoxLayout)

from .. import thumbnail as th
from ..models import Project
from .theme import fmt_time

PREVIEW_W = 480
FONTS = ("Impact", "Anton", "Arial Black", "Montserrat ExtraBold", "Arial", "DejaVu Sans")


def _qimage(rgb: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(rgb)
    return QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888).copy()


def compose(rgb: np.ndarray, text: str, color: str = "#FFD400", place: str = "bottom",
            size_pct: float = 14.0) -> QImage:
    """The framed still with `text` on it: bold, a thick dark outline, as big as size_pct of the height allows
    (made smaller to fit the width), at the bottom, top or middle."""
    from ..fonts import first_installed
    img = _qimage(rgb).convertToFormat(QImage.Format_RGB32)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return img
    w, h = img.width(), img.height()
    font = QFont(first_installed(*FONTS) or "Arial")
    font.setBold(True)
    px = max(8.0, size_pct / 100.0 * h)
    while True:
        font.setPixelSize(int(px))
        fm = QFontMetricsF(font)
        if max(fm.horizontalAdvance(ln) for ln in lines) <= 0.92 * w or px <= 10:
            break
        px *= 0.92
    line_h = fm.height() * 0.95
    block = line_h * len(lines)
    margin = 0.05 * h
    top = {"top": margin, "middle": (h - block) / 2}.get(place, h - margin - block)
    path = QPainterPath()
    for k, ln in enumerate(lines):
        x = (w - fm.horizontalAdvance(ln)) / 2
        path.addText(QPointF(x, top + k * line_h + fm.ascent()), font, ln)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(0, 0, 0, 230), max(2.0, px * 0.14), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)                        # the outline first, the letters over it
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawPath(path)
    p.end()
    return img


class ThumbnailDialog(QDialog):
    def __init__(self, project: Project, analyses: dict, playhead: Optional[th.Still] = None, parent=None,
                 use_proxies: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Make a thumbnail")
        self.project, self.analyses, self.use_proxies = project, analyses, use_proxies
        self.stills = th.candidates(project, analyses)
        if playhead is not None:
            self.stills.insert(0, playhead)
        self._frame: Optional[np.ndarray] = None
        self._face = None
        self.image: Optional[QImage] = None
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel("The best stills of the edit" + (" (first: the frame at the playhead)"
                                                              if playhead is not None else "")))
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(160, 90))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setMinimumWidth(370)
        for s in self.stills:
            f = project.footage[s.footage_id]
            it = QListWidgetItem(f"{f.name}\n{fmt_time(s.t, frames=False)}")
            pix = self._thumb(s)
            if pix is not None:
                it.setIcon(QIcon(pix))
            self.list.addItem(it)
        self.list.currentRowChanged.connect(self._pick)
        left.addWidget(self.list, 1)
        lay.addLayout(left)
        right = QVBoxLayout()
        self.preview = QLabel("No still to show yet: analyze the footage or make an edit first")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(PREVIEW_W, PREVIEW_W * 9 // 16)
        self.preview.setStyleSheet("background: #111; color: #9aa0ab;")
        right.addWidget(self.preview)
        form = QFormLayout()
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("Big words on the thumbnail (Enter for a second line)")
        self.text.setFixedHeight(56)
        self.text.textChanged.connect(self._render)
        self.cb_size = QComboBox()
        for key, (label, _, _) in th.SIZES.items():
            self.cb_size.addItem(label, key)
        aspect = project.settings.aspect.split(" ")[0]
        self.cb_size.setCurrentIndex(max(0, self.cb_size.findData(aspect if aspect in th.SIZES else "16:9")))
        self.cb_size.currentIndexChanged.connect(self._render)
        self.cb_color = QComboBox()
        for key, label in th.COLORS.items():
            self.cb_color.addItem(label, key)
        self.cb_color.currentIndexChanged.connect(self._render)
        self.cb_place = QComboBox()
        for key, label in th.PLACES.items():
            self.cb_place.addItem(label, key)
        self.cb_place.currentIndexChanged.connect(self._render)
        self.sl_size = QSlider(Qt.Horizontal)
        self.sl_size.setRange(6, 26)
        self.sl_size.setValue(14)
        self.sl_size.valueChanged.connect(self._render)
        self.chk_pop = QCheckBox("Livelier colours and contrast")
        self.chk_pop.setChecked(True)
        self.chk_pop.toggled.connect(self._render)
        form.addRow("Text", self.text)
        form.addRow("Text colour", self.cb_color)
        form.addRow("Text place", self.cb_place)
        form.addRow("Text size", self.sl_size)
        form.addRow("Size", self.cb_size)
        form.addRow(self.chk_pop)
        right.addLayout(form)
        self.lbl = QLabel("")
        self.lbl.setStyleSheet("color: #9aa0ab;")
        right.addWidget(self.lbl)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        self.btn_save = QPushButton("💾 Save JPG…")
        self.btn_save.clicked.connect(self._save_as)
        self.btn_save.setEnabled(False)
        bb.addButton(self.btn_save, QDialogButtonBox.ActionRole)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)
        lay.addLayout(right, 1)
        if self.stills:
            self.list.setCurrentRow(0)

    def _thumb(self, s: th.Still) -> Optional[QPixmap]:
        a = self.analyses.get(s.footage_id)
        if a is None or not len(a.thumbs):
            return None
        k = int(np.argmin(np.abs(np.asarray(a.thumb_times) - s.t))) if len(a.thumb_times) else 0
        return QPixmap.fromImage(_qimage(np.asarray(a.thumbs[min(k, len(a.thumbs) - 1)])))

    def _pick(self, row: int) -> None:
        if not (0 <= row < len(self.stills)):
            return
        from ..reframe import find_face
        s = self.stills[row]
        f = self.project.footage[s.footage_id]
        i = self.project.index_of(s.clip_id)
        grade = self.project.clips[i].grade if i >= 0 else None
        self.setCursor(Qt.WaitCursor)
        try:
            self._frame = th.frame(f, s.t, grade, self.use_proxies)
            face = find_face([self._frame]) if self._frame is not None else None
            self._face = face[:2] if face is not None else None
        finally:
            self.unsetCursor()
        self._render()

    def _render(self) -> None:
        if self._frame is None:
            return
        _, w, h = th.SIZES[self.cb_size.currentData()]
        rgb = th.framed(self._frame, w, h, self._face)
        if self.chk_pop.isChecked():
            rgb = th.pop(rgb)
        self.image = compose(rgb, self.text.toPlainText(), self.cb_color.currentData(), self.cb_place.currentData(),
                             float(self.sl_size.value()))
        pix = QPixmap.fromImage(self.image).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)
        self.btn_save.setEnabled(True)
        self.lbl.setText(f"{w} × {h}" + (" · framed on the face" if self._face is not None else ""))

    def save(self, path: str) -> bool:
        """Save the thumbnail as a JPG (or PNG by its extension)."""
        if self.image is None:
            return False
        ok = self.image.save(path, None, 92)
        self.lbl.setText(f"Saved {path}" if ok else f"Couldn't save {path}")
        return ok

    def _save_as(self) -> None:
        import os
        base = os.path.splitext(os.path.basename(self.project.path or "video"))[0] or "video"
        path, _ = QFileDialog.getSaveFileName(self, "Save the thumbnail", f"{base}_thumbnail.jpg",
                                              "JPEG (*.jpg);;PNG (*.png)")
        if path:
            self.save(path)
