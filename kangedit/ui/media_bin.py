"""Media bin: imported footage with status, thumbnails and per-footage actions.

Two views of the same footage: a detail list (columns) and a thumbnail grid. In the grid, moving the
mouse across a thumbnail scrubs through the clip (analysis keeps one thumbnail per second). A green tick
marks footage the edit uses (main track or video layers). Drag footage onto the timeline to add it.

The row above the list picks a bin (see bins.py) and filters by kind (video / pictures) and picture shape;
right-click for bins: new, move to, by date, rename, delete.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import QEvent, QMimeData, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QInputDialog, QListView, QListWidget,
                               QListWidgetItem, QMenu, QPushButton, QStackedWidget, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import bins as binlib
from .. import images
from .. import lut as lutlib
from .. import spherical
from ..analysis import Analysis
from ..color import LOG_CHOICES, lut_profile, profile_label
from ..config import VIDEO_EXTENSIONS, open_path, settings
from ..models import Project
from .theme import fmt_time
from .timeline import FOOTAGE_MIME

GRID_COLUMNS = 2                          # thumbnails per row; they grow with the bin's width


def _pixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img)


USED_COLOR = "#3ecf7a"


def _badge(pm: QPixmap, used: int) -> QPixmap:
    """The thumbnail with a green tick (and how many times) when the edit uses the footage."""
    if not used:
        return pm
    pm = QPixmap(pm)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    h = max(12.0, pm.height() * 0.3)
    f = QFont()
    f.setPixelSize(int(h * 0.75))
    f.setBold(True)
    p.setFont(f)
    text = "✓" if used == 1 else f"✓{used}"
    w = max(h, p.fontMetrics().horizontalAdvance(text) + h * 0.5)
    r = QRectF(pm.width() - w - 2, 2, w, h)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(USED_COLOR))
    p.drawRoundedRect(r, h / 2, h / 2)
    p.setPen(QColor("#08210f"))
    p.drawText(r, Qt.AlignCenter, text)
    p.end()
    return pm


def _mime(ids: list[str]) -> QMimeData:
    md = QMimeData()
    md.setData(FOOTAGE_MIME, "\n".join(ids).encode("utf-8"))
    return md


class _Tree(QTreeWidget):
    def mimeTypes(self):   # noqa: N802 (Qt name)
        return [FOOTAGE_MIME]

    def mimeData(self, items):   # noqa: N802
        return _mime([it.data(0, Qt.UserRole) for it in items])


class _Grid(QListWidget):
    """Thumbnail grid: fixed number of columns, hovering scrubs through a clip's thumbnails."""
    resized = Signal()
    hover = Signal(object, float)        # item (or None), horizontal position over its icon 0..1

    def __init__(self):
        super().__init__()
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)
        self.setUniformItemSizes(True)
        self.setSpacing(6)
        self.setMouseTracking(True)
        self.viewport().installEventFilter(self)

    def mimeTypes(self):   # noqa: N802 (Qt name)
        return [FOOTAGE_MIME]

    def mimeData(self, items):   # noqa: N802
        return _mime([it.data(Qt.UserRole) for it in items])

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.resized.emit()

    def eventFilter(self, obj, e):
        if e.type() == QEvent.MouseMove:
            pos = e.position().toPoint()
            it = self.itemAt(pos)
            if it is not None:
                r = self.visualItemRect(it)
                self.hover.emit(it, (pos.x() - r.left()) / max(1, r.width()))
            else:
                self.hover.emit(None, 0.0)
        elif e.type() == QEvent.Leave:
            self.hover.emit(None, 0.0)
        return super().eventFilter(obj, e)


class MediaBin(QWidget):
    import_requested = Signal()
    files_dropped = Signal(list)
    action = Signal(str, list)            # action name, footage ids
    log_profile = Signal(list, str)       # footage ids, profile key
    edit_started = Signal()               # bins are about to change (an undo step)
    edited = Signal(str)                  # bins changed (a status message)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.analyses: dict[str, Analysis] = {}
        self._qs = settings()
        self._scrubbing = None            # grid item whose icon shows a scrubbed frame
        self._usage: dict[str, int] = {}  # footage id: times the edit uses it
        self.tree = _Tree()
        self.tree.setHeaderLabels(["Footage", "In edit", "Duration", "Format", "Camera", "Log", "Status", "Bin"])
        self.tree.headerItem().setToolTip(1, "✓ the edit uses this footage (how many times), — not used yet")
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setIconSize(QSize(64, 36))
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemDoubleClicked.connect(lambda *_: self.action.emit("play", self.selected_ids()[:1]))
        self.tree.setColumnWidth(0, 200)
        self.tree.setColumnWidth(1, 52)
        self.tree.setColumnWidth(2, 60)
        self.tree.setColumnWidth(3, 120)
        self.tree.setColumnWidth(4, 90)
        self.tree.setColumnWidth(5, 70)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.DragOnly)

        self.grid = _Grid()
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._menu)
        self.grid.itemDoubleClicked.connect(lambda *_: self.action.emit("play", self.selected_ids()[:1]))
        self._icons_timer = QTimer(self, singleShot=True, interval=120)
        self._icons_timer.timeout.connect(self._refresh_grid_icons)
        self.grid.resized.connect(self._apply_thumb_size)
        self.grid.hover.connect(self._scrub)
        self.grid.setDragEnabled(True)
        self.grid.setDragDropMode(QAbstractItemView.DragOnly)
        self._apply_thumb_size()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.tree)
        self.stack.addWidget(self.grid)

        b_import = QPushButton("＋ Import media")
        b_import.setToolTip("Add video files and images (JPG, PNG, GIF, BMP, WebP, TIFF)")
        b_import.setObjectName("primary")
        b_import.clicked.connect(self.import_requested.emit)
        b_remove = QPushButton("Remove")
        b_remove.clicked.connect(lambda: self.action.emit("remove", self.selected_ids()))
        self.btn_details = QToolButton(text="☰", checkable=True, autoExclusive=True)
        self.btn_details.setToolTip("Detail view: duration, format, camera, log profile and status")
        self.btn_thumbs = QToolButton(text="▦", checkable=True, autoExclusive=True)
        self.btn_thumbs.setToolTip("Thumbnail view (hover to scrub)")
        self.btn_details.clicked.connect(lambda: self.set_view_mode("details"))
        self.btn_thumbs.clicked.connect(lambda: self.set_view_mode("thumbnails"))
        top = QHBoxLayout()
        top.addWidget(b_import, 1)
        top.addWidget(b_remove)
        top.addWidget(self.btn_details)
        top.addWidget(self.btn_thumbs)
        # Bin and filters
        self.cb_bin = QComboBox()
        self.cb_bin.setToolTip("Show one bin of the media (right-click the list for bins: new, move to, by date…)")
        self.cb_kind = QComboBox()
        for k, label in binlib.KINDS.items():
            self.cb_kind.addItem(label, k)
        self.cb_kind.setToolTip("Show videos, photos / images, or both")
        self.cb_ratio = QComboBox()
        for k, (label, _r) in binlib.RATIOS.items():
            self.cb_ratio.addItem(label, k)
        self.cb_ratio.setToolTip("Show only pictures of this shape (as shown: a phone clip filmed upright is "
                                 "portrait)")
        for cb in (self.cb_bin, self.cb_kind, self.cb_ratio):
            cb.setMinimumContentsLength(6)
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.currentIndexChanged.connect(lambda _: self.refresh())
        filters = QHBoxLayout()
        filters.addWidget(self.cb_bin, 2)
        filters.addWidget(self.cb_kind, 1)
        filters.addWidget(self.cb_ratio, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        lay.addLayout(filters)
        lay.addWidget(self.stack, 1)
        self.setAcceptDrops(True)
        self.set_view_mode(self._qs.value("bin_view", "details", type=str))

    # ------------------------------------------------------------------ view mode
    def set_view_mode(self, mode: str):
        """"details" (list with columns) or "thumbnails" (grid)."""
        thumbs = mode == "thumbnails"
        sel = self.selected_ids()
        self.stack.setCurrentWidget(self.grid if thumbs else self.tree)
        (self.btn_thumbs if thumbs else self.btn_details).setChecked(True)
        self._qs.setValue("bin_view", "thumbnails" if thumbs else "details")
        self._select(sel)

    def view_mode(self) -> str:
        return "thumbnails" if self.stack.currentWidget() is self.grid else "details"

    def _apply_thumb_size(self):
        """Size the thumbnails so GRID_COLUMNS of them fill the width (room kept for the scrollbar,
        so it appearing doesn't change the layout)."""
        g = self.grid
        avail = g.width() - 2 * g.frameWidth() - g.verticalScrollBar().sizeHint().width()
        cell = max(48, avail // GRID_COLUMNS - 1)
        w = max(32, cell - 2 * g.spacing() - 4)
        h = w * 9 // 16
        if g.iconSize() != QSize(w, h):
            g.setIconSize(QSize(w, h))
            g.setGridSize(QSize(cell, h + 2 * g.spacing() + 4))
            self._icons_timer.start()          # redraw the thumbnails at the new size once resizing settles

    def _thumb(self, fid: str, index=None) -> Optional[QPixmap]:
        """The footage's thumbnail (index None: the middle one), without the used badge."""
        f = self.project.footage.get(fid)
        if f is not None and f.is_image:
            arr = images.thumbnail(f.path) if os.path.exists(f.path) else None
            return None if arr is None else _pixmap(arr)
        a = self.analyses.get(fid)
        if a is None or not len(a.thumbs):
            return None
        return _pixmap(a.thumbs[len(a.thumbs) // 2 if index is None else index])

    def _scrub(self, item, x: float):
        """Show the clip's frame under the mouse on its grid thumbnail."""
        if self._scrubbing is not None and self._scrubbing is not item:
            self._set_grid_icon(self._scrubbing, None)
            self._scrubbing = None
        if item is None:
            return
        a = self.analyses.get(item.data(Qt.UserRole))
        if a is None or not len(a.thumbs):
            return
        self._scrubbing = item
        self._set_grid_icon(item, min(len(a.thumbs) - 1, max(0, int(x * len(a.thumbs)))))

    def _set_grid_icon(self, item, index):
        """index None: the clip's representative (middle) thumbnail. Scaled to the grid's icon size: Qt never
        enlarges an icon past its own pixels, so a small thumbnail would stay small in a wide cell."""
        fid = item.data(Qt.UserRole)
        pm = self._thumb(fid, index)
        pm = pm if pm is not None else self._placeholder()
        size = self.grid.iconSize()
        if pm.width() != size.width() and size.width() > 0:
            pm = pm.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item.setIcon(QIcon(_badge(pm, self._usage.get(fid, 0))))

    def _refresh_grid_icons(self):
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            if it is not self._scrubbing:
                self._set_grid_icon(it, None)

    def _placeholder(self) -> QPixmap:
        pm = QPixmap(128, 72)
        pm.fill(QColor("#22252b"))
        p = QPainter(pm)
        p.setPen(QColor("#6b6e76"))
        p.drawText(pm.rect(), Qt.AlignCenter, "Analyzing…")
        p.end()
        return pm

    # ------------------------------------------------------------------ content
    def set_project(self, project: Project):
        self.project = project
        self.refresh()

    def selected_ids(self) -> list[str]:
        if self.stack.currentWidget() is self.grid:
            return [it.data(Qt.UserRole) for it in self.grid.selectedItems()]
        return [it.data(0, Qt.UserRole) for it in self.tree.selectedItems()]

    def select_all(self):
        """Every media item shown (Ctrl+A in the Media panel)."""
        self.stack.currentWidget().selectAll()

    def _select(self, ids):
        ids = set(ids)
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            it.setSelected(it.data(0, Qt.UserRole) in ids)
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            it.setSelected(it.data(Qt.UserRole) in ids)

    def reveal(self, fid: str) -> bool:
        """Show and select this footage (and only it), clearing the bin and filters when they hide it."""
        f = self.project.footage.get(fid)
        if f is None:
            return False
        shown = (self.current_bin(), self.cb_kind.currentData() or "all", self.cb_ratio.currentData() or "all")
        if not binlib.visible(f, *shown):
            for cb in (self.cb_bin, self.cb_kind, self.cb_ratio):
                cb.blockSignals(True)
                cb.setCurrentIndex(0)
                cb.blockSignals(False)
            self.refresh()
        self._select([fid])
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole) == fid:
                self.tree.setCurrentItem(it)
                self.tree.scrollToItem(it)
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            if it.data(Qt.UserRole) == fid:
                self.grid.setCurrentItem(it)
                self.grid.scrollToItem(it)
        self._select([fid])
        self.stack.currentWidget().setFocus()
        return True

    def current_bin(self) -> str:
        data = self.cb_bin.currentData()
        return binlib.ALL if data is None else data

    def _fill_bins(self):
        """The bin choices: all media, not in a bin, and each bin, with how much is in it."""
        cur = self.current_bin()
        foot = list(self.project.footage.values())
        self.cb_bin.blockSignals(True)
        self.cb_bin.clear()
        self.cb_bin.addItem(f"All media ({len(foot)})", binlib.ALL)
        names = binlib.all_bins(self.project)
        if names:
            self.cb_bin.addItem(f"Not in a bin ({sum(1 for f in foot if not f.bin)})", binlib.UNSORTED)
            for b in names:
                self.cb_bin.addItem(f"📁 {b} ({sum(1 for f in foot if f.bin == b)})", b)
        i = self.cb_bin.findData(cur)
        self.cb_bin.setCurrentIndex(i if i >= 0 else 0)
        self.cb_bin.blockSignals(False)

    def show_bin(self, name: str):
        self._fill_bins()
        i = self.cb_bin.findData(name)
        if i >= 0:
            self.cb_bin.setCurrentIndex(i)

    def refresh(self):
        sel = set(self.selected_ids())
        self.tree.clear()
        self.grid.clear()
        self._scrubbing = None
        self._usage = self.project.usage()
        self._fill_bins()
        shown = (self.current_bin(), self.cb_kind.currentData() or "all", self.cb_ratio.currentData() or "all")
        for f in self.project.footage.values():
            if not binlib.visible(f, *shown):
                continue
            used = self._usage.get(f.id, 0)
            in_edit = (f"Used in the edit ({used}×)" if used > 1 else "Used in the edit") if used else "Not used yet"
            status = [in_edit]
            if f.is_image:
                status.append("Animated GIF" if f.codec == "gif" else "Image")
            else:
                status.append("Proxy ✓" if f.has_proxy() else "No proxy")
                status.append("Analyzed ✓" if f.id in self.analyses else "Not analyzed")
            if f.transcript is not None:
                status.append("Captions ✓")
            log = "—" if f.log_profile == "none" else profile_label(f.log_profile).split(" /")[0]
            length = "image" if f.is_image else fmt_time(f.duration, frames=False)
            fmt = f"{f.width}×{f.height} {f.codec.upper()}" if f.is_image else \
                f"{f.width}×{f.height} {f.fps:.2f}p {f.bit_depth}bit"
            it = QTreeWidgetItem([f.name, (f"✓ {used}" if used > 1 else "✓") if used else "—", length, fmt,
                                  f.camera or "—", log, " · ".join(status), f.bin or "—"])
            it.setData(0, Qt.UserRole, f.id)
            it.setTextAlignment(1, Qt.AlignCenter)
            it.setForeground(1, QBrush(QColor(USED_COLOR if used else "#5d6068")))
            it.setToolTip(1, in_edit)
            it.setToolTip(0, f"{f.path}\n{f.codec} {f.pix_fmt} · {f.color_space or 'untagged'} "
                             f"{f.color_transfer or ''}\n{in_edit}")
            it.setToolTip(5, f"{profile_label(f.log_profile)}" + (f"\n{f.log_note}" if f.log_note else ""))
            pm = self._thumb(f.id)
            if pm is not None:
                it.setIcon(0, QIcon(_badge(pm, used)))
            self.tree.addTopLevelItem(it)

            badges = [b for b, on in (("P", f.has_proxy()), ("CC", f.transcript is not None),
                                      (f"✓×{used}", used)) if on]
            gi = QListWidgetItem()
            gi.setData(Qt.UserRole, f.id)
            gi.setToolTip(f"{f.name} · {length}"
                          + (f" · {' '.join(badges)}" if badges else "") + f"\n{f.path}\n{f.width}×{f.height} {f.fps:.2f}p {f.bit_depth}bit · {f.camera or 'unknown camera'}"
                          f"\n{profile_label(f.log_profile)}\n{' · '.join(status)}")
            self.grid.addItem(gi)
            self._set_grid_icon(gi, None)
        self._select(sel)

    # ------------------------------------------------------------------ bins
    def _bin_edit(self, fn, message: str, show: str | None = None):
        self.edit_started.emit()
        fn()
        if show is not None:
            self.show_bin(show)
        self.refresh()
        self.edited.emit(message)

    def new_bin(self, ids: list[str] | None = None):
        name, ok = QInputDialog.getText(self, "New bin", "Bin name:")
        name = name.strip()
        if not ok or not name:
            return
        if ids:
            self._bin_edit(lambda: binlib.move_to_bin(self.project, ids, name),
                           f"Moved {len(ids)} item(s) to the new bin “{name}”", name)
        else:
            self._bin_edit(lambda: binlib.add_bin(self.project, name), f"New bin “{name}”", name)

    def add_to_broll(self, ids: list[str]):
        self._bin_edit(lambda: binlib.move_to_bin(self.project, ids, binlib.BROLL),
                       f"Added {len(ids)} item(s) to the {binlib.BROLL} bin: scenes open with it first")

    def move_to_bin(self, ids: list[str], name: str):
        self._bin_edit(lambda: binlib.move_to_bin(self.project, ids, name),
                       f"Moved {len(ids)} item(s) " + (f"to the bin “{name}”" if name else "out of their bin"))

    def bins_by_date(self, ids: list[str] | None = None):
        out: dict = {}
        self._bin_edit(lambda: out.update(binlib.bins_by_date(self.project, ids)), "")
        self.edited.emit(f"Sorted {sum(out.values())} item(s) into {len(out)} bin(s) by recording date")

    def rename_bin(self, name: str):
        new, ok = QInputDialog.getText(self, "Rename bin", "Bin name:", text=name)
        if ok and new.strip() and new.strip() != name:
            self._bin_edit(lambda: binlib.rename_bin(self.project, name, new), f"Renamed the bin to “{new.strip()}”",
                           new.strip())

    def clear_bins(self):
        n = len(binlib.all_bins(self.project))
        self._bin_edit(lambda: binlib.clear_bins(self.project),
                       f"Cleared {n} bin(s) (all the media stays in the project)", binlib.ALL)

    def delete_bin(self, name: str):
        self._bin_edit(lambda: binlib.delete_bin(self.project, name),
                       f"Deleted the bin “{name}” (its media stays in the project)", binlib.ALL)

    def _bin_menu(self, m: QMenu, ids: list[str]):
        sub = m.addMenu("Bins")
        if ids:
            move = sub.addMenu(f"Move {len(ids)} item(s) to bin")
            for b in binlib.all_bins(self.project):
                move.addAction(f"📁 {b}", lambda b=b: self.move_to_bin(ids, b))
            move.addAction("New bin…", lambda: self.new_bin(ids))
            if any(self.project.footage[i].bin for i in ids if i in self.project.footage):
                move.addSeparator()
                move.addAction("Out of its bin", lambda: self.move_to_bin(ids, ""))
        else:
            sub.addAction("New bin…", lambda: self.new_bin())
        sub.addSeparator()
        sub.addAction("Bins by date: sort all media by recording day", lambda: self.bins_by_date())
        if ids:
            sub.addAction(f"Bins by date: only the {len(ids)} selected", lambda: self.bins_by_date(ids))
        cur = self.current_bin()
        if cur not in (binlib.ALL, binlib.UNSORTED):
            sub.addSeparator()
            sub.addAction(f"Rename bin “{cur}”…", lambda: self.rename_bin(cur))
            sub.addAction(f"Delete bin “{cur}” (keeps its media)", lambda: self.delete_bin(cur))
        if binlib.all_bins(self.project):
            sub.addSeparator()
            sub.addAction("Clear all bins (keeps the media)", self.clear_bins)

    def _menu(self, pos):
        ids = self.selected_ids()
        m = QMenu(self)
        if not ids:                         # right-click on empty space: bins only
            self._bin_menu(m, [])
            m.exec(self.stack.currentWidget().viewport().mapToGlobal(pos))
            return
        self._bin_menu(m, ids)
        a = m.addAction(f"Add to {binlib.BROLL} bin", lambda: self.add_to_broll(ids))
        a.setToolTip("Cutaway footage (views, travel, details): scenes of a long-form auto edit open with it first")
        m.addSeparator()
        for key, label in (("play", "▶ Play in the viewer (mark the part to use)"),
                           (None, None),
                           ("append_best", "Append best moment to timeline"),
                           ("append_whole", "Append whole clip to timeline"),
                           ("overlay", "Add on a video layer at the playhead (picture in picture)"),
                           (None, None),
                           ("proxy", "Generate proxy"), ("delete_proxy", "Delete proxy"),
                           ("analyze", "Analyze (re-run)"), ("transcribe", "Transcribe (again) for captions"),
                           ("transcript", "Show the full transcript…"),
                           (None, None)):
            if key is None:
                m.addSeparator()
                continue
            a = QAction(label, m)
            a.triggered.connect(lambda _=False, k=key: self.action.emit(k, ids))
            m.addAction(a)
        sub = m.addMenu("Log profile → Rec.709")
        a = QAction("Detect from camera", sub)
        a.setToolTip("Identify the camera and choose its log profile automatically")
        a.triggered.connect(lambda: self.action.emit("detect_log", ids))
        sub.addAction(a)
        sub.addSeparator()
        current = {self.project.footage[i].log_profile for i in ids if i in self.project.footage}
        choices = list(LOG_CHOICES)
        luts = lutlib.library()
        if luts:
            choices.append((None, None))
            choices += [(lut_profile(path), f"LUT: {name}") for name, path in luts]
        for key, label in choices:
            if key is None:
                sub.addSeparator()
                continue
            a = QAction(label, sub, checkable=True)
            a.setChecked(current == {key})
            a.triggered.connect(lambda _=False, k=key: self.log_profile.emit(ids, k))
            sub.addAction(a)
        sub.addSeparator()
        a = QAction("Import conversion LUT (.cube)…", sub)
        a.triggered.connect(lambda: self.action.emit("import_lut_log", ids))
        sub.addAction(a)
        sub = m.addMenu("360° footage")
        sub.setToolTipsVisible(True)
        current = {getattr(self.project.footage[i], "projection", "") for i in ids if i in self.project.footage}
        for key, label in spherical.PROJECTIONS.items():
            a = QAction(label, sub, checkable=True)
            a.setChecked(current == {key})
            a.triggered.connect(lambda _=False, k=key: self.action.emit("projection:" + k, ids))
            sub.addAction(a)
        m.addSeparator()
        a = QAction("Show in folder", m)
        a.triggered.connect(lambda: open_path(os.path.dirname(self.project.footage[ids[0]].path)))
        m.addAction(a)
        a = QAction("Remove from project", m)
        a.triggered.connect(lambda: self.action.emit("remove", ids))
        m.addAction(a)
        m.addSeparator()
        view = m.addMenu("View")
        for mode, label in (("details", "Details"), ("thumbnails", "Thumbnails")):
            a = QAction(label, view, checkable=True)
            a.setChecked(self.view_mode() == mode)
            a.triggered.connect(lambda _=False, md=mode: self.set_view_mode(md))
            view.addAction(a)
        m.exec(self.stack.currentWidget().viewport().mapToGlobal(pos))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            exts = VIDEO_EXTENSIONS + images.IMAGE_EXTENSIONS
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    paths += [os.path.join(root, f) for f in files if f.lower().endswith(exts)]
            elif p.lower().endswith(exts):
                paths.append(p)
        if paths:
            self.files_dropped.emit(sorted(paths))
