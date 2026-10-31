"""Timeline: the main track (reorder, trim and slip clips; magnetic, or with the magnet off clips placed freely with
black gaps between them, see gaps.py), video layers above it, titles, sound effects and captions. Lane heights can
be changed by dragging their bottom edges."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFormLayout, QFrame, QWidgetAction, QHBoxLayout, QLabel, QMenu,
                               QPushButton, QScrollArea, QScrollBar, QSizePolicy, QSlider, QToolButton, QVBoxLayout,
                               QWidget)

from .. import gaps, images, sfx
from ..analysis import WAVE_RATE, Analysis
from ..captions import Caption, caption_color, shift_group
from ..edits import TRANSITIONS, edits
from ..models import OverlayClip, Project, SoundClip
from ..speed import RAMPS, SOUND_MODES, SPEED_PRESETS, sound_mode
from ..speed import label as speed_label
from ..titles import ANIM_IN, ANIM_OUT, TEXT_PRESETS
from .flow import FlowBar
from .slip_transcript import SlipTranscript
from .theme import CLIP_COLORS, fmt_time

RULER = 24
TEXT_Y = RULER + 4                 # text / title track: two rows so overlapping titles stay visible
TEXT_ROW = 16
TEXT_H = TEXT_ROW * 2
OV_Y = TEXT_Y + TEXT_H + 4         # video layers (overlays), the top layer first, above the main track
CAP_H = 24
EDGE = 7
MIN_DUR = 0.25
MAX_LAYERS = 8
MIN_PPS, MAX_PPS = 0.05, 400.0    # zoom limits, pixels per second (0.05: an hour fits 180 px)
FIT_MARGIN = 40                    # pixels left free after the end of the edit (and when fitted)
# Lane heights in pixels: the main video track, each video layer row and each sound-effect row
# (the sound lane has two rows so overlapping sounds stay visible). Drag a lane's bottom edge to change it.
DEFAULT_HEIGHTS = {"video": 84, "layer": 40, "sound": 17}
HEIGHT_RANGES = {"video": (48, 300), "layer": (24, 220), "sound": (14, 100)}
FOOTAGE_MIME = "application/x-kangedit-footage"   # footage ids dragged from the media bin, one per line


ROLES = {"hook": ("🎣", "hook", "#ff9f43"), "scene": ("🏞", "scene opener", "#4cd08a"),
         "opening": ("👋", "opening", "#7aa7ff"), "closing": ("👋", "closing", "#7aa7ff")}


def clip_marks(c) -> list[tuple[str, str]]:
    """What a clip is and what is done to it, as (icon, words): its role in an auto edit (hook, scene opener,
    opening / closing) and its dynamic zoom, stabilization and voice clean-up."""
    from ..motion import MOTIONS
    out = []
    role = ROLES.get(getattr(c, "label", ""))
    if role:
        out.append(role[:2])
    if getattr(c, "motion", ""):
        out.append(("🔍", MOTIONS.get(c.motion, "dynamic zoom").lower()))
    if getattr(c, "stabilize", 0) > 0:
        out.append(("〰", "stabilized"))
    if getattr(c, "voice_clean", 0) > 0:
        out.append(("🎙", "voice cleaned up"))
    return out


def marks_text(c) -> str:
    """The clip's marks for a tooltip line ("" when none)."""
    m = clip_marks(c)
    return " · ".join(f"{icon} {words}" for icon, words in m)


class TimelineView(QWidget):
    seek = Signal(float)
    selection_changed = Signal(str)
    edit_started = Signal()
    edited = Signal()
    edit_finished = Signal()
    slip_preview = Signal(str, float)
    slip_end = Signal()
    caption_clicked = Signal(int)
    captions_shifted = Signal(int, float)   # caption index, timeline seconds: it and its clip's later ones moved
    context_action = Signal(str, str)
    view_changed = Signal()
    sound_selected = Signal(str)
    sounds_changed = Signal()            # a sound was moved / trimmed (after edit_started)
    sound_dropped = Signal(list, float)  # audio file paths dropped on the timeline, at timeline time
    text_selected = Signal(str)
    texts_changed = Signal()             # a title was moved / trimmed (after edit_started)
    overlay_selected = Signal(str)
    overlays_changed = Signal()          # an overlay was moved / trimmed (after edit_started)
    footage_dropped = Signal(list, float, int)   # footage ids from the media bin, time, layer (-1: main track)
    heights_changed = Signal(dict)       # a lane was resized by dragging its edge

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.analyses: dict[str, Analysis] = {}
        self.captions: list[Caption] = []
        self.playhead = 0.0
        self.pps = 40.0          # pixels per second
        self.scroll_x = 0.0
        self._selected: str = ""          # primary selection (shown in the inspector)
        self._selection: set[str] = set()  # every selected clip, including the primary one
        self._ov_selection: set[str] = set()   # selected clips on the video layers (Ctrl+A, Ctrl+click)
        self.mode = "move"       # move | slip
        self.magnetic = True     # off: a moved clip leaves black behind and goes where it is dropped (gaps.place)
        self.min_gap = 0.0
        self._conflicts: set[str] = set()
        self._drag: Optional[dict] = None
        self._thumb_cache: dict[tuple, QImage] = {}
        self._static: Optional[QPixmap] = None   # everything but the playhead
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.heights = dict(DEFAULT_HEIGHTS)
        self._layout()
        self.setAcceptDrops(True)
        self.selected_sound = ""
        self.selected_text = ""
        self.selected_overlay = ""
        self._image_thumbs: dict[str, Optional[QImage]] = {}
        self._sound_max: dict[str, float] = {}   # file length per sound file (trim limit)
        self._scrub_x = 0.0                      # last mouse x while scrubbing (for edge auto-scroll)
        self._edge_timer = QTimer(self)
        self._edge_timer.setInterval(30)
        self._edge_timer.timeout.connect(self._edge_scroll)

    # ------------------------------------------------------------------ selection
    @property
    def selected(self) -> str:
        return self._selected

    @selected.setter
    def selected(self, clip_id: str):
        """Select one clip only (or none with "")."""
        self._selected = clip_id
        self._selection = {clip_id} if clip_id else set()
        self._ov_selection = set()

    def selected_ids(self) -> list[str]:
        """Selected clips in timeline order."""
        return [c.id for c in self.project.clips if c.id in self._selection]

    def selected_overlay_ids(self) -> list[str]:
        """Selected clips on the video layers (the one shown in the Layer tab and the rest of the selection)."""
        sel = self._ov_selection | ({self.selected_overlay} if self.selected_overlay else set())
        return [c.id for c in self.project.overlays if c.id in sel]

    def select_all(self):
        """Every clip on the main track and on the video layers."""
        self._selection = {c.id for c in self.project.clips}
        self._ov_selection = {c.id for c in self.project.overlays}
        if self._selected not in self._selection:
            self._selected = self.project.clips[0].id if self.project.clips else ""
        self.selection_changed.emit(self._selected)
        self.update()

    def _click_select(self, clip_id: str, mods) -> None:
        """Ctrl: toggle the clip. Shift: select the range from the primary clip. Plain: only this clip."""
        if mods & Qt.ControlModifier:
            if clip_id in self._selection and len(self._selection) > 1:
                self._selection.discard(clip_id)
                if self._selected == clip_id:
                    self._selected = self.selected_ids()[0]
            else:
                self._selection.add(clip_id)
                self._selected = clip_id
        elif mods & Qt.ShiftModifier and self._selected:
            ids = [c.id for c in self.project.clips]
            a, b = sorted((ids.index(self._selected), ids.index(clip_id))) if self._selected in ids else (0, 0)
            self._selection = set(ids[a:b + 1]) | {clip_id}
        else:
            self.selected = clip_id
        self.selection_changed.emit(self._selected)
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_A and e.modifiers() & Qt.ControlModifier:
            self.select_all()
            return
        super().keyPressEvent(e)

    # ------------------------------------------------------------------ lane layout
    @property
    def layers(self) -> int:
        """Video layers shown above the main track (at least one, and enough for every overlay)."""
        top = max((c.track for c in self.project.overlays), default=-1)
        return max(1, min(MAX_LAYERS, max(self.project.settings.overlay_tracks, top + 1)))

    def _layout(self):
        """Lane positions from the lane heights and the number of video layers."""
        h = self.heights
        self.ov_row = int(h["layer"])
        self.ov_h = self.ov_row * self.layers
        self.track_y = OV_Y + self.ov_h + 4
        self.track_h = int(h["video"])
        self.wave_h = max(18, int(round(self.track_h * 0.28)))
        self.sfx_y = self.track_y + self.track_h + 6
        self.sfx_row = int(h["sound"])
        self.sfx_h = self.sfx_row * 2
        self.cap_y = self.sfx_y + self.sfx_h + 6
        self.setMinimumHeight(self.cap_y + CAP_H + 10)

    def set_heights(self, heights: dict, emit: bool = False):
        for k, (lo, hi) in HEIGHT_RANGES.items():
            if k in heights:
                self.heights[k] = int(max(lo, min(hi, float(heights[k]))))
        self._layout()
        self.update()
        if emit:
            self.heights_changed.emit(dict(self.heights))

    def fit_heights(self, available: int):
        """Share `available` pixels of height among the lanes (the main track gets the most)."""
        fixed = OV_Y + 4 + 6 + 6 + CAP_H + 10
        room = max(0, available - fixed)
        unit = room / (2.6 + self.layers + 2 * 0.45)
        self.set_heights({"video": unit * 2.6, "layer": unit, "sound": unit * 0.45}, emit=True)

    def _edge_at(self, y: float) -> str:
        """The lane whose bottom edge is under y ("" for none)."""
        for key, bottom in (("layer", OV_Y + self.ov_h), ("video", self.track_y + self.track_h),
                            ("sound", self.sfx_y + self.sfx_h)):
            if -2 <= y - bottom <= 4:
                return key
        return ""

    # ------------------------------------------------------------------ helpers
    def set_project(self, project: Project):
        self.project = project
        self.selected = ""
        self.selected_overlay = ""
        self._thumb_cache.clear()
        self._image_thumbs.clear()
        self.refresh()

    def refresh(self):
        self._conflicts = self.project.conflicts(self.min_gap)
        self._layout()
        self.update()
        self.view_changed.emit()

    def t_to_x(self, t: float) -> float:
        return t * self.pps - self.scroll_x

    def x_to_t(self, x: float) -> float:
        return max(0.0, (x + self.scroll_x) / self.pps)

    def content_end(self) -> float:
        """Where the last thing on any lane ends."""
        p = self.project
        return max([p.duration] + [x.end for x in p.sounds] + [x.end for x in p.texts] + [x.end for x in p.overlays])

    def content_width(self) -> float:
        return self.content_end() * self.pps + FIT_MARGIN

    def fit(self):
        """Zoom so the whole edit (every lane) fills the width."""
        dur = self.content_end()
        if dur > 0:
            self.pps = max(MIN_PPS, min(MAX_PPS, (self.width() - FIT_MARGIN) / dur))
        self.scroll_x = 0
        self.refresh()

    def zoom_by(self, factor: float):
        """Zoom in (> 1) or out around the playhead when it is visible, else the middle."""
        x = self.t_to_x(self.playhead)
        self.set_zoom(self.pps * factor, x if 0 <= x <= self.width() else None)

    def set_zoom(self, pps: float, anchor_x: Optional[float] = None):
        anchor_x = self.width() / 2 if anchor_x is None else anchor_x
        t = self.x_to_t(anchor_x)
        self.pps = max(MIN_PPS, min(MAX_PPS, pps))
        self.scroll_x = max(0.0, t * self.pps - anchor_x)
        self.refresh()

    def _edge_speed(self, x: float) -> float:
        """Pixels to scroll per tick when scrubbing near / past the left or right edge."""
        margin = 30.0
        if x < margin:
            return -min(60.0, 4.0 + (margin - x) * 0.6)
        if x > self.width() - margin:
            return min(60.0, 4.0 + (x - self.width() + margin) * 0.6)
        return 0.0

    def _edge_scroll(self):
        d = self._drag
        speed = self._edge_speed(self._scrub_x)
        if d is None or d.get("kind") != "scrub" or speed == 0.0:
            self._edge_timer.stop()
            return
        max_scroll = max(0.0, self.content_width() - self.width())
        new = max(0.0, min(self.scroll_x + speed, max_scroll))
        if new != self.scroll_x:
            self.scroll_x = new
            self.update()
            self.view_changed.emit()
        self.seek.emit(self.x_to_t(min(max(0.0, self._scrub_x), self.width())))

    def ensure_visible(self, t: float):
        x = self.t_to_x(t)
        if x < 0 or x > self.width() - 20:
            self.scroll_x = max(0.0, t * self.pps - self.width() * 0.3)
            self.view_changed.emit()

    def update(self, *args):
        """A full update means the content changed; a partial one (playhead moves) reuses the cache."""
        if not args:
            self._static = None
        super().update(*args)

    def set_playhead(self, t: float, follow: bool = False):
        old_x, old_scroll = self.t_to_x(self.playhead), self.scroll_x
        self.playhead = t
        x = self.t_to_x(t)
        # (a jump out of view: the start, the end, the next clip... from the viewer's buttons or keys)
        if follow or x < 0 or x > self.width():
            self.ensure_visible(t)
        if self.scroll_x != old_scroll:
            self.update()
            return
        for x in (old_x, self.t_to_x(t)):
            self.update(QRect(int(x) - 8, 0, 17, self.height()))

    def _clip_rects(self):
        out, t = [], 0.0
        for c in self.project.clips:
            x0, x1 = self.t_to_x(t), self.t_to_x(t + c.duration)
            out.append((c, QRectF(x0, self.track_y, max(1.0, x1 - x0), self.track_h), t))
            t += c.duration
        return out

    def _hit(self, pos: QPointF):
        if not (self.track_y <= pos.y() <= self.track_y + self.track_h):
            return None, None
        for c, r, t in self._clip_rects():
            if r.left() - 2 <= pos.x() <= r.right() + 2:
                if pos.x() - r.left() <= EDGE and r.width() > EDGE * 3:
                    return c, "trim_l"
                if r.right() - pos.x() <= EDGE and r.width() > EDGE * 3:
                    return c, "trim_r"
                return c, "body"
        return None, None

    def _footage_color(self, fid: str) -> QColor:
        ids = list(self.project.footage)
        k = ids.index(fid) if fid in ids else 0
        return QColor(CLIP_COLORS[k % len(CLIP_COLORS)])

    def _thumb(self, fid: str, idx: int) -> Optional[QImage]:
        key = (fid, idx)
        img = self._thumb_cache.get(key)
        if img is None:
            a = self.analyses.get(fid)
            if a is None or not len(a.thumbs):
                return None
            arr = np.ascontiguousarray(a.thumbs[min(idx, len(a.thumbs) - 1)])
            img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888).copy()
            self._thumb_cache[key] = img
        return img

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _):
        dpr = self.devicePixelRatioF()
        size = QSize(max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)))
        if self._static is None or self._static.size() != size:
            self._static = QPixmap(size)
            self._static.setDevicePixelRatio(dpr)
            sp = QPainter(self._static)
            self._paint_static(sp)
            sp.end()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._static)
        self._paint_playhead(p)

    def _paint_static(self, p: QPainter):
        self._layout()
        p.fillRect(self.rect(), QColor("#16171a"))
        self._paint_ruler(p)
        self._paint_overlays(p)
        p.fillRect(QRectF(0, self.track_y, self.width(), self.track_h), QColor("#1d1f23"))
        p.fillRect(QRectF(0, self.cap_y, self.width(), CAP_H), QColor("#1a1b1f"))
        self._paint_sounds(p)
        self._paint_texts(p)

        if not self.project.clips:
            p.setPen(QColor("#5d6068"))
            p.drawText(QRectF(0, self.track_y, self.width(), self.track_h), Qt.AlignCenter,
                       "Timeline is empty — use Auto Edit, or right-click footage → Append")
        for c, r, t in self._clip_rects():
            if r.right() < 0 or r.left() > self.width():
                continue
            self._paint_clip(p, c, r)
        self._paint_edits(p)
        self._paint_fades(p)

        d = self._drag
        moving, shift = set(), 0.0
        if d and d.get("kind") == "cap_shift" and d.get("active"):
            moving, shift = set(d["group"]), d.get("dt", 0.0)      # (drawn where the drag puts them)
        for i, cap in enumerate(self.captions):
            dt = shift if i in moving else 0.0
            x0, x1 = self.t_to_x(cap.start + dt), self.t_to_x(cap.end + dt)
            if x1 < 0 or x0 > self.width():
                continue
            r = QRectF(x0 + 1, self.cap_y + 2, max(2.0, x1 - x0 - 2), CAP_H - 4)
            p.setPen(QPen(QColor("#ffd34d"), 1) if i in moving else Qt.NoPen)
            p.setBrush(QColor("#5b4a8a"))
            p.drawRoundedRect(r, 3, 3)
            col = caption_color(self.project.settings.captions, cap) if cap.speaker >= 0 else ""
            if col and self.project.settings.captions.speakers != "off":
                p.fillRect(QRectF(r.left(), r.top(), min(4.0, r.width()), r.height()), QColor(col))  # who speaks
            if r.width() > 24:
                p.setPen(QColor("#ece8f7"))
                f = QFont()
                f.setPixelSize(11)
                p.setFont(f)
                p.drawText(r.adjusted(4, 0, -3, 0), Qt.AlignVCenter | Qt.AlignLeft,
                           p.fontMetrics().elidedText(cap.text, Qt.ElideRight, int(r.width() - 7)))

        d = self._drag
        if d and d.get("kind") in ("trim_l", "trim_r") and d.get("active") and "dur0" in d:
            self._paint_trim_length(p, d)
        if d and d.get("kind") == "move" and d.get("active") and d.get("free") is not None:
            # Magnet off: the clip drawn where it will go (a black gap stays where it was)
            x0, x1 = self.t_to_x(d["free"]), self.t_to_x(d["free"] + d["dur0"])
            p.setPen(QPen(QColor("#ffd34d"), 2))
            p.setBrush(QColor(255, 211, 77, 50))
            p.drawRoundedRect(QRectF(x0, self.track_y + 2, max(3.0, x1 - x0), self.track_h - 4), 4, 4)
            p.setPen(QColor("#ffd34d"))
            p.drawText(QPointF(x0 + 4, self.track_y - 2), fmt_time(d["free"]))
        elif d and d.get("kind") == "move" and d.get("active") and d.get("drop") is not None:
            starts = self.project.clip_starts() + [self.project.duration]
            x = self.t_to_x(starts[d["drop"]])
            p.setPen(QPen(QColor("#ffd34d"), 3))
            p.drawLine(QPointF(x, self.track_y - 4), QPointF(x, self.track_y + self.track_h + 4))

    def _paint_trim_length(self, p: QPainter, d: dict):
        """While a clip is trimmed: its new length and the change, at the edge being dragged."""
        rect = next((r for c, r, _t in self._clip_rects() if c.id == d["clip"]), None)
        c = next((c for c in self.project.clips if c.id == d["clip"]), None)
        if rect is None or c is None:
            return
        diff = c.duration - d["dur0"]
        text = f"{c.duration:.2f} s" + (f"  {diff:+.2f}" if abs(diff) >= 0.005 else "")
        f = p.font()
        f.setBold(True)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 12
        h = p.fontMetrics().height() + 4
        x = rect.left() + 4 if d["kind"] == "trim_l" else rect.right() - w - 4
        x = min(max(2.0, x), self.width() - w - 2.0)
        box = QRectF(x, self.track_y + 4, w, h)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 190))
        p.drawRoundedRect(box, 4, 4)
        p.setPen(QColor("#ffd34d"))
        p.drawText(box, Qt.AlignCenter, text)

    def _paint_fades(self, p: QPainter):
        """Fade in / out at the ends of the timeline: a shaded ramp over the picture and / or the sound."""
        st = self.project.settings
        dur = self.project.duration
        if not self.project.clips:
            return
        wy = self.track_y + self.track_h - self.wave_h
        bands = []
        if st.fades(dur) != (0.0, 0.0):
            bands.append((self.track_y, wy, st.fades(dur)))
        if st.fades(dur, sound=True) != (0.0, 0.0):
            bands.append((wy, self.track_y + self.track_h, st.fades(dur, sound=True)))
        p.setPen(QPen(QColor(255, 255, 255, 200), 1))
        p.setBrush(QColor(0, 0, 0, 140))
        for top, bot, (fi, fo) in bands:
            if fi > 0:
                x0, x1 = self.t_to_x(0.0), self.t_to_x(fi)
                p.drawPolygon(QPolygonF([QPointF(x0, top), QPointF(x1, top), QPointF(x0, bot)]))
            if fo > 0:
                x0, x1 = self.t_to_x(dur - fo), self.t_to_x(dur)
                p.drawPolygon(QPolygonF([QPointF(x0, top), QPointF(x1, top), QPointF(x1, bot)]))

    def _paint_edits(self, p: QPainter):
        """Transitions (bow tie over the cut) and J/L split edits (badge where the sound cuts)."""
        if len(self.project.clips) < 2:
            return
        f = QFont()
        f.setPixelSize(10)
        f.setBold(True)
        p.setFont(f)
        starts = self.project.clip_starts()
        for k, e in enumerate(edits(self.project)):
            if k == 0:
                continue
            x = self.t_to_x(starts[k])
            if x < -40 or x > self.width() + 40:
                continue
            if e.half:
                w = max(8.0, e.half * self.pps)
                top, bot = self.track_y + 17, self.track_y + self.track_h - self.wave_h - 1
                poly = QPolygonF([QPointF(x - w, top), QPointF(x + w, bot), QPointF(x + w, top), QPointF(x - w, bot)])
                p.setPen(QPen(QColor(255, 255, 255, 220), 1))
                p.setBrush(QColor(20, 20, 24, 150))
                p.drawPolygon(poly)
            elif e.split:
                xa = self.t_to_x(starts[k] + e.split)
                wy = self.track_y + self.track_h - self.wave_h
                p.setPen(QPen(QColor("#ffd34d"), 2))
                p.drawLine(QPointF(x, wy - 4), QPointF(xa, wy - 4))
                p.drawLine(QPointF(xa, wy - 4), QPointF(xa, self.track_y + self.track_h))
                r = QRectF(xa - 7, wy - 13, 14, 12)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#ffd34d"))
                p.drawRoundedRect(r, 2, 2)
                p.setPen(QColor("#1a1a1a"))
                p.drawText(r, Qt.AlignCenter, "J" if e.split < 0 else "L")

    def _paint_playhead(self, p: QPainter):
        x = self.t_to_x(self.playhead)
        p.setPen(QPen(QColor("#ff4d4d"), 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setBrush(QColor("#ff4d4d"))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(x - 6, 0), QPointF(x + 6, 0), QPointF(x, 9)]))

    def _paint_ruler(self, p: QPainter):
        p.fillRect(QRectF(0, 0, self.width(), RULER), QColor("#202227"))
        steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600]
        major = next((s for s in steps if s * self.pps >= 70), 3600)
        minor = major / 5
        t0 = self.x_to_t(0)
        t1 = self.x_to_t(self.width())
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        k = int(t0 / minor)
        while k * minor <= t1:
            t = k * minor
            x = self.t_to_x(t)
            is_major = abs(t / major - round(t / major)) < 1e-6
            p.setPen(QColor("#6c707a") if is_major else QColor("#3d4048"))
            p.drawLine(QPointF(x, RULER - (10 if is_major else 4)), QPointF(x, RULER))
            if is_major:
                p.setPen(QColor("#9aa0ab"))
                label = fmt_time(t, frames=False) if major >= 1 else f"{t:.2f}s"
                p.drawText(QPointF(x + 3, 11), label)
            k += 1

    def _paint_clip(self, p: QPainter, c, r: QRectF):
        f = self.project.footage.get(c.footage_id)
        if f is not None and f.generated == gaps.BLACK:
            self._paint_black(p, c, r)
            return
        col = self._footage_color(c.footage_id)
        path = QPainterPath()
        path.addRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        p.save()
        p.setClipPath(path)
        p.fillRect(r, col.darker(160))

        # Filmstrip
        a = self.analyses.get(c.footage_id)
        strip = QRectF(r.left(), r.top() + 16, r.width(), self.track_h - 16 - self.wave_h)
        if f is not None and f.is_image:
            self._paint_image_strip(p, f, strip, r)
        elif a is not None and len(a.thumbs):
            th_h = strip.height()
            th_w = th_h * a.thumbs.shape[2] / a.thumbs.shape[1]
            x = r.left()
            if x < 0:  # skip tiles scrolled out on the left
                x += int(-x // th_w) * th_w
            while x < min(r.right(), self.width()):
                src_t = c.src_at((x - r.left()) / self.pps)
                idx = int(np.searchsorted(a.thumb_times, src_t, side="right") - 1)
                img = self._thumb(c.footage_id, max(0, idx))
                if img is not None:
                    p.drawImage(QRectF(x, strip.top(), th_w, th_h), img)
                x += th_w

        # Waveform
        wr = QRectF(r.left(), r.bottom() - self.wave_h, r.width(), self.wave_h)
        p.fillRect(wr, QColor(0, 0, 0, 110))
        if a is not None and a.wave.size > 1 and f is not None and f.has_audio:
            x0 = int(max(r.left(), 0))
            x1 = int(min(r.right(), self.width()))
            if x1 > x0:
                xs = np.arange(x0, x1, 2)
                ts = np.asarray(c.sound_src_at((xs - r.left()) / self.pps), dtype=np.float64)
                vol = 0.0 if c.audio_mode == "mute" else c.volume
                vals = a.wave[np.clip((ts * WAVE_RATE).astype(int), 0, a.wave.size - 1)] * vol
                mid = wr.center().y()
                amp = (self.wave_h / 2 - 1) * np.clip(vals, 0, 1)
                poly = QPolygonF([QPointF(x, mid - h) for x, h in zip(xs, amp)] +
                                 [QPointF(x, mid + h) for x, h in zip(xs[::-1], amp[::-1])])
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#7fd1a8"))
                p.drawPolygon(poly)

        # Header
        p.fillRect(QRectF(r.left(), r.top(), r.width(), 16), col)
        role = ROLES.get(c.label)
        if role:                           # its role in the edit: a coloured line under the name
            p.fillRect(QRectF(r.left(), r.top() + 16, r.width(), 2), QColor(role[2]))
        if r.width() > 30 and f is not None:
            fnt = QFont()
            fnt.setPixelSize(10)
            p.setFont(fnt)
            p.setPen(QColor("white"))
            icons = "".join(icon for icon, _ in clip_marks(c))
            label = f"{icons + ' ' if icons else ''}{f.name}  {c.duration:.1f}s"
            if speed_label(c.speed, c.ramp):
                label += f"  ⏱{speed_label(c.speed, c.ramp)}"
            if abs(c.rotation) > 0.05:
                label += f"  ⟳{c.rotation:g}°"
            if c.score is not None:
                label += f"  ★{max(0.0, c.score) * 100:.0f}"
            p.drawText(QRectF(r.left() + 4, r.top(), r.width() - 8, 16), Qt.AlignVCenter | Qt.AlignLeft,
                       p.fontMetrics().elidedText(label, Qt.ElideRight, int(r.width() - 8)))
        p.restore()

        if c.id in self._conflicts:
            p.setPen(QPen(QColor("#ff5252"), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 4, 4)
        if c.id in self._selection and c.id != self.selected:
            p.setPen(QPen(QColor(255, 255, 255, 170), 2))
            p.setBrush(QColor(255, 255, 255, 28))
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 4, 4)
        if c.id == self.selected:
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 4, 4)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 211, 77, 220))
            p.drawRect(QRectF(r.left(), r.top() + 16, 4, self.track_h - 16))
            p.drawRect(QRectF(r.right() - 4, r.top() + 16, 4, self.track_h - 16))

    # ------------------------------------------------------------------ interaction
    def _paint_black(self, p: QPainter, c, r: QRectF):
        """A black clip (a black frame, or a gap left with the magnet off): dark, hatched, labelled."""
        sel = c.id in self._selection
        p.save()
        p.setPen(QPen(QColor("#4c8dff") if sel else QColor("#3a3d44"), 2 if sel else 1, Qt.DashLine))
        p.setBrush(QBrush(QColor("#2a2c31"), Qt.BDiagPattern))
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        if r.width() > 30:
            p.setPen(QColor("#8b909a"))
            f = QFont()
            f.setPixelSize(10)
            p.setFont(f)
            p.drawText(r.adjusted(4, 2, -2, 0), Qt.AlignTop | Qt.AlignLeft,
                       p.fontMetrics().elidedText(f"Black {c.duration:.2f}s", Qt.ElideRight, int(r.width() - 6)))
        p.restore()

    def _snap(self, t: float) -> float:
        fps = max(1.0, self.project.settings.fps)
        return round(t * fps) / fps

    # ------------------------------------------------------------------ video layers (overlays)
    def _image_thumb(self, f) -> Optional[QImage]:
        if f.id not in self._image_thumbs:
            arr = images.thumbnail(f.path) if os.path.exists(f.path) else None
            self._image_thumbs[f.id] = None if arr is None else QImage(
                arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888).copy()
        return self._image_thumbs[f.id]

    def _paint_image_strip(self, p: QPainter, f, strip: QRectF, r: QRectF):
        """An image clip's picture repeated along it."""
        img = self._image_thumb(f)
        if img is None or strip.height() < 4:
            return
        th_h = strip.height()
        th_w = th_h * img.width() / max(1, img.height())
        x = r.left()
        if x < 0:
            x += int(-x // th_w) * th_w
        while x < min(r.right(), self.width()):
            p.drawImage(QRectF(x, strip.top(), th_w, th_h), img)
            x += th_w

    def _paint_footage_strip(self, p: QPainter, c, strip: QRectF, r: QRectF):
        fid = c.footage_id
        f = self.project.footage.get(fid)
        if f is None:
            return
        if f.is_image:
            self._paint_image_strip(p, f, strip, r)
            return
        a = self.analyses.get(fid)
        if a is None or not len(a.thumbs) or strip.height() < 4:
            return
        th_h = strip.height()
        th_w = th_h * a.thumbs.shape[2] / a.thumbs.shape[1]
        x = r.left()
        if x < 0:
            x += int(-x // th_w) * th_w
        while x < min(r.right(), self.width()):
            idx = int(np.searchsorted(a.thumb_times, c.src_at((x - r.left()) / self.pps), side="right") - 1)
            img = self._thumb(fid, max(0, idx))
            if img is not None:
                p.drawImage(QRectF(x, strip.top(), th_w, th_h), img)
            x += th_w

    def _layer_top(self, track: int) -> float:
        return OV_Y + (self.layers - 1 - track) * self.ov_row

    def _track_at(self, y: float) -> int:
        k = int((y - OV_Y) // max(1, self.ov_row))
        return max(0, min(self.layers - 1, self.layers - 1 - k))

    def _overlay_rects(self) -> list[tuple[OverlayClip, QRectF]]:
        out = []
        for c in self.project.overlays:
            x0, x1 = self.t_to_x(c.start), self.t_to_x(c.end)
            out.append((c, QRectF(x0, self._layer_top(c.track) + 1, max(3.0, x1 - x0), self.ov_row - 2)))
        return out

    def _overlay_hit(self, pos: QPointF):
        if not (OV_Y <= pos.y() < OV_Y + self.ov_h):
            return None, None
        for c, r in reversed(self._overlay_rects()):
            if r.left() - 2 <= pos.x() <= r.right() + 2 and r.top() <= pos.y() <= r.bottom():
                if pos.x() - r.left() <= 6 and r.width() > 18:
                    return c, "o_trim_l"
                if r.right() - pos.x() <= 6 and r.width() > 18:
                    return c, "o_trim_r"
                return c, "o_move"
        return None, "lane"

    def _paint_overlays(self, p: QPainter):
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        for k in range(self.layers):
            top = self._layer_top(k)
            p.fillRect(QRectF(0, top, self.width(), self.ov_row - 1), QColor("#1b1d21" if k % 2 == 0 else "#191b1f"))
        if not self.project.overlays:
            p.setPen(QColor("#4d5058"))
            p.drawText(QRectF(30, self._layer_top(0), self.width() - 38, self.ov_row), Qt.AlignVCenter | Qt.AlignLeft,
                       "Video layers — drag footage or images here for picture-in-picture, logos and B-roll on top "
                       "(several cameras: ✨ Auto Edit → Multicam)")
        for c, r in self._overlay_rects():
            if r.right() < 0 or r.left() > self.width():
                continue
            ft = self.project.footage.get(c.footage_id)
            col = self._footage_color(c.footage_id)
            path = QPainterPath()
            path.addRoundedRect(r, 3, 3)
            p.save()
            p.setClipPath(path)
            p.fillRect(r, col.darker(170))
            head = min(14.0, r.height())
            if r.height() > 26:
                self._paint_footage_strip(p, c, QRectF(r.left(), r.top() + head, r.width(), r.height() - head), r)
            p.fillRect(QRectF(r.left(), r.top(), r.width(), head), col.darker(115))
            fi, fo = c.fades()
            p.setPen(QPen(QColor(255, 255, 255, 190), 1))
            p.setBrush(QColor(0, 0, 0, 100))
            for w, left in ((fi * self.pps, True), (fo * self.pps, False)):
                if w >= 2:
                    pts = ([r.topLeft(), QPointF(r.left() + w, r.top()), r.bottomLeft()] if left else
                           [QPointF(r.right() - w, r.top()), r.topRight(), r.bottomRight()])
                    p.drawPolygon(QPolygonF(pts))
            if r.width() > 24 and ft is not None:
                p.setPen(QColor("white"))
                link = ("🔗 " if c.linked else "⛓ unlinked · ") if c.link else ""
                icons = "".join(icon for icon, _ in clip_marks(c))
                label = f"{link}{icons + ' ' if icons else ''}{'🖼 ' if ft.is_image else ''}{ft.name}  {c.duration:.1f}s"
                if speed_label(c.speed, c.ramp):
                    label += f"  ⏱{speed_label(c.speed, c.ramp)}"
                if c.opacity < 0.999:
                    label += f"  {c.opacity * 100:.0f}%"
                p.drawText(QRectF(r.left() + 4, r.top(), r.width() - 8, head), Qt.AlignVCenter | Qt.AlignLeft,
                           p.fontMetrics().elidedText(label, Qt.ElideRight, int(r.width() - 8)))
            p.restore()
            if c.id in self._ov_selection and c.id != self.selected_overlay:
                p.setPen(QPen(QColor(255, 255, 255, 170), 2))
                p.setBrush(QColor(255, 255, 255, 28))
                p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 3, 3)
            if c.id == self.selected_overlay:
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 3, 3)
        for k in range(self.layers):   # layer names on the left edge
            rr = QRectF(2, self._layer_top(k) + 1, 24, min(14, self.ov_row - 2))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 140))
            p.drawRoundedRect(rr, 2, 2)
            p.setPen(QColor("#9aa0ab"))
            p.drawText(rr, Qt.AlignCenter, f"V{k + 2}")

    def select_overlay(self, oid: str):
        self.selected_overlay = oid
        self._ov_selection = {oid} if oid else set()
        if oid:
            if self._selected:
                self.selected = ""
                self.selection_changed.emit("")
            if self.selected_sound:
                self.selected_sound = ""
                self.sound_selected.emit("")
            if self.selected_text:
                self.selected_text = ""
                self.text_selected.emit("")
        self.overlay_selected.emit(oid)
        self.update()

    def _overlay_press(self, e, pos) -> bool:
        c, part = self._overlay_hit(pos)
        if part is None:
            return False
        t = self.x_to_t(pos.x())
        if c is None:
            if e.button() == Qt.RightButton:
                self._layer_menu(e.globalPosition().toPoint(), t, self._track_at(pos.y()))
                return True
            self.select_overlay("")
            self.seek.emit(t)
            self._drag = {"kind": "scrub"}
            return True
        if e.button() == Qt.LeftButton and e.modifiers() & Qt.ControlModifier:
            # Ctrl+click: add the layer clip to the selection (or take it out), keeping the rest of it.
            if c.id in self._ov_selection:
                self._ov_selection.discard(c.id)
            else:
                self._ov_selection.add(c.id)
            self.selection_changed.emit(self._selected)
            self.update()
            return True
        self.select_overlay(c.id)
        if e.button() == Qt.RightButton:
            self._overlay_menu(e.globalPosition().toPoint(), c)
            return True
        main = self._linked_main(c)
        if main is not None:
            # A synced camera angle is a handle on its multicam group: its edges trim, its body slips the main
            # clip it follows, and every angle follows that (so they all stay in sync).
            kind = {"o_trim_l": "trim_l", "o_trim_r": "trim_r"}.get(part, "slip")
            self._drag = {"kind": kind, "clip": main.id, "x0": pos.x(), "in0": main.src_in, "out0": main.src_out,
                          "active": False, "drop": None, "mods": Qt.NoModifier, "angle": c.id}
            return True
        tm = c.timemap()
        self._drag = {"kind": part, "overlay": c.id, "x0": pos.x(), "start0": c.start, "in0": c.src_in,
                      "out0": c.src_out, "track0": c.track, "active": False, "v_in": tm.v_in, "v_out": tm.v_out}
        return True

    def _linked_main(self, c: OverlayClip):
        """The main-track clip a synced camera angle follows (None for other layer clips and unlinked angles)."""
        if not c.link or not c.linked:
            return None
        return next((x for x in self.project.clips if x.id == c.link), None)

    def _has_angles(self, clip_id: str) -> bool:
        return any(o.link == clip_id and o.linked for o in self.project.overlays)

    def _overlay_move(self, pos, d: dict):
        c = next((x for x in self.project.overlays if x.id == d["overlay"]), None)
        if c is None:
            return
        dx = pos.x() - d["x0"]
        track = self._track_at(pos.y()) if d["kind"] == "o_move" else c.track
        if not d["active"]:
            if abs(dx) < 3 and track == d["track0"]:
                return
            d["active"] = True
            self.edit_started.emit()
        dt = dx / self.pps
        snaps = [self.playhead] + self.project.clip_starts() + [self.project.duration] + \
            [t for x in self.project.overlays if x.id != c.id for t in (x.start, x.end)]
        f = self.project.footage.get(c.footage_id)
        if d["kind"] == "o_move":
            start = max(0.0, d["start0"] + dt)
            dur = c.duration
            for t in snaps:
                if abs((start - t) * self.pps) < 6:
                    start = t
                    break
                if abs((start + dur - t) * self.pps) < 6:
                    start = t - dur
                    break
            c.start = max(0.0, self._snap(start))
            c.track = track
        elif d["kind"] == "o_trim_l":
            v = d.get("v_in", 1.0)          # timeline seconds -> source seconds at the clip's speed
            new_in = min(max(0.0, d["in0"] + dt * v), d["out0"] - MIN_DUR)
            new_in = max(new_in, d["in0"] - d["start0"] * v)      # can't start before the timeline
            c.src_in = new_in
            c.start = d["start0"] + (new_in - d["in0"]) / v
        else:
            limit = f.duration if f is not None else d["out0"]
            c.src_out = max(min(limit, d["out0"] + dt * d.get("v_out", 1.0)), d["in0"] + MIN_DUR)
        self.update()
        self.overlays_changed.emit()

    def _overlay_menu(self, gpos, c: OverlayClip):
        m = QMenu(self)
        oid = c.id
        self._menu_action(m, "Move to playhead", "ov_to_playhead", oid)
        self._menu_action(m, "Split at playhead", "ov_split", oid)
        self._menu_action(m, "Duplicate", "ov_duplicate", oid)
        if c.link:
            self._menu_action(m, "Unlink from the main clip\tCtrl+L" if c.linked else
                              "Link to the main clip again (back in sync)\tCtrl+L", "link_toggle", oid)
        sub = m.addMenu("Position && size")
        for key, label in (("full", "Full frame"), ("center", "Centre (half size)"),
                           ("tl", "Picture in picture: top left"), ("tr", "Picture in picture: top right"),
                           ("bl", "Picture in picture: bottom left"), ("br", "Picture in picture: bottom right")):
            self._menu_action(sub, label, f"ov_place:{key}", oid)
        sub = m.addMenu("Rotate")
        for key, label in (("-90", "⟲ 90° left"), ("90", "⟳ 90° right"), ("0", "Straight (0°)")):
            self._menu_action(sub, label, f"ov_rotate:{key}", oid)
        if not (self.project.footage.get(c.footage_id) and self.project.footage[c.footage_id].is_image):
            self._speed_menu(m, "ov_", oid, c.speed, c.ramp, audio=c.audio_mode)
        sub = m.addMenu("Color")
        self._menu_action(sub, "Color, log && LUT…", "ov_color", oid)
        self._menu_action(sub, "Auto color", "ov_auto_color", oid)
        self._menu_action(sub, "Reset color", "ov_reset_color", oid)
        self._menu_action(sub, "Copy this grade to all clips (every layer)", "ov_copy_grade", oid)
        for key, label, cur in (("ov_fade_in", "Fade in", c.fade_in), ("ov_fade_out", "Fade out", c.fade_out)):
            sub = m.addMenu(f"{label} ({cur:g} s)" if cur > 0 else label)
            for v in (0.0, 0.25, 0.5, 1.0, 2.0):
                a = sub.addAction("Off" if v == 0 else f"{v:g} s")
                a.setCheckable(True)
                a.setChecked(abs(cur - v) < 1e-3)
                a.triggered.connect(lambda _=False, k=key, v=v: self.context_action.emit(f"{k}:{v}", oid))
        sub = m.addMenu("Layer")
        if c.track + 1 < MAX_LAYERS:
            self._menu_action(sub, "Up (in front)", "ov_layer:up", oid)
        if c.track > 0:
            self._menu_action(sub, "Down (behind)", "ov_layer:down", oid)
        if c.angle:
            from ..multicam import TEMPLATES
            sub = m.addMenu("Multicam layout (all angles)")
            for key, label in TEMPLATES.items():
                self._menu_action(sub, label, f"multicam_layout:{key}", oid)
        self._menu_action(m, "Settings…\tDouble-click", "ov_edit", oid)
        self._menu_action(m, "Show in Media panel", "show_media", c.footage_id)
        m.addSeparator()
        self._menu_action(m, "Delete\tDel", "ov_delete", oid)
        m.exec(gpos)

    def _layer_menu(self, gpos, t: float, track: int):
        m = QMenu(self)
        self._menu_action(m, f"Add the selected media here (V{track + 2})", "ov_add_media", f"{t:.4f}:{track}")
        m.addSeparator()
        if self.layers < MAX_LAYERS:
            self._menu_action(m, "Add a video layer", "layer_add", "")
        top = self.layers - 1
        if top > 0 and not any(c.track == top for c in self.project.overlays):
            self._menu_action(m, f"Remove the empty top layer (V{top + 2})", "layer_remove", "")
        m.addSeparator()
        self._menu_action(m, "Move playhead here", "seek", f"{t:.4f}")
        self._menu_action(m, "Fit timeline\tShift+Z", "fit", "")
        m.exec(gpos)

    # ------------------------------------------------------------------ sound-effect layer
    def _sound_rows(self) -> list[tuple[SoundClip, int]]:
        """Each sound with its display row (0/1): overlapping sounds alternate rows."""
        return self._rows(self.project.sounds)

    def _sound_rects(self) -> list[tuple[SoundClip, QRectF]]:
        out = []
        for x, row in self._sound_rows():
            x0, x1 = self.t_to_x(x.start), self.t_to_x(x.end)
            out.append((x, QRectF(x0, self.sfx_y + row * self.sfx_row + 1, max(3.0, x1 - x0), self.sfx_row - 2)))
        return out

    def _sound_hit(self, pos: QPointF):
        if not (self.sfx_y <= pos.y() <= self.sfx_y + self.sfx_h):
            return None, None
        for x, r in reversed(self._sound_rects()):
            if r.left() - 2 <= pos.x() <= r.right() + 2 and r.top() <= pos.y() <= r.bottom():
                if pos.x() - r.left() <= 5 and r.width() > 15:
                    return x, "s_trim_l"
                if r.right() - pos.x() <= 5 and r.width() > 15:
                    return x, "s_trim_r"
                return x, "s_move"
        return None, "lane"

    def _paint_sounds(self, p: QPainter):
        p.fillRect(QRectF(0, self.sfx_y, self.width(), self.sfx_h), QColor("#1b1d21"))
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        if not self.project.sounds:
            p.setPen(QColor("#4d5058"))
            p.drawText(QRectF(8, self.sfx_y, self.width() - 16, self.sfx_h), Qt.AlignVCenter | Qt.AlignLeft,
                       "Sound effects — drop audio files here, or add them from the Sounds tab")
            return
        for x, r in self._sound_rects():
            if r.right() < 0 or r.left() > self.width():
                continue
            sel = x.id == self.selected_sound
            p.setPen(QPen(QColor("#ffffff"), 1.5) if sel else Qt.NoPen)
            p.setBrush(QColor("#b9672c") if sel else QColor("#8f5428"))
            p.drawRoundedRect(r, 3, 3)
            env = sfx.envelope(x.path) if os.path.exists(x.path) else None
            if env is not None and r.width() > 6:
                xs = np.arange(max(r.left(), 0), min(r.right(), self.width()), 2.0)
                if xs.size > 1:
                    ts = x.source_time((xs - r.left()) / self.pps)
                    amp = env[np.clip((ts * sfx.ENVELOPE_RATE).astype(int), 0, env.size - 1)] * min(1.0, x.volume)
                    mid, hh = r.center().y(), (r.height() / 2 - 1) * amp
                    poly = QPolygonF([QPointF(a, mid - h) for a, h in zip(xs, hh)] +
                                     [QPointF(a, mid + h) for a, h in zip(xs[::-1], hh[::-1])])
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(255, 220, 180, 110))
                    p.drawPolygon(poly)
            self._paint_sound_fades(p, x, r)
            if x.looping:   # where each repeat crossfades into the next
                p.setPen(QPen(QColor(255, 244, 234, 150), 1, Qt.DashLine))
                for n in range(1, x.repeats()):
                    sx = r.left() + n * x.loop_step() * self.pps
                    if 0 <= sx <= self.width():
                        p.drawLine(QPointF(sx, r.top() + 2), QPointF(sx, r.bottom() - 2))
            if r.width() > 24:
                p.setPen(QColor("#fff4ea"))
                label = ("⇩ " if x.duck else "") + ("🔁 " if x.looping else "") + \
                    (x.name or os.path.splitext(os.path.basename(x.path))[0])
                if x.looping:
                    label += f"  ×{x.repeats()}"
                if abs(x.volume - 1.0) > 0.01:
                    label += f"  {x.volume * 100:.0f}%"
                p.drawText(r.adjusted(4, 0, -3, 0), Qt.AlignVCenter | Qt.AlignLeft,
                           p.fontMetrics().elidedText(label, Qt.ElideRight, int(r.width() - 7)))

    def _paint_sound_fades(self, p: QPainter, x: SoundClip, r: QRectF):
        """Fade in / out as shaded corners with a ramp line."""
        fi, fo = x.fades()
        p.setPen(QPen(QColor(255, 244, 234, 200), 1))
        p.setBrush(QColor(0, 0, 0, 90))
        for w, left in ((fi * self.pps, True), (fo * self.pps, False)):
            if w < 2:
                continue
            if left:
                p.drawPolygon(QPolygonF([r.topLeft(), QPointF(r.left() + w, r.top()), r.bottomLeft()]))
            else:
                p.drawPolygon(QPolygonF([QPointF(r.right() - w, r.top()), r.topRight(), r.bottomRight()]))

    # ------------------------------------------------------------------ text track
    @staticmethod
    def _rows(items) -> list[tuple[object, int]]:
        """Each item (with .start / .end) and its display row (0/1): overlapping items alternate rows."""
        out, row_end = [], [-1e9, -1e9]
        for x in sorted(items, key=lambda x: x.start):
            row = 0 if row_end[0] <= x.start + 1e-6 else 1 if row_end[1] <= x.start + 1e-6 else \
                (0 if row_end[0] <= row_end[1] else 1)
            row_end[row] = max(row_end[row], x.end)
            out.append((x, row))
        return out

    def _text_rects(self) -> list[tuple[object, QRectF]]:
        out = []
        for x, row in self._rows(self.project.texts):
            x0, x1 = self.t_to_x(x.start), self.t_to_x(x.end)
            out.append((x, QRectF(x0, TEXT_Y + row * TEXT_ROW + 1, max(3.0, x1 - x0), TEXT_ROW - 2)))
        return out

    def _text_hit(self, pos: QPointF):
        if not (TEXT_Y <= pos.y() <= TEXT_Y + TEXT_H):
            return None, None
        for x, r in reversed(self._text_rects()):
            if r.left() - 2 <= pos.x() <= r.right() + 2 and r.top() <= pos.y() <= r.bottom():
                if pos.x() - r.left() <= 5 and r.width() > 15:
                    return x, "t_trim_l"
                if r.right() - pos.x() <= 5 and r.width() > 15:
                    return x, "t_trim_r"
                return x, "t_move"
        return None, "lane"

    def _paint_texts(self, p: QPainter):
        p.fillRect(QRectF(0, TEXT_Y, self.width(), TEXT_H), QColor("#1b1d21"))
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        if not self.project.texts:
            p.setPen(QColor("#4d5058"))
            p.drawText(QRectF(8, TEXT_Y, self.width() - 16, TEXT_H), Qt.AlignVCenter | Qt.AlignLeft,
                       "Titles & text — right-click here to add one, or use the Text tab")
            return
        for x, r in self._text_rects():
            if r.right() < 0 or r.left() > self.width():
                continue
            sel = x.id == self.selected_text
            p.setPen(QPen(QColor("#ffffff"), 1.5) if sel else Qt.NoPen)
            p.setBrush(QColor("#2f9e8f") if sel else QColor("#237a6f"))
            p.drawRoundedRect(r, 3, 3)
            if r.width() > 18:
                p.setPen(QColor("#eafffb"))
                label = "T  " + x.text.replace("\n", " / ")
                p.drawText(r.adjusted(4, 0, -3, 0), Qt.AlignVCenter | Qt.AlignLeft,
                           p.fontMetrics().elidedText(label, Qt.ElideRight, int(r.width() - 7)))

    def select_text(self, text_id: str):
        self.selected_text = text_id
        if text_id:
            if self._selected:
                self.selected = ""
                self.selection_changed.emit("")
            self.selected_sound = ""
            self._clear_overlay()
        self.text_selected.emit(text_id)
        self.update()

    def _text_press(self, e, pos) -> bool:
        x, part = self._text_hit(pos)
        if part is None:
            return False
        t = self.x_to_t(pos.x())
        if x is None:     # empty part of the lane
            if e.button() == Qt.RightButton:
                self._lane_menu(e.globalPosition().toPoint(), t)
                return True
            self.select_text("")
            self.seek.emit(t)
            self._drag = {"kind": "scrub"}
            return True
        self.select_text(x.id)
        if e.button() == Qt.RightButton:
            self._text_menu(e.globalPosition().toPoint(), x.id)
            return True
        self._drag = {"kind": part, "text": x.id, "x0": pos.x(), "start0": x.start, "dur0": x.duration,
                      "active": False}
        return True

    def _text_move(self, pos, d: dict):
        x = next((s for s in self.project.texts if s.id == d["text"]), None)
        if x is None:
            return
        dx = pos.x() - d["x0"]
        if not d["active"]:
            if abs(dx) < 3:
                return
            d["active"] = True
            self.edit_started.emit()
        dt = dx / self.pps
        snaps = [self.playhead] + self.project.clip_starts() + [self.project.duration]
        if d["kind"] == "t_move":
            start = max(0.0, d["start0"] + dt)
            for t in snaps:   # snap either edge to the playhead and the cuts
                if abs((start - t) * self.pps) < 6:
                    start = t
                    break
                if abs((start + x.duration - t) * self.pps) < 6:
                    start = t - x.duration
                    break
            x.start = max(0.0, self._snap(start))
        elif d["kind"] == "t_trim_l":
            end = d["start0"] + d["dur0"]
            start = min(max(0.0, d["start0"] + dt), end - MIN_DUR)
            start = next((t for t in snaps if abs((start - t) * self.pps) < 6 and t < end - MIN_DUR), start)
            x.start, x.duration = start, end - start
        else:
            end = max(d["start0"] + MIN_DUR, d["start0"] + d["dur0"] + dt)
            end = next((t for t in snaps if abs((end - t) * self.pps) < 6 and t > d["start0"] + MIN_DUR), end)
            x.duration = end - d["start0"]
        self.update()
        self.texts_changed.emit()

    def _speed_menu(self, m: QMenu, prefix: str, payload: str, speed: float, ramp: str, many: str = "",
                    audio: str = "sync"):
        """Speed submenus: constant slow / fast motion, the ramp templates and what the sound does (actions
        {prefix}speed:<x>, {prefix}speed_custom, {prefix}ramp:<key>, {prefix}audio:<mode>)."""
        cur = speed_label(speed, ramp)
        sub = m.addMenu(f"Speed{many} ({cur})" if cur else f"Speed{many}")
        for v in SPEED_PRESETS:
            name = "Normal (1×)" if v == 1.0 else (f"{v:g}× slow motion" if v < 1 else f"{v:g}× fast motion")
            a = self._menu_action(sub, name, f"{prefix}speed:{v:g}", payload)
            a.setCheckable(True)
            a.setChecked(abs(speed - v) < 1e-6)
        self._menu_action(sub, "Custom speed…", f"{prefix}speed_custom", payload)
        sub = m.addMenu("Speed ramp" + (f" ({RAMPS[ramp][0]})" if ramp in RAMPS else ""))
        a = self._menu_action(sub, "None (constant speed)", f"{prefix}ramp:", payload)
        a.setCheckable(True)
        a.setChecked(ramp not in RAMPS)
        for k, (name, desc, _keys) in RAMPS.items():
            a = self._menu_action(sub, name, f"{prefix}ramp:{k}", payload)
            a.setCheckable(True)
            a.setChecked(ramp == k)
            a.setToolTip(desc)
        sub.setToolTipsVisible(True)
        sub = m.addMenu("Sound at this speed")
        for k, (name, desc) in SOUND_MODES.items():
            a = self._menu_action(sub, name, f"{prefix}audio:{k}", payload)
            a.setCheckable(True)
            a.setChecked(sound_mode(audio) == k)
            a.setToolTip(desc)
        sub.setToolTipsVisible(True)

    def _menu_action(self, m: QMenu, label: str, key: str, payload: str):
        a = QAction(label, m)
        a.triggered.connect(lambda _=False: self.context_action.emit(key, payload))
        m.addAction(a)
        return a

    def _add_text_menu(self, m: QMenu, t: float, title: str = "Add text here"):
        sub = m.addMenu(title)
        for key, cfg in TEXT_PRESETS.items():
            self._menu_action(sub, cfg["label"], f"text_add:{key}", f"{t:.4f}")

    def _text_menu(self, gpos, text_id: str):
        m = QMenu(self)
        self._menu_action(m, "Edit text…\tDouble-click", "text_edit", text_id)
        self._menu_action(m, "Move to playhead", "text_to_playhead", text_id)
        self._menu_action(m, "Duplicate", "text_duplicate", text_id)
        for title, table, prefix in (("Animation in", ANIM_IN, "text_anim_in"), ("Animation out", ANIM_OUT, "text_anim_out")):
            sub = m.addMenu(title)
            for k, label in table.items():
                self._menu_action(sub, label, f"{prefix}:{k}", text_id)
        sub = m.addMenu("Position")
        for k, label in (("top", "Top"), ("middle", "Middle"), ("bottom", "Bottom"), ("lower_left", "Lower left")):
            self._menu_action(sub, label, f"text_pos:{k}", text_id)
        m.addSeparator()
        self._menu_action(m, "Delete text\tDel", "text_delete", text_id)
        m.exec(gpos)

    def _lane_menu(self, gpos, t: float):
        """Right-click on an empty part of the timeline."""
        m = QMenu(self)
        self._add_text_menu(m, t)
        self._menu_action(m, "Add sound effect…", "open_sounds", "")
        if self.layers < MAX_LAYERS:
            self._menu_action(m, "Add a video layer", "layer_add", "")
        m.addSeparator()
        self._menu_action(m, "Move playhead here", "seek", f"{t:.4f}")
        self._black_menu(m, t)
        if self.project.clips:
            self._menu_action(m, "Split at playhead\tS", "split_playhead", "")
            self._menu_action(m, "Select all clips\tCtrl+A", "select_all", "")
        self._menu_action(m, "Fit timeline\tShift+Z", "fit", "")
        m.addSeparator()
        self._menu_action(m, "Generate captions", "captions_generate", "")
        m.exec(gpos)

    def _black_menu(self, m: QMenu, t: float, title: str = "Insert black here"):
        sub = m.addMenu(title)
        for label, length in gaps.BLACK_LENGTHS:
            self._menu_action(sub, label, "insert_black", f"{length}@{t:.4f}")

    def _caption_menu(self, gpos, index: int, t: float):
        m = QMenu(self)
        if index >= 0:
            self._menu_action(m, "Edit caption…", "caption_edit", str(index))
            sub = m.addMenu("Move this and the clip's later captions")
            for dt in (-0.5, -0.1, 0.1, 0.5):
                a = sub.addAction(f"{abs(dt):g} s {'earlier' if dt < 0 else 'later'}")
                a.triggered.connect(lambda _=False, dt=dt: self.captions_shifted.emit(index, dt))
            self._menu_action(m, "Auto-sync this clip's captions to its speech", "caption_sync", str(index))
            sub = m.addMenu("Speaker")
            current = self.captions[index].speaker if index < len(self.captions) else -1
            for k in range(4):
                a = self._menu_action(sub, f"Speaker {k + 1}", "caption_speaker", f"{index}:{k}")
                a.setCheckable(True)
                a.setChecked(current == k)
            a = self._menu_action(sub, "None", "caption_speaker", f"{index}:-1")
            a.setCheckable(True)
            a.setChecked(current < 0)
            self._menu_action(m, "Delete caption", "caption_delete", str(index))
            m.addSeparator()
        if self.captions:
            self._menu_action(m, "Delete all captions…", "caption_delete_all", "")
        self._menu_action(m, "Generate captions", "captions_generate", "")
        self._menu_action(m, "Caption style…", "caption_style", "")
        m.addSeparator()
        self._add_text_menu(m, t)
        m.exec(gpos)

    def _clear_overlay(self):
        self._ov_selection = set()
        if self.selected_overlay:
            self.selected_overlay = ""
            self.overlay_selected.emit("")

    def _select_sound(self, sound_id: str):
        self.selected_sound = sound_id
        if sound_id and self._selected:
            self.selected = ""
            self.selection_changed.emit("")
        if sound_id and self.selected_text:
            self.select_text("")
        if sound_id:
            self._clear_overlay()
        self.sound_selected.emit(sound_id)
        self.update()

    def _sound_press(self, e, pos) -> bool:
        x, part = self._sound_hit(pos)
        if part is None:
            return False
        if x is None:     # empty part of the lane: deselect and move the playhead
            self._select_sound("")
            self.seek.emit(self.x_to_t(pos.x()))
            self._drag = {"kind": "scrub"}
            return True
        self._select_sound(x.id)
        if e.button() == Qt.RightButton:
            self._sound_menu(e.globalPosition().toPoint(), x.id)
            return True
        if x.path not in self._sound_max:
            self._sound_max[x.path] = sfx.duration(x.path) or x.src_out
        self._drag = {"kind": part, "sound": x.id, "x0": pos.x(), "start0": x.start, "in0": x.src_in,
                      "out0": x.src_out, "loop0": x.loop, "active": False}
        return True

    def _sound_move(self, pos, d: dict):
        x = next((s for s in self.project.sounds if s.id == d["sound"]), None)
        if x is None:
            return
        dx = pos.x() - d["x0"]
        if not d["active"]:
            if abs(dx) < 3:
                return
            d["active"] = True
            self.edit_started.emit()
        dt = dx / self.pps
        if d["kind"] == "s_move":
            start = max(0.0, d["start0"] + dt)
            # Snap the start to the playhead and to video cuts within a few pixels.
            for t in [self.playhead] + self.project.clip_starts():
                if abs((start - t) * self.pps) < 6:
                    start = t
                    break
            x.start = self._snap(start)
        elif d["kind"] == "s_trim_l":
            new_in = min(max(0.0, d["in0"] + dt), d["out0"] - 0.05)
            new_in = max(new_in, d["in0"] - d["start0"])          # the sound can't start before 0
            x.src_in = new_in
            x.start = d["start0"] + (new_in - d["in0"])
            if x.loop > 0:   # the repeated part can't run past the end of the file
                x.loop = max(0.0, min(d.get("loop0", x.loop), self._sound_max.get(x.path, 1e9) - new_in))
        elif x.looping:   # a looping sound can be made as long as wanted (it keeps repeating)
            x.src_out = max(d["out0"] + dt, d["in0"] + x.loop + 0.05)
        else:
            x.src_out = max(min(self._sound_max.get(x.path, d["out0"]), d["out0"] + dt), d["in0"] + 0.05)
        self.update()

    def _sound_menu(self, gpos, sound_id: str):
        m = QMenu(self)
        for key, label, tip in (("sound_to_playhead", "Move to playhead", ""), ("sound_preview", "Preview", ""),
                                ("sound_fit", "Fit to timeline",
                                 "Cover the whole timeline with this one sound: it loops with crossfades if it is "
                                 "shorter, and is cut with a fade out if it is longer")):
            a = QAction(label, m)
            a.setToolTip(tip)
            a.triggered.connect(lambda _=False, k=key: self.context_action.emit(k, sound_id))
            m.addAction(a)
        m.setToolTipsVisible(True)
        x = next((s for s in self.project.sounds if s.id == sound_id), None)
        if x is not None:
            a = QAction("Auto ducking (lower it under the clips' sound)", m)
            a.setCheckable(True)
            a.setChecked(x.duck)
            a.setToolTip("Turn this sound (music) down automatically while the clips have speech or sound")
            a.triggered.connect(lambda on: self.context_action.emit(f"sound_duck:{int(on)}", sound_id))
            m.addAction(a)
            a = QAction("Caption its speech (voice-over)", m)
            a.setCheckable(True)
            a.setChecked(x.caption)
            a.setToolTip("A voice-over recorded on its own: what is said in it goes into the captions too "
                         "(transcribed with the speech model set in the Captions panel)")
            a.triggered.connect(lambda on: self.context_action.emit(f"sound_caption:{int(on)}", sound_id))
            m.addAction(a)
            for key, label, cur in (("sound_fade_in", "Fade in", x.fade_in), ("sound_fade_out", "Fade out", x.fade_out)):
                sub = m.addMenu(f"{label} ({cur:g} s)" if cur > 0 else label)
                for v in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
                    a = sub.addAction("Off" if v == 0 else f"{v:g} s")
                    a.setCheckable(True)
                    a.setChecked(abs(cur - v) < 1e-3)
                    a.triggered.connect(lambda _=False, k=key, v=v: self.context_action.emit(f"{k}:{v}", sound_id))
            m.addSeparator()
            box = QWidget()
            row = QHBoxLayout(box)
            row.setContentsMargins(12, 4, 12, 4)
            lbl = QLabel(f"Volume {x.volume * 100:.0f}%")
            lbl.setMinimumWidth(80)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 200)
            sl.setValue(round(x.volume * 100))
            sl.setMinimumWidth(140)
            sl.valueChanged.connect(lambda v: (lbl.setText(f"Volume {v}%"),
                                               self.context_action.emit(f"sound_volume:{v / 100}", sound_id)))
            row.addWidget(lbl)
            row.addWidget(sl, 1)
            act = QWidgetAction(m)
            act.setDefaultWidget(box)
            m.addAction(act)
        m.addSeparator()
        a = QAction("Delete sound\tDel", m)
        a.triggered.connect(lambda: self.context_action.emit("sound_delete", sound_id))
        m.addAction(a)
        m.exec(gpos)

    def _drop_paths(self, e) -> list[str]:
        md = e.mimeData()
        if not md.hasUrls():
            return []
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile()
                and u.toLocalFile().lower().endswith(sfx.AUDIO_EXTENSIONS)]

    @staticmethod
    def _drop_footage(e) -> list[str]:
        md = e.mimeData()
        if not md.hasFormat(FOOTAGE_MIME):
            return []
        return [x for x in bytes(md.data(FOOTAGE_MIME)).decode("utf-8").splitlines() if x]

    def dragEnterEvent(self, e):
        if self._drop_paths(e) or self._drop_footage(e):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if self._drop_paths(e) or self._drop_footage(e):
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = self._drop_paths(e)
        pos = e.position()
        t = self._snap(self.x_to_t(pos.x()))
        if paths:
            e.acceptProposedAction()
            self.sound_dropped.emit(paths, t)
            return
        ids = self._drop_footage(e)
        if ids:
            e.acceptProposedAction()
            # Onto a video layer: an overlay there. Anywhere else: into the main track at the nearest cut.
            track = self._track_at(pos.y()) if pos.y() < OV_Y + self.ov_h else -1
            self.footage_dropped.emit(ids, t, track)

    def mousePressEvent(self, e):
        pos = e.position()
        edge = self._edge_at(pos.y()) if e.button() == Qt.LeftButton else ""
        if edge:
            self._drag = {"kind": "resize", "lane": edge}
            return
        if e.button() in (Qt.LeftButton, Qt.RightButton) and self._text_press(e, pos):
            return
        if e.button() in (Qt.LeftButton, Qt.RightButton) and self._overlay_press(e, pos):
            return
        if e.button() == Qt.RightButton and self.sfx_y <= pos.y() <= self.sfx_y + self.sfx_h and self._sound_hit(pos)[0] is None:
            self._lane_menu(e.globalPosition().toPoint(), self.x_to_t(pos.x()))
            return
        if e.button() in (Qt.LeftButton, Qt.RightButton) and self._sound_press(e, pos):
            return
        if self.selected_sound and e.button() == Qt.LeftButton:
            self._select_sound("")
        if self.selected_text and e.button() == Qt.LeftButton:
            self.select_text("")
        if self.selected_overlay and e.button() == Qt.LeftButton:
            self.select_overlay("")
        if e.button() == Qt.RightButton:
            t = self.x_to_t(pos.x())
            if self.cap_y <= pos.y() <= self.cap_y + CAP_H:
                idx = next((i for i, cap in enumerate(self.captions) if cap.start <= t < cap.end), -1)
                self._caption_menu(e.globalPosition().toPoint(), idx, t)
                return
            c, _ = self._hit(pos)
            if c is None:
                self._lane_menu(e.globalPosition().toPoint(), t)
                return
            if c is not None:
                if c.id not in self._selection:   # keep a multi-selection the clip belongs to
                    self.selected = c.id
                self._selected = c.id
                self.selection_changed.emit(c.id)
                self.update()
                self._context_menu(e.globalPosition().toPoint(), c.id)
            return
        if e.button() != Qt.LeftButton:
            return
        if pos.y() < RULER:
            self._drag = {"kind": "scrub"}
            self.seek.emit(self.x_to_t(pos.x()))
            return
        if self.cap_y <= pos.y() <= self.cap_y + CAP_H:
            t = self.x_to_t(pos.x())
            for i, cap in enumerate(self.captions):
                if cap.start <= t < cap.end:
                    self.caption_clicked.emit(i)
                    # dragged sideways: it and the clip's later captions move in time (see captions_shifted)
                    self._drag = {"kind": "cap_shift", "cap": i, "x0": pos.x(), "active": False,
                                  "group": shift_group(self.captions, i)}
                    break
            self.seek.emit(t)
            return
        c, part = self._hit(pos)
        if c is None:
            self.selected = ""
            self.selection_changed.emit("")
            self.seek.emit(self.x_to_t(pos.x()))
            self._drag = {"kind": "scrub"}
            self.update()
            return
        mods = e.modifiers()
        if mods & Qt.ControlModifier:
            self._click_select(c.id, mods)
            return
        # Shift+click selects a range, Shift+drag slips: decided on release (see mouseReleaseEvent).
        if not (mods & Qt.ShiftModifier) and (c.id not in self._selection or len(self._selection) == 1):
            if self.selected != c.id:
                self.selected = c.id
                self.selection_changed.emit(c.id)
        kind = part
        if part == "body":
            slip = self.mode == "slip" or (mods & (Qt.AltModifier | Qt.ShiftModifier))
            kind = "slip" if slip else "move"
        tm = c.timemap()
        self._drag = {"kind": kind, "clip": c.id, "x0": pos.x(), "in0": c.src_in, "out0": c.src_out,
                      "active": False, "drop": None, "mods": mods, "v_in": tm.v_in, "v_out": tm.v_out,
                      "v": tm.speed, "dur0": c.duration}
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        d = self._drag
        if d is None:
            if self._edge_at(pos.y()):
                self.setCursor(Qt.SplitVCursor)
                self.setToolTip("Drag to change the height of this lane (double-click: usual height)")
                return
            if self.cap_y <= pos.y() <= self.cap_y + CAP_H:
                t = self.x_to_t(pos.x())
                on = any(c.start <= t < c.end for c in self.captions)
                self.setCursor(Qt.SizeHorCursor if on else Qt.ArrowCursor)
                self.setToolTip("Click: go there · Drag left / right: move this caption and the clip's later "
                                "ones in time (right words, wrong timing) · Right-click: edit, nudge, delete"
                                if on else "")
                return
            oc, opart = self._overlay_hit(pos)
            if oc is not None:
                self.setCursor(Qt.SizeHorCursor if opart in ("o_trim_l", "o_trim_r") else Qt.OpenHandCursor)
                f = self.project.footage.get(oc.footage_id)
                linked = self._linked_main(oc) is not None
                what = ((" · facecam, linked to the main clip" if oc.role == "facecam" else
                         f" · camera angle {oc.angle}, linked to the main clip") if linked else "")
                marks = marks_text(oc)
                self.setToolTip(f"{f.name if f else ''} on layer V{oc.track + 2}{what}\n"
                                f"{oc.start:.2f}s, {oc.duration:.2f}s long\n" + (marks + "\n" if marks else "")
                                + ("Drag: slip every angle together · Edges: trim every angle · R: other moment\n"
                                   if linked else "Drag: move (also up / down to another layer) · Edges: trim\n")
                                + "In the viewer: drag to place it, wheel to resize\n"
                                "Double-click: size, crop, rotation, opacity · Right-click: more")
                return
            tx, tpart = self._text_hit(pos)
            if tx is not None:
                self.setCursor(Qt.SizeHorCursor if tpart in ("t_trim_l", "t_trim_r") else Qt.OpenHandCursor)
                self.setToolTip(f"{tx.text}\n{tx.start:.2f}s, {tx.duration:.2f}s long\n"
                                "Drag: move · Edges: change length · Double-click: edit · Right-click: more")
                return
            x, spart = self._sound_hit(pos)
            if x is not None:
                self.setCursor(Qt.SizeHorCursor if spart in ("s_trim_l", "s_trim_r") else Qt.OpenHandCursor)
                self.setToolTip(f"{x.name}\n{x.start:.2f}s, {x.duration:.2f}s long, volume {x.volume * 100:.0f}%\n"
                                "Drag: move · Edges: trim · Right-click: volume / delete")
                return
            c, part = self._hit(pos)
            if part in ("trim_l", "trim_r"):
                self.setCursor(Qt.SizeHorCursor)
            elif part == "body":
                slip = self.mode == "slip" or (e.modifiers() & (Qt.AltModifier | Qt.ShiftModifier))
                self.setCursor(Qt.OpenHandCursor if slip else Qt.PointingHandCursor)
                f = self.project.footage.get(c.footage_id)
                spd = speed_label(c.speed, c.ramp)
                marks = marks_text(c)
                self.setToolTip(f"{f.name if f else ''}\nSource {c.src_in:.2f}s → {c.src_out:.2f}s"
                                + (f" · speed {spd}" if spd else "") + "\n" + (marks + "\n" if marks else "")
                                + ("Drag: reorder" if self.magnetic else "Drag: place anywhere (magnet off)")
                                + " · Alt/Shift+drag: slip · Edges: trim")
            else:
                self.setCursor(Qt.ArrowCursor)
            return
        if d["kind"] == "cap_shift":
            dx = pos.x() - d["x0"]
            if not d["active"] and abs(dx) < 4:
                return
            d["active"] = True
            d["dt"] = dx / self.pps
            self.setCursor(Qt.SizeHorCursor)
            self.update()
            return
        if d["kind"] == "resize":
            lane = d["lane"]
            if lane == "layer":
                self.set_heights({"layer": (pos.y() - OV_Y) / self.layers})
            elif lane == "video":
                self.set_heights({"video": pos.y() - self.track_y})
            else:
                self.set_heights({"sound": (pos.y() - self.sfx_y) / 2})
            return
        if "overlay" in d:
            self._overlay_move(pos, d)
            return
        if d["kind"] == "scrub":
            self._scrub_x = pos.x()
            self.seek.emit(self.x_to_t(min(max(0.0, pos.x()), self.width())))
            if self._edge_speed(pos.x()) and not self._edge_timer.isActive():
                self._edge_timer.start()
            return
        if "sound" in d:
            self._sound_move(pos, d)
            return
        if "text" in d:
            self._text_move(pos, d)
            return
        idx = self.project.index_of(d["clip"])
        if idx < 0:
            return
        c = self.project.clips[idx]
        f = self.project.footage[c.footage_id]
        dx = pos.x() - d["x0"]
        if not d["active"]:
            if abs(dx) < 3:
                return
            d["active"] = True
            if d["kind"] != "move":
                self.edit_started.emit()
        dt = dx / self.pps
        group = self._has_angles(c.id)     # multicam: the viewer keeps showing every angle, in sync
        if d["kind"] == "trim_l":
            c.src_in = self._snap(min(max(0.0, d["in0"] + dt * d.get("v_in", 1.0)), d["out0"] - MIN_DUR))
            if not group:
                self.slip_preview.emit(c.id, c.src_in)
        elif d["kind"] == "trim_r":
            c.src_out = self._snap(max(min(f.duration, d["out0"] + dt * d.get("v_out", 1.0)), d["in0"] + MIN_DUR))
            if not group:
                self.slip_preview.emit(c.id, max(c.src_in, c.src_out - 1.0 / max(1.0, f.fps)))
        elif d["kind"] == "slip":
            dur = d["out0"] - d["in0"]
            new_in = self._snap(min(max(0.0, d["in0"] - dt * d.get("v", 1.0)), max(0.0, f.duration - dur)))
            c.src_in, c.src_out = new_in, new_in + dur
            if not group:
                self.slip_preview.emit(c.id, c.src_in)
            self.setCursor(Qt.ClosedHandCursor)
        elif d["kind"] == "move" and not self.magnetic:
            starts = self.project.clip_starts()
            start0 = starts[idx]
            free = max(0.0, start0 + dt)
            # snaps to the cuts (and the end) nearby, either edge of the moved clip
            cuts = [s for k, s in enumerate(starts + [self.project.duration]) if k not in (idx, idx + 1)]
            tol = 8.0 / self.pps
            for edge in (0.0, d["dur0"]):
                near = min(cuts, key=lambda s: abs(s - (free + edge)), default=None)
                if near is not None and abs(near - (free + edge)) < tol:
                    free = max(0.0, near - edge)
                    break
            d["free"] = free
            self.update()
            return
        elif d["kind"] == "move":
            t = self.x_to_t(pos.x())
            starts = self.project.clip_starts()
            drop = len(starts)
            for k, (s, cl) in enumerate(zip(starts, self.project.clips)):
                if t < s + cl.duration / 2:
                    drop = k
                    break
            d["drop"] = drop
            self.update()
            return
        self._conflicts = self.project.conflicts(self.min_gap)
        self.update()
        self.edited.emit()

    def mouseReleaseEvent(self, e):
        d = self._drag
        self._drag = None
        self._edge_timer.stop()
        if d is None:
            return
        if d["kind"] == "cap_shift":
            self.setCursor(Qt.ArrowCursor)
            if d["active"] and abs(d.get("dt", 0.0)) >= 0.01:
                self.captions_shifted.emit(d["cap"], d["dt"])
            self.update()
            return
        if d["kind"] == "resize":
            self.heights_changed.emit(dict(self.heights))
            self.refresh()
            return
        if "overlay" in d:
            if d["active"]:
                self.overlays_changed.emit()
                self.edit_finished.emit()
            else:
                self.seek.emit(self.x_to_t(e.position().x()))
            self.setCursor(Qt.ArrowCursor)
            self.refresh()
            return
        if "sound" in d:
            if d["active"]:
                self.sounds_changed.emit()
                self.edit_finished.emit()
            self.setCursor(Qt.ArrowCursor)
            self.refresh()
            return
        if "text" in d:
            if d["active"]:
                self.texts_changed.emit()
                self.edit_finished.emit()
            else:
                self.seek.emit(self.x_to_t(e.position().x()))
            self.setCursor(Qt.ArrowCursor)
            self.refresh()
            return
        if d.get("angle") and not d.get("active"):   # a click on a camera angle: it stays selected
            self.seek.emit(self.x_to_t(e.position().x()))
            self.setCursor(Qt.ArrowCursor)
            self.refresh()
            return
        if d["kind"] == "move" and d.get("active") and d.get("free") is not None:
            i = self.project.index_of(d["clip"])
            if i >= 0 and abs(d["free"] - self.project.clip_starts()[i]) > 1e-3:
                self.edit_started.emit()
                gaps.place(self.project, d["clip"], d["free"])
                self.edited.emit()
                self.edit_finished.emit()
        elif d["kind"] == "move" and d.get("active") and d.get("drop") is not None:
            i = self.project.index_of(d["clip"])
            drop = d["drop"]
            if i >= 0 and drop not in (i, i + 1):
                self.edit_started.emit()
                clip = self.project.clips.pop(i)
                self.project.clips.insert(drop - 1 if drop > i else drop, clip)
                self.edited.emit()
                self.edit_finished.emit()
        elif d["kind"] in ("trim_l", "trim_r", "slip") and d.get("active"):
            self.slip_end.emit()
            self.edit_finished.emit()
        elif not d.get("active") and d.get("mods", Qt.NoModifier) & Qt.ShiftModifier:
            self._click_select(d["clip"], d["mods"])
        elif not d.get("active") and d["kind"] != "scrub":
            # Simple click on a clip: select only it, and move the playhead there.
            if len(self._selection) > 1:
                self.selected = d["clip"]
                self.selection_changed.emit(d["clip"])
            if d["kind"] == "move":
                self.seek.emit(self.x_to_t(e.position().x()))
        self.setCursor(Qt.ArrowCursor)
        self.refresh()

    def mouseDoubleClickEvent(self, e):
        edge = self._edge_at(e.position().y())
        if edge:   # double-click a lane edge: back to its usual height
            self.set_heights({edge: DEFAULT_HEIGHTS[edge]}, emit=True)
            return
        oc, _ = self._overlay_hit(e.position())
        if oc is not None:
            self.context_action.emit("ov_edit", oc.id)
            return
        tx, _ = self._text_hit(e.position())
        if tx is not None:
            self.context_action.emit("text_edit", tx.id)
            return
        c, _ = self._hit(e.position())
        if c is not None:
            i = self.project.index_of(c.id)
            self.seek.emit(self.project.clip_starts()[i])

    def wheelEvent(self, e):
        delta = e.angleDelta().y() or e.angleDelta().x()
        if e.modifiers() & Qt.ShiftModifier:
            e.ignore()       # the panel scrolls the lanes up / down
            return
        if e.modifiers() & Qt.ControlModifier:
            self.set_zoom(self.pps * (1.2 if delta > 0 else 1 / 1.2), e.position().x())
        else:
            self.scroll_x = max(0.0, min(self.scroll_x - delta, max(0.0, self.content_width() - self.width())))
            self.refresh()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.view_changed.emit()

    def _context_menu(self, gpos, clip_id: str):
        m = QMenu(self)
        n = len(self._selection) if clip_id in self._selection else 1
        many = f" {n} clips" if n > 1 else ""
        items = [("split", "Split at playhead\tS")] if n == 1 else []
        items += [("transition", None), ("split", None),
                  ("duplicate", f"Duplicate{many}"), ("auto_color", f"Auto color{many}"),
                  ("reset_color", f"Reset color{many}"),
                  ("copy_grade", "Copy this clip's grade to the selected clips" if n > 1 else "Copy grade to all clips"),
                  ("alt_moment", f"Other moment{many}	R"),
                  ("link_toggle", None), ("speed", None),
                  ("face_zoom", f"Zoom{many} to face"),
                  ("facecam", f"Facecam layout{many} (game + webcam corner)"),
                  ("rotate:-90", f"Rotate{many} 90° left"), ("rotate:90", f"Rotate{many} 90° right"),
                  (None, None), ("show_media", "Show in Media panel"),
                  ("transcribe_force", f"Transcribe its footage again from scratch{many}"),
                  ("select_all", "Select all clips\tCtrl+A"), ("delete", f"Delete{many}\tDel")]
        for key, label in items:
            if key is None:
                m.addSeparator()
                continue
            if key == "link_toggle":
                angles = [o for o in self.project.overlays if o.link == clip_id and o.angle >= 2]
                if angles:
                    on = all(o.linked for o in angles)
                    self._menu_action(m, ("Unlink its camera angles\tCtrl+L" if on else
                                          "Link its camera angles again (back in sync)\tCtrl+L"), "link_toggle",
                                      clip_id)
                continue
            if key == "speed":
                c = next((x for x in self.project.clips if x.id == clip_id), None)
                if c is not None:
                    self._speed_menu(m, "", clip_id, c.speed, c.ramp, many, c.audio_mode)
                continue
            if key == "transition":
                sub = m.addMenu(f"Transition into {'these clips' if n > 1 else 'this clip'}")
                for tk, tl in [("", "None (cut)")] + [(k, v[0]) for k, v in TRANSITIONS.items()]:
                    a = QAction(tl, sub)
                    a.triggered.connect(lambda _=False, tk=tk: self.context_action.emit(f"transition:{tk}", clip_id))
                    sub.addAction(a)
                continue
            if key == "split" and label is None:          # (with a label: Split at playhead, below)
                sub = m.addMenu("Audio edit (J / L cut)")
                for sv, sl in ((0.0, "None (straight cut)"), (-0.5, "J-cut: sound starts 0.5 s early"),
                               (-1.0, "J-cut: sound starts 1 s early"), (0.5, "L-cut: sound runs on 0.5 s"),
                               (1.0, "L-cut: sound runs on 1 s")):
                    a = QAction(sl, sub)
                    a.triggered.connect(lambda _=False, sv=sv: self.context_action.emit(f"split:{sv}", clip_id))
                    sub.addAction(a)
                m.addSeparator()
                continue
            act = QAction(label, m)
            act.triggered.connect(lambda _=False, k=key: self.context_action.emit(k, clip_id))
            m.addAction(act)
            if key == "rotate:90":
                self._black_menu(m, self.playhead, "Insert black at the playhead")
        m.exec(gpos)


class TimelinePanel(QWidget):
    """Timeline view + mode buttons, zoom and lane-height controls and scrollbars."""

    fades_changed = Signal()   # the fade controls edited project.settings
    magnet_changed = Signal(bool)   # the magnetic timeline was turned on / off

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.view = TimelineView(project)
        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.valueChanged.connect(self._on_scroll)
        self.view.view_changed.connect(self._sync_scroll)

        self.btn_undo = QToolButton(text="↶ Undo")
        self.btn_undo.setToolTip("Undo the last change (Ctrl+Z)")
        self.btn_redo = QToolButton(text="↷ Redo")
        self.btn_redo.setToolTip("Redo the change you undid (Ctrl+Shift+Z or Ctrl+Y)")
        for b in (self.btn_undo, self.btn_redo):
            b.setEnabled(False)
        self.btn_move = QToolButton(text="↔ Move", checkable=True, checked=True)
        self.btn_slip = QToolButton(text="⇋ Slip", checkable=True)
        self.btn_move.setToolTip("Drag clips to reorder them. Drag clip edges to trim.")
        self.btn_slip.setToolTip("Drag inside a clip to slide which moment it shows (keeps its length).\n"
                                 "Tip: hold Alt or Shift in Move mode to slip temporarily.")
        grp = QButtonGroup(self)
        grp.addButton(self.btn_move)
        grp.addButton(self.btn_slip)
        self.btn_move.toggled.connect(lambda on: setattr(self.view, "mode", "move" if on else "slip"))
        self.btn_magnet = QToolButton(text="🧲 Magnetic", checkable=True, checked=True)
        self.btn_magnet.setToolTip("Magnetic timeline: clips stay together, a moved clip goes in at a cut and "
                                   "deleting closes the gap.\nOff: drag a clip anywhere to make space; where it was "
                                   "stays black, and deleting leaves a black gap (M).")
        self.btn_magnet.toggled.connect(self._magnet_toggled)
        self.btn_split = QToolButton(text="✂ Split")
        self.btn_delete = QToolButton(text="🗑 Delete")
        self.btn_fade = QToolButton(text="◐ Fade")
        self.btn_fade.setToolTip("Fade the picture and / or sound in from black at the start of the timeline "
                                 "and out at its end")
        self.btn_fade.setPopupMode(QToolButton.InstantPopup)
        self.btn_fade.setMenu(self._fade_menu())
        self.btn_hook = QToolButton(text="🎣 Hook")
        self.btn_hook.setToolTip("Choose the opening hook (H): the strongest moments of the whole video as "
                                 "thumbnails, the best ticked. Tick others, ask for more, and set how many clips "
                                 "it has. Undo brings the previous hook back.")
        self.btn_alt = QToolButton(text="🔀 Other moment")
        self.btn_alt.setToolTip("Swap the selected clip for the best moment close by in its footage (R). Its length "
                                "follows the new moment. Click again for the next one; Undo brings the previous back.")
        self.btn_link = QToolButton(text="🔗 Link", checkable=True)
        self.btn_link.setToolTip("Multicam: the camera angles of the selected clip are linked to it — trimming, "
                                 "slipping or Other moment on either keeps every angle in sync. Click to unlink "
                                 "them (edit them on their own), click again to link and re-sync (Ctrl+L).")
        self.btn_link.setEnabled(False)
        self.btn_fit = QToolButton(text="⇔ Fit")
        self.btn_fit.setToolTip("Zoom so the whole edit fits the timeline's width (Shift+Z)")
        self.btn_zoom_out = QToolButton(text="−")
        self.btn_zoom_out.setToolTip("Zoom out (Ctrl+−, or Ctrl+wheel on the timeline)")
        self.btn_zoom_in = QToolButton(text="+")
        self.btn_zoom_in.setToolTip("Zoom in (Ctrl+=, or Ctrl+wheel on the timeline)")
        self.zoom = QSlider(Qt.Horizontal)
        self.zoom.setRange(1, int(MAX_PPS))
        self.zoom.setValue(40)
        self.zoom.setFixedWidth(110)
        self.zoom.setToolTip("Timeline zoom")
        self.zoom.valueChanged.connect(lambda v: self.view.set_zoom(float(v)) if abs(v - self.view.pps) > 0.5 else None)
        self.btn_fit.clicked.connect(self.view.fit)
        self.btn_zoom_out.clicked.connect(lambda: self.view.zoom_by(1 / 1.5))
        self.btn_zoom_in.clicked.connect(lambda: self.view.zoom_by(1.5))
        self.btn_heights = QToolButton(text="↕ Lanes")
        self.btn_heights.setToolTip("Height of the video, layer and sound lanes (or drag a lane's bottom edge)")
        self.btn_heights.setPopupMode(QToolButton.InstantPopup)
        self.btn_heights.setMenu(self._heights_menu())
        self.lbl = QLabel("")
        self.lbl.setStyleSheet("color: #8b909a;")

        # Wraps onto a second row when the timeline is narrow, instead of hiding buttons or widening the window.
        top = FlowBar(hspacing=6, vspacing=2)
        top.flow.setContentsMargins(4, 2, 4, 2)

        def sep():
            lbl = QLabel("|")
            lbl.setStyleSheet("color: #45484f;")
            return lbl

        top.add(self.btn_undo, self.btn_redo, sep(), self.btn_move, self.btn_slip, self.btn_magnet, self.btn_split,
                self.btn_delete,
                self.btn_fade, sep(), self.btn_hook, self.btn_alt, self.btn_link)
        self.lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Always room for the clip count and length; the overlap warning after it may be clipped (tooltip).
        self.lbl.setMinimumWidth(self.lbl.fontMetrics().horizontalAdvance("000 clips · 00:00:00") + 8)
        top.add(self.lbl)
        zoom = QWidget()
        row = QHBoxLayout(zoom)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel("Zoom"))
        row.addWidget(self.btn_zoom_out)
        row.addWidget(self.zoom)
        row.addWidget(self.btn_zoom_in)
        row.addWidget(self.btn_fit)
        row.addWidget(self.btn_heights)
        top.add(zoom)
        # The lanes scroll up / down when they are taller than the panel (taller lanes, more video layers).
        self.area = QScrollArea()
        self.area.setWidget(self.view)
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.NoFrame)
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.area.setMinimumHeight(120)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(top)
        # Slip mode: what is said in the selected clip's footage, to slip it to a sentence (main window fills it).
        self.slip_text = SlipTranscript()
        self.slip_text.hide()
        lay.addWidget(self.slip_text)
        lay.addWidget(self.area, 1)
        lay.addWidget(self.scroll)
        self.view.heights_changed.connect(lambda _h: self._sync_heights())

    def _magnet_toggled(self, on: bool):
        self.view.magnetic = on
        self.btn_magnet.setText("🧲 Magnetic" if on else "🧲 Free")
        self.magnet_changed.emit(on)

    def set_magnetic(self, on: bool):
        self.btn_magnet.setChecked(on)
        self._magnet_toggled(on)

    def _heights_menu(self) -> QMenu:
        m = QMenu(self)
        box = QWidget()
        f = QFormLayout(box)
        f.setContentsMargins(10, 8, 10, 8)
        self.height_sliders: dict[str, QSlider] = {}
        for key, label in (("video", "Video track"), ("layer", "Each video layer"), ("sound", "Sound rows")):
            lo, hi = HEIGHT_RANGES[key]
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setMinimumWidth(160)
            sl.setValue(self.view.heights[key])
            sl.valueChanged.connect(lambda v, k=key: self.view.set_heights({k: v}, emit=True))
            self.height_sliders[key] = sl
            f.addRow(label, sl)
        row = QHBoxLayout()
        b_fit = QPushButton("Fit to panel")
        b_fit.setToolTip("Make the lanes fill the timeline panel's height")
        b_fit.clicked.connect(lambda: self.view.fit_heights(self.area.viewport().height()))
        b_reset = QPushButton("Reset")
        b_reset.clicked.connect(lambda: self.view.set_heights(DEFAULT_HEIGHTS, emit=True))
        row.addWidget(b_fit)
        row.addWidget(b_reset)
        f.addRow(row)
        act = QWidgetAction(m)
        act.setDefaultWidget(box)
        m.addAction(act)
        return m

    def _sync_heights(self):
        for key, sl in self.height_sliders.items():
            sl.blockSignals(True)
            sl.setValue(self.view.heights[key])
            sl.blockSignals(False)

    FADE_LENGTHS = (0.5, 1.0, 2.0, 3.0)

    def _fade_menu(self) -> QMenu:
        m = QMenu(self)
        box = QWidget()
        f = QFormLayout(box)
        f.setContentsMargins(10, 8, 10, 8)
        self.btn_fade_in, self.btn_fade_out = QPushButton("Fade in at start"), QPushButton("Fade out at end")
        toggles = QHBoxLayout()
        for b in (self.btn_fade_in, self.btn_fade_out):
            b.setCheckable(True)
            b.setObjectName("toggle")
            b.toggled.connect(self._fades_edited)
            toggles.addWidget(b)
        self._fade_len = 1.0
        self.fade_len_group = QButtonGroup(box)
        self.fade_len_group.setExclusive(True)
        lengths = QHBoxLayout()
        for k, v in enumerate(self.FADE_LENGTHS):
            b = QPushButton(f"{v:g} s", checkable=True)
            b.setObjectName("toggle")
            self.fade_len_group.addButton(b, k)
            lengths.addWidget(b)
        self.fade_len_group.idClicked.connect(self._fade_length_picked)
        self.chk_fade_picture = QCheckBox("Picture (from / to black)")
        self.chk_fade_sound = QCheckBox("Sound (from / to silence)")
        for chk in (self.chk_fade_picture, self.chk_fade_sound):
            chk.toggled.connect(self._fades_edited)
        f.addRow(toggles)
        f.addRow("Length", lengths)
        f.addRow(self.chk_fade_picture)
        f.addRow(self.chk_fade_sound)
        act = QWidgetAction(m)
        act.setDefaultWidget(box)
        m.addAction(act)
        return m

    def sync_fades(self):
        """Show the project's fade settings."""
        s = self.view.project.settings
        ws = (self.btn_fade_in, self.btn_fade_out, self.chk_fade_picture, self.chk_fade_sound)
        for w in ws:
            w.blockSignals(True)
        self._fade_len = s.fade_in if s.fade_in > 0 else (s.fade_out if s.fade_out > 0 else self._fade_len)
        self.btn_fade_in.setChecked(s.fade_in > 0)
        self.btn_fade_out.setChecked(s.fade_out > 0)
        self.chk_fade_picture.setChecked(s.fade_picture)
        self.chk_fade_sound.setChecked(s.fade_sound)
        for w in ws:
            w.blockSignals(False)
        self._show_fade_length()
        self._fade_label()

    def _show_fade_length(self):
        k = min(range(len(self.FADE_LENGTHS)), key=lambda i: abs(self.FADE_LENGTHS[i] - self._fade_len))
        self.fade_len_group.button(k).setChecked(True)

    def _fade_length_picked(self, k: int):
        self._fade_len = self.FADE_LENGTHS[k]
        self._fades_edited()

    def _fades_edited(self):
        s = self.view.project.settings
        s.fade_in = self._fade_len if self.btn_fade_in.isChecked() else 0.0
        s.fade_out = self._fade_len if self.btn_fade_out.isChecked() else 0.0
        s.fade_picture, s.fade_sound = self.chk_fade_picture.isChecked(), self.chk_fade_sound.isChecked()
        self._fade_label()
        self.view.update()
        self.fades_changed.emit()

    def _fade_label(self):
        s = self.view.project.settings
        on = (s.fade_in > 0 or s.fade_out > 0) and (s.fade_picture or s.fade_sound)
        ends = "/".join(n for n, v in (("in", s.fade_in), ("out", s.fade_out)) if v > 0)
        self.btn_fade.setText("◐ Fade" + (f" {ends} {max(s.fade_in, s.fade_out):g} s" if on else ""))

    def _on_scroll(self, v):
        if abs(v - self.view.scroll_x) >= 1:
            self.view.scroll_x = float(v)
            self.view.update()

    def _sync_scroll(self):
        v = self.view
        maxv = int(max(0.0, v.content_width() - v.width()))
        # Only shown when there is something to scroll to (not after Fit).
        self.scroll.setVisible(maxv > 0 or v.scroll_x > 0)
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, maxv)
        self.scroll.setPageStep(max(1, v.width()))
        self.scroll.setValue(int(v.scroll_x))
        self.scroll.blockSignals(False)
        self.zoom.blockSignals(True)
        self.zoom.setValue(max(self.zoom.minimum(), int(round(v.pps))))
        self.zoom.blockSignals(False)
        n = len(v.project.clips)
        bad = len(v._conflicts)
        txt = f"{n} clips · {fmt_time(v.project.duration, frames=False)}"
        if bad:
            txt += f"  ·  ⚠ {bad} clips overlap / too close to another clip from the same footage"
        self.lbl.setText(txt)
        self.lbl.setToolTip(txt)   # clipped when the timeline is narrow
