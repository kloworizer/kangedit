"""Timeline viewer: decodes, grades, reframes and overlays captions in real time."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QRectF, QSize, Qt, QTimer, QUrl, Signal, QPointF
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QFontMetricsF, QTransform
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget, QStyle

from .. import motion, reframe, spherical
from ..captions import POP_MS, POP_SCALE, WORD_SCALE, Caption, caption_layout, wrap_words
from ..color import Grade
from ..config import frame_size
from ..decoder import DecoderPool
from ..edits import TRANSITIONS, edits
from ..fonts import ensure_qt_font
from ..models import OverlayClip, Project, TextClip, TimelineClip
from ..titles import text_state
from ..preview import Look, OverlayShot, PreviewEngine, Rendered, Shot
from .platform_overlay import draw_platform_overlay
from .source_view import SourcePanel
from .text_render import draw_text, text_rect
from .theme import fmt_time

ADJUST_MODES = [("on", "Color: on"), ("playback", "Color: off while playing"), ("off", "Color: off")]


class Canvas(QWidget):
    text_pressed = Signal(str)                 # a title was clicked (before a possible drag)
    text_dragged = Signal(str, float, float)   # title id, new anchor x / y (0..1 of the frame)
    text_released = Signal(str, bool)          # title id, whether it was moved
    text_activated = Signal(str)               # double-click: edit the title
    context_menu = Signal(object, str)         # global position, title id under the mouse ("" = none)
    background_activated = Signal()            # double-click outside the titles
    overlay_pressed = Signal(str)              # a video-layer clip was clicked
    overlay_dragged = Signal(str, float, float)   # overlay id, new centre x / y (0..1 of the frame)
    overlay_released = Signal(str, bool)       # overlay id, whether it was moved
    overlay_scaled = Signal(str, float)        # overlay id, size factor (mouse wheel over it)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: Optional[QImage] = None
        self.texts: list[TextClip] = []        # the project's titles (shared list)
        self.text_t = 0.0                      # timeline time the titles are drawn at
        self.selected_text = ""
        self.overlay_gain = 1.0                # timeline fade in / out applied to titles and captions
        self._frame_rect = QRectF()
        self._text_rects: list[tuple[str, QRectF]] = []
        self._text_drag: Optional[dict] = None
        # Video-layer clips showing now: (id, centre x, y, width, height as parts of the frame, rotation °).
        self.overlay_boxes: list[tuple[str, float, float, float, float, float]] = []
        self.selected_overlay = ""
        self._ov_drag: Optional[dict] = None
        self.setMouseTracking(True)
        self.caption: Optional[Caption] = None
        self.caption_t = 0.0
        self.caption_style = None
        self._font_ready = ""
        self.layout = None
        self.out_h = 1080
        self.message = "Import footage and press Auto Edit"
        self.show_guides = False
        self.platform_overlay = ""             # PLATFORM_OVERLAYS key: app interface drawn over the picture
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 180)

    def sizeHint(self):
        return QSize(800, 450)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d0e10"))
        if self.image is None:
            p.setPen(QColor("#6b6e76"))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        r = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(r, self.image)
        self._frame_rect = r
        if self.show_guides:
            p.setPen(QPen(QColor(255, 255, 255, 70), 1, Qt.DashLine))
            for k in (1, 2):
                p.drawLine(QPointF(r.left() + r.width() * k / 3, r.top()), QPointF(r.left() + r.width() * k / 3, r.bottom()))
                p.drawLine(QPointF(r.left(), r.top() + r.height() * k / 3), QPointF(r.right(), r.top() + r.height() * k / 3))
        self._draw_overlay_outline(p, r)
        self._draw_texts(p, r)
        if self.caption and self.caption_style is not None:
            p.setOpacity(self.overlay_gain)
            self._draw_caption(p, r)
            p.setOpacity(1.0)
        if self.platform_overlay:
            draw_platform_overlay(p, r, self.platform_overlay)

    def _draw_texts(self, p: QPainter, r: QRectF):
        self._text_rects = []
        for tc in self.texts:
            st = text_state(tc, self.text_t)
            if st is None:
                continue
            draw_text(p, r, tc, st, self.overlay_gain)
            rect = text_rect(tc, r)
            self._text_rects.append((tc.id, rect))
            if tc.id == self.selected_text:
                p.setPen(QPen(QColor("#4c8dff"), 1.5, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(rect)

    def _draw_overlay_outline(self, p: QPainter, r: QRectF):
        for oid, cx, cy, w, h, rot in self.overlay_boxes:
            if oid != self.selected_overlay:
                continue
            p.save()
            p.translate(r.left() + cx * r.width(), r.top() + cy * r.height())
            p.rotate(rot)
            p.setPen(QPen(QColor("#ffb84d"), 1.5, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(-w * r.width() / 2, -h * r.height() / 2, w * r.width(), h * r.height()))
            p.restore()

    def _overlay_at(self, pos: QPointF) -> str:
        """The topmost video-layer clip under pos ("" for none)."""
        r = self._frame_rect
        if r.width() <= 0:
            return ""
        for oid, cx, cy, w, h, rot in reversed(self.overlay_boxes):
            pt = QTransform().rotate(-rot).map(QPointF(pos.x() - (r.left() + cx * r.width()),
                                                        pos.y() - (r.top() + cy * r.height())))
            if abs(pt.x()) <= w * r.width() / 2 and abs(pt.y()) <= h * r.height() / 2:
                return oid
        return ""

    def wheelEvent(self, e):
        oid = self._overlay_at(e.position())
        if oid and oid == self.selected_overlay:
            self.overlay_scaled.emit(oid, 1.05 if e.angleDelta().y() > 0 else 1 / 1.05)
            return
        super().wheelEvent(e)

    def _text_at(self, pos: QPointF) -> str:
        for tid, rect in reversed(self._text_rects):
            if rect.contains(pos):
                return tid
        return ""

    def mousePressEvent(self, e):
        pos = e.position()
        tid = self._text_at(pos)
        if e.button() == Qt.RightButton:
            if tid:
                self.text_pressed.emit(tid)
            self.context_menu.emit(e.globalPosition().toPoint(), tid)
            return
        if e.button() == Qt.LeftButton and tid:
            tc = next((x for x in self.texts if x.id == tid), None)
            if tc is not None:
                self.text_pressed.emit(tid)
                self._text_drag = {"id": tid, "p0": pos, "x0": tc.x, "y0": tc.y, "moved": False}
                return
        oid = self._overlay_at(pos) if e.button() == Qt.LeftButton else ""
        if oid:
            box = next(b for b in self.overlay_boxes if b[0] == oid)
            self.overlay_pressed.emit(oid)
            self._ov_drag = {"id": oid, "p0": pos, "x0": box[1], "y0": box[2], "moved": False}
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        d = self._text_drag
        pos = e.position()
        od = self._ov_drag
        if od is not None:
            r = self._frame_rect
            dx, dy = pos.x() - od["p0"].x(), pos.y() - od["p0"].y()
            if r.width() <= 0 or (not od["moved"] and abs(dx) + abs(dy) < 3):
                return
            od["moved"] = True
            x, y = od["x0"] + dx / r.width(), od["y0"] + dy / r.height()
            if abs(x - 0.5) * r.width() < 6:      # snap to the centre lines
                x = 0.5
            if abs(y - 0.5) * r.height() < 6:
                y = 0.5
            self.overlay_dragged.emit(od["id"], x, y)
            return
        if d is None:
            self.setCursor(Qt.SizeAllCursor if self._text_at(pos) or self._overlay_at(pos) else Qt.ArrowCursor)
            return
        r = self._frame_rect
        if r.width() <= 0:
            return
        dx, dy = pos.x() - d["p0"].x(), pos.y() - d["p0"].y()
        if not d["moved"] and abs(dx) + abs(dy) < 3:
            return
        d["moved"] = True
        x = min(1.0, max(0.0, d["x0"] + dx / r.width()))
        y = min(1.0, max(0.0, d["y0"] + dy / r.height()))
        if abs(x - 0.5) * r.width() < 6:      # snap to the centre lines
            x = 0.5
        if abs(y - 0.5) * r.height() < 6:
            y = 0.5
        self.text_dragged.emit(d["id"], x, y)

    def mouseReleaseEvent(self, e):
        od, self._ov_drag = self._ov_drag, None
        if od is not None:
            self.overlay_released.emit(od["id"], od["moved"])
            return
        d, self._text_drag = self._text_drag, None
        if d is not None:
            self.text_released.emit(d["id"], d["moved"])
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        tid = self._text_at(e.position())
        if tid:
            self.text_activated.emit(tid)
        elif self._overlay_at(e.position()):
            pass
        else:
            self.background_activated.emit()

    def _draw_caption(self, p: QPainter, r: QRectF):
        """Same layout and word animations as the burned-in export (captions.write_ass)."""
        st, lay, cap, t = self.caption_style, self.layout, self.caption, self.caption_t
        if lay is None:
            return
        anim = st.animation
        words = [w for _, _, w in cap.words] or cap.text.split()
        active = cap.active_word(t) if cap.words else -1
        if anim in ("pop", "reveal", "word") and cap.words and active < 0:
            return   # nothing spoken yet
        px = max(8.0, r.height() * lay.size_pct / 100.0)
        if anim == "word" and cap.words:
            index_of = [active]
            words = [words[active]]
            px *= WORD_SCALE
            lines = [[0]]
        else:
            index_of = list(range(len(words)))
            lines = wrap_words(words, lay.max_chars, lay.max_lines)
        if not lines:
            return
        if st.font != self._font_ready:
            ensure_qt_font(st.font)
            self._font_ready = st.font
        font = QFont(st.font)
        font.setPixelSize(int(px))
        font.setBold(True)
        fm = QFontMetricsF(font)
        space = fm.horizontalAdvance(" ")
        line_h = fm.height()
        block_h = line_h * len(lines)
        y = r.top() + r.height() * lay.y
        y0 = {"bottom": y - block_h, "center": y - block_h / 2}.get(lay.anchor, y)
        p.setRenderHint(QPainter.Antialiasing)
        boxed = st.box and anim != "box"
        for li, line in enumerate(lines):
            widths = [fm.horizontalAdvance(words[i]) for i in line]
            tw = sum(widths) + space * (len(line) - 1)
            x = r.center().x() - tw / 2
            base = y0 + li * line_h + fm.ascent()
            if boxed:
                pad = px * 0.2
                p.fillRect(QRectF(x - pad, y0 + li * line_h - pad / 2, tw + 2 * pad, line_h + pad / 2), QColor(0, 0, 0, 140))
            for i, wdt in zip(line, widths):
                k = index_of[i]
                if not (anim in ("pop", "reveal") and k > active):
                    path = QPainterPath()
                    path.addText(QPointF(x, base), font, words[i])
                    if anim in ("pop", "word") and k == active and cap.words:
                        done = min(1.0, max(0.0, (t - cap.words[k][0]) * 1000.0 / POP_MS))
                        sc = POP_SCALE + (1.0 - POP_SCALE) * done
                        if sc != 1.0:
                            c = path.boundingRect().center()
                            path = QTransform().translate(c.x(), c.y()).scale(sc, sc).translate(-c.x(), -c.y()).map(path)
                    if anim == "box" and k == active:
                        p.strokePath(path, QPen(QColor(st.highlight), px * 0.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                        p.fillPath(path, QColor(st.highlight))
                    elif st.outline and not boxed:
                        p.strokePath(path, QPen(QColor(0, 0, 0, 230), max(2.0, px / 7), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    color = st.highlight if anim == "highlight" and k == active else st.color
                    p.fillPath(path, QColor(color))
                x += wdt + space


class Viewer(QWidget):
    time_changed = Signal(float)               # the timeline's playhead (not sent in Source mode)
    play_state = Signal(bool)
    adjust_mode_changed = Signal(str)
    mode_changed = Signal(str)                 # "timeline" | "source"
    use_source = Signal(str, float, float, str)   # footage id, in, out, "insert" | "append"
    transcribe_source = Signal(str)            # footage id

    PREVIEW_SHORT = 540

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.decoders = DecoderPool()      # UI-side sampling (auto color, log detection)
        self._bridge = _Bridge(self)
        self._bridge.still.connect(self._on_still)
        self.engine = PreviewEngine(self._bridge.still.emit)
        self._play_shots: list[Shot] = []
        self._play_overlays: tuple = ()
        self._frame: Optional[np.ndarray] = None
        self.captions: list[Caption] = []
        self.t = 0.0
        self.playing = False
        self._play_t0 = 0.0
        self._play_clock = 0.0
        self._override: Optional[tuple[str, float]] = None   # (clip_id, source time) during slip
        self._audio_path: Optional[str] = None
        self.audio_valid = False
        # Log conversion, LUTs and grading in the viewer (never affects export). "playback": off only while
        # playing, so playback is lighter and a paused frame still shows the finished color.
        self.adjust_mode = "on"            # "on" | "playback" | "off"
        # Source mode: a file from the Media panel plays on its own (a one-clip project of it), with its sound;
        # the timeline's project, playhead and sound are kept to come back to.
        self.mode = "timeline"
        self.source_fid = ""
        self.marks: dict[str, list] = {}   # footage id -> [in, out] (None: not set)
        self._timeline: Optional[tuple] = None   # (project, playhead, sound file, sound valid) while in Source

        self.canvas = Canvas()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)

        self.audio_out = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self._audio_resume = False     # new mix loaded during playback: start it at the clock time
        self._audio_check = 0.0        # next time _sync_audio looks at the audio player

        st = self.style()
        self.btn_start = QPushButton(st.standardIcon(QStyle.SP_MediaSkipBackward), "")
        self.btn_prev = QPushButton(st.standardIcon(QStyle.SP_MediaSeekBackward), "")
        self.btn_play = QPushButton(st.standardIcon(QStyle.SP_MediaPlay), "")
        self.btn_next = QPushButton(st.standardIcon(QStyle.SP_MediaSeekForward), "")
        self.btn_end = QPushButton(st.standardIcon(QStyle.SP_MediaSkipForward), "")
        for b, tip in ((self.btn_start, "Go to start (Home)"), (self.btn_prev, "Previous clip (Up)"),
                       (self.btn_play, "Play / Pause (Space)"), (self.btn_next, "Next clip (Down)"),
                       (self.btn_end, "Go to end (End)")):
            b.setToolTip(tip)
            b.setFixedWidth(34)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_start.clicked.connect(lambda: self.seek(0))
        self.btn_end.clicked.connect(lambda: self.seek(max(0.0, self.project.duration - 1e-3)))
        self.btn_prev.clicked.connect(self.prev_clip)
        self.btn_next.clicked.connect(self.next_clip)
        self.lbl_time = QLabel("00:00:00 / 00:00:00")
        self.lbl_time.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #8b909a;")
        # Clipped rather than widening the window: a long clip name would otherwise set its minimum width.
        self.lbl_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.cb_adjust = QComboBox()
        for key, label in ADJUST_MODES:
            self.cb_adjust.addItem(label, key)
        self.cb_adjust.setToolTip("Color profile, LUTs and grading in the viewer.\n"
                                  "Turn them off for lighter playback: export always includes them.")
        self.cb_adjust.currentIndexChanged.connect(lambda _: self.set_adjust_mode(self.cb_adjust.currentData()))
        self.btn_full = QPushButton("⛶")
        self.btn_full.setFixedWidth(36)
        self.btn_full.setToolTip("Full screen preview (F11, or double-click the picture). Esc to leave.")
        self.btn_full.clicked.connect(self.toggle_fullscreen)
        self.canvas.background_activated.connect(self.toggle_fullscreen)
        self._home: Optional[tuple[QWidget, int]] = None   # where the viewer sits while full screen

        # Shown while the preview sound is being mixed again after a change (music, sounds, volume, trims):
        # playback has the sound from before the change until then.
        self.lbl_audio = QLabel("♪ updating")
        self.lbl_audio.setStyleSheet("color: #1e1f22; background: #f0b429; border-radius: 4px; padding: 1px 6px;")
        self.lbl_audio.setToolTip("The preview sound is being mixed again after your last change. Until it is "
                                  "ready, playback has the sound from before (a music track or sound just added "
                                  "is heard once it is). The export always has the latest.")
        self.lbl_audio.hide()
        self.lbl_time.setMinimumWidth(self.lbl_time.sizeHint().width())   # the badge never squeezes the time
        self.btn_timeline = QPushButton("Timeline")
        self.btn_source = QPushButton("Source")
        for b, tip in ((self.btn_timeline, "Play the edit (the timeline)"),
                       (self.btn_source, "Play a file from the Media panel on its own (double-click it there), "
                                         "mark the part to use (I / O, or its words) and put it on the timeline")):
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.setStyleSheet("padding: 3px 6px;")
        self.btn_timeline.setChecked(True)
        self.btn_source.setEnabled(False)
        self.btn_timeline.clicked.connect(lambda: self.show_timeline())
        self.btn_source.clicked.connect(lambda: self.open_source(self.source_fid))
        self.source_panel = SourcePanel()
        self.source_panel.hide()
        self.source_panel.bar.seek.connect(self.seek)
        self.source_panel.words.seek.connect(self.seek)
        self.source_panel.words.marked.connect(self.set_marks)
        self.source_panel.mark.connect(self.mark)
        self.source_panel.use.connect(lambda how: self.use_source.emit(self.source_fid, *self.marked_range(), how))
        self.source_panel.transcribe.connect(lambda: self.transcribe_source.emit(self.source_fid))
        bar = QHBoxLayout()
        bar.addWidget(self.btn_timeline)
        bar.addWidget(self.btn_source)
        bar.addWidget(self.lbl_time)
        bar.addWidget(self.lbl_audio)
        bar.addStretch(1)
        for b in (self.btn_start, self.btn_prev, self.btn_play, self.btn_next, self.btn_end):
            bar.addWidget(b)
        bar.addStretch(1)
        bar.addWidget(self.lbl_info)
        bar.addWidget(self.cb_adjust)
        bar.addWidget(self.btn_full)
        # Keys go to the viewer (full screen playback keys), never to a focused button.
        for w in (self.btn_start, self.btn_prev, self.btn_play, self.btn_next, self.btn_end, self.cb_adjust,
                  self.btn_full):
            w.setFocusPolicy(Qt.NoFocus)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(bar)
        lay.addWidget(self.source_panel)

    # ------------------------------------------------------------------ full screen
    def is_fullscreen(self) -> bool:
        return self._home is not None

    def toggle_fullscreen(self):
        """Show the viewer alone on the whole screen (the timeline keeps playing in it), or put it back."""
        if self._home is None:
            parent = self.parentWidget()
            if parent is None:
                return
            idx = parent.indexOf(self) if hasattr(parent, "indexOf") else 0
            self._home = (parent, idx)
            screen = self.window().screen()
            self.setParent(None)
            self.setWindowFlags(Qt.Window)
            self.setWindowTitle("Preview")
            if screen is not None:
                self.setGeometry(screen.geometry())
            self.showFullScreen()
            self.setFocus()
        else:
            parent, idx = self._home
            self._home = None
            self.setWindowFlags(Qt.Widget)
            if hasattr(parent, "insertWidget"):
                parent.insertWidget(idx, self)
            else:
                self.setParent(parent)
            self.show()
            parent.window().activateWindow()
        self.btn_full.setText("✕" if self._home else "⛶")

    def keyPressEvent(self, e):
        """Playback keys while full screen (the main window's shortcuts do not reach this window)."""
        if self._home is None:
            return super().keyPressEvent(e)
        k, shift = e.key(), bool(e.modifiers() & Qt.ShiftModifier)
        if k in (Qt.Key_Escape, Qt.Key_F11, Qt.Key_F):
            self.toggle_fullscreen()
        elif k == Qt.Key_Space:
            self.toggle_play()
        elif k in (Qt.Key_Left, Qt.Key_Right):
            self.step((10 if shift else 1) * (-1 if k == Qt.Key_Left else 1))
        elif k == Qt.Key_Up:
            self.prev_clip()
        elif k == Qt.Key_Down:
            self.next_clip()
        elif k == Qt.Key_Home:
            self.seek(0)
        elif k == Qt.Key_End:
            self.seek(max(0.0, self.project.duration - 1e-3))
        elif k in (Qt.Key_I, Qt.Key_O) and self.mode == "source":
            self.mark("in" if k == Qt.Key_I else "out")
        else:
            super().keyPressEvent(e)

    # ------------------------------------------------------------------ public api
    def set_project(self, project: Project):
        self.stop()
        self._leave_source()
        self.source_fid = ""
        self.marks = {}
        self.btn_source.setEnabled(False)
        self.project = project
        self.drop_decoders()
        self.t = 0.0
        self.set_audio(None)
        self.refresh()

    def drop_decoders(self):
        """Release every open media file (before proxies or cache files are replaced or removed)."""
        self.decoders.drop()
        self.engine.drop_decoders()

    def shutdown(self):
        self.stop()
        self.decoders.close()
        self.engine.close()

    def set_adjust_mode(self, mode: str):
        """"on" | "playback" (off while playing) | "off": color adjustments shown in the viewer."""
        if mode not in dict(ADJUST_MODES):
            mode = "on"
        if self.cb_adjust.currentData() != mode:
            self.cb_adjust.blockSignals(True)
            self.cb_adjust.setCurrentIndex(self.cb_adjust.findData(mode))
            self.cb_adjust.blockSignals(False)
        if mode == self.adjust_mode:
            return
        self.adjust_mode = mode
        self.adjust_mode_changed.emit(mode)
        if self.playing:
            self._restart_playback()
        self.refresh()

    def set_captions(self, caps: list[Caption]):
        self.captions = caps
        self.refresh()

    def set_audio(self, path: Optional[str]):
        """Use a new preview mix. During playback it takes over at the current time (live sound edits). In
        Source mode it is kept for the timeline."""
        if self.mode == "source" and self._timeline is not None:
            proj, t, _, _ = self._timeline
            self._timeline = (proj, t, path, bool(path) and os.path.exists(path))
            return
        self._audio_path = path
        self.audio_valid = bool(path) and os.path.exists(path)
        self._audio_resume = self.playing and self._play_clock is not None and self.audio_valid
        self.player.setSource(QUrl.fromLocalFile(path) if self.audio_valid else QUrl())

    def _on_media_status(self, status):
        if not self._audio_resume or status not in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia):
            return
        self._audio_resume = False
        if self.playing and self._play_clock is not None:
            self.player.setPosition(int((self._play_t0 + time.perf_counter() - self._play_clock) * 1000))
            self.player.play()

    def invalidate_audio(self):
        if self.mode == "source" and self._timeline is not None:
            proj, t, path, _ = self._timeline
            self._timeline = (proj, t, path, False)
            return
        self.audio_valid = False

    def set_audio_pending(self, on: bool):
        """Show (or hide) that the preview sound is being mixed again (the timeline's: not in Source mode)."""
        self._audio_pending = on
        self.lbl_audio.setVisible(on and self.mode == "timeline")

    def seek(self, t: float):
        dur = self.project.duration
        self.t = max(0.0, min(t, max(0.0, dur - 1e-3)))
        if self.playing:
            self._restart_playback()
        self.refresh()
        self._emit_time(self.t)

    def _emit_time(self, t: float):
        """The playhead moved: the timeline's (time_changed), or in Source mode the file's (its panel)."""
        if self.mode == "source":
            a, b = self.marks.get(self.source_fid, [None, None])
            self.source_panel.show_state(t, a, b, self.project.settings.fps)
        else:
            self.time_changed.emit(t)

    # ------------------------------------------------------------------ source mode
    def open_source(self, fid: str, footage=None):
        """Play a file from the Media panel on its own (Source mode), from its mark in (or where it was left)."""
        f = footage if footage is not None else (self._timeline[0] if self._timeline else self.project).footage.get(fid)
        if f is None:
            self.btn_source.setChecked(self.mode == "source")      # (a checkable button: as it was)
            self.btn_timeline.setChecked(self.mode != "source")
            return
        self.stop()
        if self.mode == "timeline":
            self._timeline = (self.project, self.t, self._audio_path, self.audio_valid)
        tl = self._timeline[0]
        src = Project()
        src.settings = copy.copy(tl.settings)
        src.settings.fade_in = src.settings.fade_out = 0.0
        src.footage = {f.id: f}
        src.clips = [TimelineClip(f.id, 0.0, max(0.05, f.duration), id="source")]
        same = self.source_fid == f.id and self.mode == "source"
        self.mode, self.source_fid, self.project = "source", f.id, src
        self.marks.setdefault(f.id, [None, None])
        if not same:
            self.t = self.marks[f.id][0] or 0.0
        sound = (f.proxy_path if tl.settings.use_proxies and f.has_proxy() else f.path) \
            if f.has_audio and not f.is_image else None
        self._audio_path = sound
        self.audio_valid = bool(sound) and os.path.exists(sound)
        self.player.setSource(QUrl.fromLocalFile(sound) if self.audio_valid else QUrl())
        self.source_panel.show_file(f)
        self.source_panel.show()
        self.btn_source.setEnabled(True)
        self.btn_source.setText("Source")
        self.btn_source.setToolTip(f"Playing {f.name} on its own. Timeline: back to the edit")
        self.btn_source.setChecked(True)
        self.btn_timeline.setChecked(False)
        self.lbl_audio.hide()
        self.refresh()
        self._emit_time(self.t)
        self.mode_changed.emit("source")

    def show_timeline(self):
        """Back to the edit (Timeline mode), where its playhead was."""
        if self.mode != "source":
            self.btn_timeline.setChecked(True)
            return
        self.stop()
        self._leave_source()
        self.refresh()
        self.time_changed.emit(self.t)
        self.mode_changed.emit("timeline")

    def _leave_source(self):
        if self.mode != "source" or self._timeline is None:
            return
        proj, t, path, valid = self._timeline
        self._timeline = None
        # (the edit may have got shorter meanwhile: an Undo after Insert)
        self.mode, self.project, self.t = "timeline", proj, max(0.0, min(t, max(0.0, proj.duration - 1e-3)))
        self._audio_path, self.audio_valid = path, valid
        self.player.setSource(QUrl.fromLocalFile(path) if path and os.path.exists(path) else QUrl())
        self.source_panel.hide()
        self.btn_timeline.setChecked(True)
        self.btn_source.setChecked(False)
        self.lbl_audio.setVisible(getattr(self, "_audio_pending", False))

    def close_source(self, fids) -> None:
        """Files taken out of the project: Source mode stops showing them (back to the edit)."""
        if self.source_fid not in fids:
            return
        self.show_timeline()
        self.marks.pop(self.source_fid, None)
        self.source_fid = ""
        self.btn_source.setEnabled(False)
        self.btn_source.setChecked(False)

    def timeline_time(self) -> float:
        """The timeline's playhead (kept while in Source mode)."""
        return self._timeline[1] if self.mode == "source" and self._timeline is not None else self.t

    def set_timeline_time(self, t: float):
        """Move the timeline's playhead (in Source mode: where it goes back to)."""
        if self.mode == "source" and self._timeline is not None:
            proj, _, path, valid = self._timeline
            self._timeline = (proj, t, path, valid)
        else:
            self.seek(t)

    def update_source(self, footage):
        """The file shown in Source mode changed (a new transcript): show it again."""
        if self.mode == "source" and footage is not None and footage.id == self.source_fid:
            self.source_panel.show_file(footage)
            self._emit_time(self.t)

    def mark(self, kind: str):
        """Mark in / out at the playhead, or clear the marks (Source mode)."""
        if self.mode != "source":
            return
        m = self.marks.setdefault(self.source_fid, [None, None])
        if kind == "in":
            m[0] = round(self.t, 3)
            if m[1] is not None and m[1] <= m[0]:
                m[1] = None
        elif kind == "out":
            m[1] = round(self.t + 1.0 / max(1.0, self.project.settings.fps), 3)
            if m[0] is not None and m[0] >= m[1]:
                m[0] = None
        else:
            m[0] = m[1] = None
        self._emit_time(self.t)

    def set_marks(self, a: float, b: float):
        """Mark in and out at once (words selected in the transcript): a little room around the words."""
        if self.mode != "source":
            return
        dur = self.project.duration
        self.marks[self.source_fid] = [round(max(0.0, a - 0.05), 3), round(min(dur, b + 0.1), 3)]
        self.seek(max(0.0, a - 0.05))

    def marked_range(self) -> tuple[float, float]:
        """The marked part of the file in Source mode (the whole file where a mark isn't set)."""
        a, b = self.marks.get(self.source_fid, [None, None])
        dur = self.project.duration
        a = a if a is not None else 0.0
        b = b if b is not None else dur
        return a, max(a + 0.05, b)

    def show_source(self, clip_id: Optional[str], src_t: float = 0.0):
        """Temporarily show a clip's source frame (used while slipping/trimming)."""
        self._override = (clip_id, src_t) if clip_id else None
        self.refresh()

    def toggle_play(self):
        if self.playing:
            self.stop()
        else:
            self.play()

    def play(self):
        if not self.project.clips:
            return
        if self.t >= self.project.duration - 0.05:
            self.t = 0.0
        self.playing = True
        self._override = None
        self._restart_playback()
        self.timer.start(max(4, int(1000 / max(1.0, self.project.settings.fps) / 4)))
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.play_state.emit(True)

    def stop(self):
        if not self.playing:
            return
        self.playing = False
        self._audio_resume = False
        self.timer.stop()
        self.engine.stop_playback()
        self.player.pause()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_state.emit(False)
        self.refresh()   # exact frame at the stop position

    def step(self, frames: int):
        self.stop()
        self.seek(self.t + frames / max(1.0, self.project.settings.fps))

    def prev_clip(self):
        if self.mode == "source":              # the file's marks
            a, _ = self.marks.get(self.source_fid, [None, None])
            self.seek(a if a is not None and a < self.t - 0.05 else 0.0)
            return
        starts = self.project.clip_starts()
        prev = [s for s in starts if s < self.t - 0.05]
        self.seek(prev[-1] if prev else 0.0)

    def next_clip(self):
        if self.mode == "source":
            _, b = self.marks.get(self.source_fid, [None, None])
            self.seek(b if b is not None and b > self.t + 0.05 else self.project.duration)
            return
        starts = self.project.clip_starts()
        nxt = [s for s in starts if s > self.t + 0.01]
        if nxt:
            self.seek(nxt[0])

    # ------------------------------------------------------------------ rendering
    def _restart_playback(self):
        """(Re)start reading ahead from self.t; the clock and audio start once frames are buffered."""
        self._play_t0 = self.t
        self._play_clock = None
        self._audio_resume = False
        self.player.pause()
        self._play_shots = self._shots()
        self._play_overlays = self._overlay_shots()
        self.engine.start_playback(self._play_shots, self._look(), self._play_t0, None,
                                   self.project.settings.fps)

    def _tick(self):
        if not self.playing:
            return
        if self._play_clock is None:
            if not self.engine.ready_to_start():
                return
            self._play_clock = time.perf_counter()
            self.engine.set_clock(self._play_clock)
            self._audio_check = 0.0
        t = self._play_t0 + (time.perf_counter() - self._play_clock)
        self._sync_audio(t)
        if t >= self.project.duration:
            self.t = max(0.0, self.project.duration - 1e-3)
            self.stop()
            self._emit_time(self.t)
            return
        fr = self.engine.take_frame(t)
        if fr is not None:
            self._present(fr)
        fps = self.project.settings.fps
        if abs(t - self.t) < 0.5 / max(1.0, fps):
            return
        self.t = t
        self.lbl_time.setText(f"{fmt_time(self.t, fps)} / {fmt_time(self.project.duration, fps)}")
        self._emit_time(self.t)

    def _sync_audio(self, t: float):
        """Keep the preview mix playing in step with the clock. Also restarts it after an edit during
        playback paused it, even when the new mix isn't ready yet (the previous one plays meanwhile)."""
        now = time.perf_counter()
        if now < self._audio_check or self._audio_resume or self.player.source().isEmpty():
            return
        self._audio_check = now + 0.5
        if self.player.mediaStatus() not in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia,
                                             QMediaPlayer.BufferingMedia, QMediaPlayer.EndOfMedia):
            return
        dur = self.player.duration() / 1000.0
        if dur > 0 and t >= dur - 0.1:
            return
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            self.player.setPosition(int(t * 1000))
            self.player.play()
        elif abs(self.player.position() / 1000.0 - t) > 0.3:
            self.player.setPosition(int(t * 1000))

    def preview_size(self) -> tuple[int, int]:
        s = self.project.settings
        tw, th = frame_size(s.aspect, s.resolution)
        k = self.PREVIEW_SHORT / min(tw, th)
        return max(2, int(tw * k) // 2 * 2), max(2, int(th * k) // 2 * 2)

    def _look(self) -> Look:
        s = self.project.settings
        tw, th = self.preview_size()
        color = self.adjust_mode == "on" or (self.adjust_mode == "playback" and not self.playing)
        end = self.project.duration
        fi, fo = s.fades(end)
        return Look(s.use_proxies, s.fit_mode, tw, th, self.PREVIEW_SHORT, color, fi, fo, end, self._overlay_shots())

    def _overlay_shots(self) -> tuple:
        """The overlay layers (private copies), bottom layer first."""
        out = []
        for c in sorted(self.project.overlays, key=lambda x: x.track):
            f = self.project.footage.get(c.footage_id)
            if f is not None and c.duration > 0 and c.start < self.project.duration:
                tw, th = self.preview_size()
                w, h = reframe.overlay_size(*spherical.dims(f, c), tw, th, c.scale, c.box_w, c.box_h)
                out.append(OverlayShot(OverlayClip.from_dict(c.to_dict()), f.log_profile, f.tonemap, f,
                                       motion.of_clip(c, f, w / max(1, h))))
        return tuple(out)

    def overlay_boxes(self, t: float) -> list:
        """Where the video-layer clips showing at t sit in the frame (see Canvas.overlay_boxes)."""
        tw, th = self.preview_size()
        out = []
        for c in sorted(self.project.overlays, key=lambda x: x.track):
            f = self.project.footage.get(c.footage_id)
            if f is None or not (c.start <= t < c.end) or c.start >= self.project.duration:
                continue
            w, h = reframe.overlay_size(*spherical.dims(f, c), tw, th, c.scale, c.box_w, c.box_h)
            out.append((c.id, c.x, c.y, w / tw, h / th, c.rotation))
        return out

    def _shot(self, clip, start: float, index: Optional[int] = None) -> Optional[Shot]:
        f = self.project.footage.get(clip.footage_id)
        if f is None:
            return None
        g = clip.grade or Grade()
        return Shot(self.project.index_of(clip.id) if index is None else index, start, clip.duration, clip.src_in,
                    f.id, Grade.from_dict(g.to_dict()), f.log_profile, f.tonemap, clip.zoom, clip.offset_x,
                    clip.offset_y, f, rot=clip.rotation,
                    box=(clip.box_x, clip.box_y, clip.box_w, clip.box_h) if clip.box_w > 0 and clip.box_h > 0 else (),
                    speed=clip.speed, ramp=clip.ramp, src_len=clip.src_dur,
                    view=spherical.view_of(clip) if spherical.is_360(f) else None,
                    motion=motion.of_clip(clip, f, self._aspect(clip)))

    def _aspect(self, clip) -> float:
        """Width / height of the picture a main-track clip shows in (its box, or the frame)."""
        tw, th = self.preview_size()
        if clip.box_w > 0 and clip.box_h > 0:
            tw, th = clip.box_w * tw, clip.box_h * th
        return tw / max(1e-6, th)

    def _shots(self) -> list[Shot]:
        out, t = [], 0.0
        es = edits(self.project) + [None]
        for i, c in enumerate(self.project.clips):
            sh = self._shot(c, t, i)
            if sh is not None:
                e_in, e_out = es[i], es[i + 1]
                sh = replace(sh, trans_in=TRANSITIONS[e_in.trans][1] if e_in.half else "", half_in=e_in.half,
                             half_out=e_out.half if e_out is not None else 0.0)
                out.append(sh)
            t += c.duration
        return out

    def refresh(self):
        """Re-render the current position (on the preview thread; the result arrives in _on_still)."""
        proj = self.project
        dur = proj.duration
        fps = proj.settings.fps
        self.lbl_time.setText(f"{fmt_time(self.t, fps)} / {fmt_time(dur, fps)}")
        self.canvas.caption_style = proj.settings.captions
        if not proj.clips:
            self.canvas.image = None
            self.canvas.caption = None
            self.lbl_info.setText("")
            self.canvas.update()
            return
        if self.playing:
            # Edited during playback: re-read the timeline from here if what plays changed.
            if self._shots() != self._play_shots or self._overlay_shots() != self._play_overlays:
                self._restart_playback()
            return
        idx = proj.index_of(self._override[0]) if self._override else -1
        if idx >= 0:
            shot = self._shot(proj.clips[idx], proj.clip_starts()[idx], idx)
            if shot is not None:
                self.engine.request_still(shot, self._override[1], self.t, self._look())
            return
        self._override = None
        idx, off = proj.clip_at(self.t)
        if idx is None:
            return
        shots = self._shots()
        shot = next((s for s in shots if s.index == idx), None)
        if shot is not None:
            self.engine.request_still(shot, shot.src_at(off), self.t, self._look(), shots)

    def _on_still(self, r: Rendered):
        if not self.playing:
            self._present(r, still=True)

    def _present(self, r: Rendered, still: bool = False):
        if r.image is None:
            self.canvas.image = None
            self.canvas.message = r.info
            self.canvas.update()
            return
        self.lbl_info.setText(r.info)
        self.lbl_info.setToolTip(r.info)
        out = self._frame = r.image  # keep buffer alive for QImage
        self.canvas.image = QImage(out.data, out.shape[1], out.shape[0], out.strides[0], QImage.Format_RGB32)
        self.canvas.caption = None
        t = self.t if still else r.t
        self.canvas.texts = self.project.texts
        self.canvas.text_t = t if self._override is None else -1.0
        self.canvas.overlay_boxes = self.overlay_boxes(t) if self._override is None else []
        self.canvas.overlay_gain = self._look().fade_gain(t)
        st = self.project.settings.captions
        if self._override is None and st.enabled and self.mode == "timeline":
            for c in self.captions:
                if c.start <= t < c.end:
                    self.canvas.caption, self.canvas.caption_t = c, t
                    tw, th = self.preview_size()
                    self.canvas.layout = caption_layout(st, tw, th, self.project.settings.video_format)
                    break
        self.canvas.update()


class _Bridge(QObject):
    """Carries rendered stills from the preview thread to the UI thread (queued signal)."""
    still = Signal(object)
