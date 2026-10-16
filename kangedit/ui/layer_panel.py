"""Layer tab: position, size, rotation, opacity, fades and sound of the overlay selected on a video layer."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from ..models import OverlayClip, Project
from ..speed import MAX_SPEED, MIN_SPEED, RAMPS, SOUND_MODES, sound_mode
from ..speed import clamp as clamp_speed
from .inspector import ParamSlider, wrap_angle

LAYER_HELP = (
    "The lanes <b>V2, V3…</b> above the main track show pictures on top of the edit: picture in picture, "
    "a logo, B-roll, a second camera angle.<br><br>"
    "<b>Add</b><br>"
    "• Drag footage or an image from <b>Media</b> onto a layer (onto the main track it becomes a clip there)<br>"
    "• Or right-click footage in Media → <i>Add on a video layer at the playhead</i>, or right-click an empty "
    "layer → <i>Add the selected media here</i><br>"
    "• Several cameras of the same moments: <b>✨ Auto Edit → Multicam</b> syncs them by their sound and lays "
    "them out for you<br><br>"
    "<b>Place</b><br>"
    "• Drag the clip in the <b>viewer</b> to move it, mouse wheel over it to resize it<br>"
    "• Here: corner presets, position, size, rotation, opacity, fades and the clip's own volume; "
    "<b>Crop X / Y</b> for clips cropped to fill a box<br><br>"
    "<b>Time</b><br>"
    "• On the timeline, drag the clip to move it (up / down: another layer), drag its edges to trim<br>"
    "• Higher layers are in front; right-click a layer for more layers, a clip for fades, rotation, split, "
    "duplicate and multicam layouts<br>"
    "• Multicam angles are <b>🔗 linked</b> to their main clip: dragging an angle slips them all, its edges trim "
    "them all, R gives them another moment, together. <b>🔗 Link</b> (Ctrl+L) unlinks them to edit one on its "
    "own, and links it back in sync"
)

TIPS = ("Drag it in the viewer to move it · wheel over it to resize · on the timeline drag it up / down to "
        "change layer · right-click it for more")

# Position presets: (x, y, scale) of the overlay's centre and size.
PLACEMENTS = {
    "full": (0.5, 0.5, 1.0),
    "center": (0.5, 0.5, 0.5),
    "tl": (0.22, 0.22, 0.35),
    "tr": (0.78, 0.22, 0.35),
    "bl": (0.22, 0.78, 0.35),
    "br": (0.78, 0.78, 0.35),
}


class LayerPanel(QScrollArea):
    before_change = Signal(str)     # key: one undo step per slider drag
    changed = Signal()
    request = Signal(str, str)      # action ("ov_delete", "ov_place:<key>"...), overlay id

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.clip: Optional[OverlayClip] = None
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)    # dragged narrower than it: scroll sideways
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 14, 6)

        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        self.lbl_title.setWordWrap(True)
        lay.addWidget(self.lbl_title)
        self.lbl_hint = QLabel(LAYER_HELP)
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setTextFormat(Qt.RichText)
        self.lbl_hint.setStyleSheet("color: #aeb3bd;")
        lay.addWidget(self.lbl_hint)
        self.lbl_tips = QLabel(TIPS)
        self.lbl_tips.setWordWrap(True)
        self.lbl_tips.setStyleSheet("color: #8b909a; font-size: 11px;")
        lay.addWidget(self.lbl_tips)

        self.body = QWidget()
        v = QVBoxLayout(self.body)
        v.setContentsMargins(0, 0, 0, 0)
        self.b_color = QPushButton("🎨 Color, log, LUT, trim && speed…")
        self.b_color.setToolTip("Refine this layer clip like a main-track clip in the Clip tab: log conversion, "
                                "exposure and color, auto color, look LUT, source in / out, speed and volume")
        self.b_color.clicked.connect(lambda: self._emit("ov_color"))
        v.addWidget(self.b_color)

        g = QGroupBox("Position && size")
        f = QFormLayout(g)
        grid = QGridLayout()
        for k, (key, label) in enumerate((("full", "Full frame"), ("center", "Centre"), ("tl", "↖ Top left"),
                                          ("tr", "↗ Top right"), ("bl", "↙ Bottom left"), ("br", "↘ Bottom right"))):
            b = QPushButton(label)
            b.setToolTip("Picture in picture: a third of the frame in this corner" if len(key) == 2 else
                         "Fill the frame (letterboxed)" if key == "full" else "Half size in the centre")
            b.clicked.connect(lambda _=False, k=key: self._emit(f"ov_place:{k}"))
            grid.addWidget(b, k // 2, k % 2)
        f.addRow(grid)
        self.sl_x = ParamSlider(-50, 150, 50, 0.5, 1, " %")
        self.sl_y = ParamSlider(-50, 150, 50, 0.5, 1, " %")
        self.sl_scale = ParamSlider(2, 300, 100, 1, 0, " %")
        self.sl_scale.setToolTip("100 % fits the whole frame. Double-click to reset")
        self.sl_x.setToolTip("Horizontal position of the centre, % of the frame width")
        self.sl_y.setToolTip("Vertical position of the centre, % of the frame height")
        self.sl_rot = ParamSlider(-180, 180, 0, 0.5, 1, "°")
        self.sl_rot.setToolTip("Rotate clockwise. Double-click to reset")
        self.sl_x.changed.connect(lambda val: self._set("x", val / 100.0))
        self.sl_y.changed.connect(lambda val: self._set("y", val / 100.0))
        self.sl_scale.changed.connect(lambda val: self._set("scale", val / 100.0))
        self.sl_rot.changed.connect(lambda val: self._set("rotation", val))
        rot = QHBoxLayout()
        rot.setSpacing(4)
        rot.addWidget(self.sl_rot, 1)
        for text, step, tip in (("⟲", -90, "Rotate 90° left"), ("⟳", 90, "Rotate 90° right")):
            b = QToolButton(text=text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=step: self._rotate_by(k))
            rot.addWidget(b)
        f.addRow("X", self.sl_x)
        f.addRow("Y", self.sl_y)
        f.addRow("Size", self.sl_scale)
        f.addRow("Rotation", rot)
        self.sl_pan_x = ParamSlider(-1, 1, 0, 0.01, 2)
        self.sl_pan_y = ParamSlider(-1, 1, 0, 0.01, 2)
        for sl, key in ((self.sl_pan_x, "pan_x"), (self.sl_pan_y, "pan_y")):
            sl.setToolTip("Which part of the picture fills its box (multicam layouts crop around the face)")
            sl.changed.connect(lambda val, k=key: self._set(k, val))
        self.lbl_pan_x, self.lbl_pan_y = QLabel("Crop X"), QLabel("Crop Y")
        f.addRow(self.lbl_pan_x, self.sl_pan_x)
        f.addRow(self.lbl_pan_y, self.sl_pan_y)
        v.addWidget(g)

        g = QGroupBox("Look && timing")
        f = QFormLayout(g)
        self.sl_opacity = ParamSlider(0, 100, 100, 1, 0, " %")
        self.sl_opacity.changed.connect(lambda val: self._set("opacity", val / 100.0))
        self.sl_fade_in = ParamSlider(0, 5, 0, 0.05, 2, " s")
        self.sl_fade_out = ParamSlider(0, 5, 0, 0.05, 2, " s")
        self.sl_fade_in.changed.connect(lambda val: self._set("fade_in", val))
        self.sl_fade_out.changed.connect(lambda val: self._set("fade_out", val))
        self.sp_start = QDoubleSpinBox(decimals=2, singleStep=0.1, suffix=" s")
        self.sp_start.setRange(0, 1e5)
        self.sp_start.valueChanged.connect(lambda val: self._set("start", val, "start"))
        self.sp_len = QDoubleSpinBox(decimals=2, singleStep=0.1, suffix=" s")
        self.sp_len.setRange(0.25, 1e5)
        self.sp_len.valueChanged.connect(self._set_length)
        self.sp_layer = QSpinBox()
        self.sp_layer.setRange(2, 9)
        self.sp_layer.setPrefix("V")
        self.sp_layer.setToolTip("Higher layers are in front of lower ones")
        self.sp_layer.valueChanged.connect(lambda val: self._set("track", int(val) - 2, "track"))
        f.addRow("Opacity", self.sl_opacity)
        f.addRow("Fade in", self.sl_fade_in)
        f.addRow("Fade out", self.sl_fade_out)
        f.addRow("Starts at", self.sp_start)
        f.addRow("Length", self.sp_len)
        self.sl_speed = ParamSlider(MIN_SPEED, MAX_SPEED, 1.0, 0.05, 2, "×")
        self.sl_speed.setToolTip("Below 1× is slow motion, above 1× fast motion")
        self.sl_speed.changed.connect(lambda v: self._set("speed", clamp_speed(v)))
        f.addRow("Speed", self.sl_speed)
        self.cb_ramp = QComboBox()
        self.cb_ramp.addItem("None (constant speed)", "")
        for key, (label, desc, _keys) in RAMPS.items():
            self.cb_ramp.addItem(label, key)
            self.cb_ramp.setItemData(self.cb_ramp.count() - 1, desc, Qt.ToolTipRole)
        self.cb_ramp.currentIndexChanged.connect(lambda _: self._set("ramp", self.cb_ramp.currentData() or ""))
        f.addRow("Speed ramp", self.cb_ramp)
        self.cb_sound = QComboBox()
        for key, (label, desc) in SOUND_MODES.items():
            self.cb_sound.addItem(label, key)
            self.cb_sound.setItemData(self.cb_sound.count() - 1, desc, Qt.ToolTipRole)
        self.cb_sound.currentIndexChanged.connect(lambda _: self._set("audio_mode", self.cb_sound.currentData()))
        f.addRow("Sound at speed", self.cb_sound)
        f.addRow("Layer", self.sp_layer)
        v.addWidget(g)

        g = QGroupBox("Sound")
        f = QFormLayout(g)
        self.sl_volume = ParamSlider(0, 200, 100, 1, 0, " %")
        self.sl_volume.setToolTip("Volume of the overlay video's own sound (0 % mutes it)")
        self.sl_volume.changed.connect(lambda val: self._set("volume", val / 100.0, "volume"))
        f.addRow("Volume", self.sl_volume)
        self.grp_sound = g
        v.addWidget(g)

        self.chk_link = QCheckBox("🔗 Linked to the main clip (kept in sync)")
        self.chk_link.setToolTip("A multicam camera angle follows the main-track clip it was cut with: trimming, "
                                 "slipping or Other moment on either keeps every angle in sync. Untick to move "
                                 "and trim this angle on its own; tick again to put it back in sync (Ctrl+L).")
        self.chk_link.toggled.connect(lambda on: self._emit("link_toggle"))
        v.addWidget(self.chk_link)
        b_del = QPushButton("Delete from the layer")
        b_del.clicked.connect(lambda: self._emit("ov_delete"))
        v.addWidget(b_del)
        lay.addWidget(self.body)
        lay.addStretch(1)
        self.set_clip(None)

    def set_project(self, project: Project):
        self.project = project
        self.set_clip(None)

    def _emit(self, action: str):
        if self.clip is not None:
            self.request.emit(action, self.clip.id)

    def set_clip(self, clip: Optional[OverlayClip]):
        self.clip = clip
        self.body.setVisible(clip is not None)
        self.lbl_hint.setVisible(clip is None)
        self.lbl_tips.setVisible(clip is not None)
        if clip is None:
            self.lbl_title.setText("Video layers")
            return
        f = self.project.footage.get(clip.footage_id)
        self.lbl_title.setText(f"{f.name if f else '?'} on layer V{clip.track + 2}"
                               + (f" · angle {clip.angle}" if clip.angle >= 2 else " · main angle" if clip.angle else ""))
        for sl, val in ((self.sl_x, clip.x * 100), (self.sl_y, clip.y * 100), (self.sl_scale, clip.scale * 100),
                        (self.sl_rot, clip.rotation), (self.sl_opacity, clip.opacity * 100),
                        (self.sl_fade_in, clip.fade_in), (self.sl_fade_out, clip.fade_out),
                        (self.sl_volume, clip.volume * 100), (self.sl_pan_x, clip.pan_x), (self.sl_pan_y, clip.pan_y)):
            sl.set_value(val)
        self.chk_link.setVisible(bool(clip.link))
        self.chk_link.blockSignals(True)
        self.chk_link.setChecked(clip.linked)
        self.chk_link.blockSignals(False)
        boxed = clip.box_w > 0 and clip.box_h > 0
        for w in (self.sl_pan_x, self.sl_pan_y, self.lbl_pan_x, self.lbl_pan_y):
            w.setVisible(boxed)
        for sp, val in ((self.sp_start, clip.start), (self.sp_len, clip.duration), (self.sp_layer, clip.track + 2)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self.sp_len.setMaximum(max(0.25, ((f.duration if f else clip.src_out) - clip.src_in) / self._ratio(clip)))
        moving = not (f is not None and f.is_image)
        self.sl_speed.set_value(clip.speed)
        self.cb_ramp.blockSignals(True)
        self.cb_ramp.setCurrentIndex(max(0, self.cb_ramp.findData(clip.ramp if clip.ramp in RAMPS else "")))
        self.cb_ramp.blockSignals(False)
        self.cb_sound.blockSignals(True)
        self.cb_sound.setCurrentIndex(max(0, self.cb_sound.findData(sound_mode(clip.audio_mode))))
        self.cb_sound.blockSignals(False)
        for w in (self.sl_speed, self.cb_ramp, self.cb_sound):
            w.setEnabled(moving)
        sound = f is not None and f.has_audio and not f.is_image
        self.grp_sound.setEnabled(sound)
        self.grp_sound.setToolTip("" if sound else "This footage has no sound")

    def _set(self, key: str, val, undo_key: str = ""):
        c = self.clip
        if c is None:
            return
        self.before_change.emit(f"layer:{c.id}:{undo_key or key}")
        setattr(c, key, val)
        if key == "track":
            self.lbl_title.setText(self.lbl_title.text().rsplit(" V", 1)[0] + f" V{c.track + 2}")
        if key in ("speed", "ramp"):
            self.sp_len.blockSignals(True)
            self.sp_len.setValue(c.duration)
            self.sp_len.blockSignals(False)
        self.changed.emit()

    @staticmethod
    def _ratio(c: OverlayClip) -> float:
        """Source seconds per timeline second (the length on the timeline scales with the source used)."""
        return c.src_dur / c.duration if c.duration > 1e-6 else c.speed

    def _set_length(self, val: float):
        c = self.clip
        if c is not None:
            self._set("src_out", c.src_in + max(0.25, val) * self._ratio(c), "length")

    def _rotate_by(self, deg: float):
        if self.clip is not None:
            self._set("rotation", wrap_angle(self.clip.rotation + deg), "rotate")
            self.sl_rot.set_value(self.clip.rotation)
