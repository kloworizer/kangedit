"""Inspector: trim points, log conversion, color correction, reframing and audio for one clip."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMenu, QPushButton, QScrollArea, QSlider, QToolButton, QVBoxLayout, QWidget)

from .. import lut as lutlib
from ..color import LOG_CHOICES, LOG_PROFILES, TONEMAPS, effective_profile, lut_profile, profile_label
from ..edits import TRANSITIONS
from ..models import OverlayClip, Project, TimelineClip
from ..speed import MAX_SPEED, MIN_SPEED, RAMPS, SOUND_MODES, sound_mode
from ..speed import clamp as clamp_speed


class ParamSlider(QWidget):
    """Slider + spinbox pair working in float units."""

    changed = Signal(float)

    def __init__(self, lo: float, hi: float, default: float, step: float = 1.0, decimals: int = 0, suffix: str = ""):
        super().__init__()
        self.default = default
        self.scale = 1.0 / step
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(lo * self.scale), int(hi * self.scale))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(82)
        self.btn_reset = QToolButton()
        self.btn_reset.setText("↺")
        self.btn_reset.setToolTip(f"Reset to {default:g}{suffix}")
        self.btn_reset.setAutoRaise(True)
        self.btn_reset.setObjectName("reset")    # no padding: the arrow fits the narrow button
        self.btn_reset.setFixedWidth(22)
        self.btn_reset.clicked.connect(self.reset)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        lay.addWidget(self.btn_reset)
        self.set_value(default)

    def reset(self):
        if abs(self.value() - self.default) > 1e-9:
            self.set_value(self.default, emit=True)

    def mouseDoubleClickEvent(self, _):
        self.reset()

    def _sync_reset(self):
        self.btn_reset.setEnabled(abs(self.spin.value() - self.default) > 1e-9)

    def _from_slider(self, v):
        val = v / self.scale
        self.spin.blockSignals(True)
        self.spin.setValue(val)
        self.spin.blockSignals(False)
        self._sync_reset()
        self.changed.emit(val)

    def _from_spin(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(val * self.scale)))
        self.slider.blockSignals(False)
        self._sync_reset()
        self.changed.emit(val)

    def set_value(self, val: float, emit: bool = False):
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.slider.setValue(int(round(val * self.scale)))
        self.spin.setValue(val)
        for w in (self.slider, self.spin):
            w.blockSignals(False)
        self._sync_reset()
        if emit:
            self.changed.emit(val)

    def value(self) -> float:
        return self.spin.value()


GRADE_PARAMS = [
    ("exposure", "Exposure", -4, 4, 0, 0.05, 2, " EV"),
    ("contrast", "Contrast", -100, 100, 0, 1, 0, ""),
    ("highlights", "Highlights", -100, 100, 0, 1, 0, ""),
    ("shadows", "Shadows", -100, 100, 0, 1, 0, ""),
    ("temperature", "Temperature", -100, 100, 0, 1, 0, ""),
    ("tint", "Tint", -100, 100, 0, 1, 0, ""),
    ("saturation", "Saturation", 0, 200, 100, 1, 0, " %"),
    ("vibrance", "Vibrance", -100, 100, 0, 1, 0, ""),
    # Sharpness and halation depend on neighbouring pixels: applied after the look, not part of the LUT.
    ("sharpness", "Sharpness", -100, 100, 0, 1, 0, ""),
    ("halation", "Halation", 0, 100, 0, 1, 0, ""),
]
PARAM_TIPS = {
    "sharpness": "Sharpen fine detail (negative values soften)",
    "halation": "Film's warm red glow around bright highlights",
}


IMPORT_LUT = "__import_lut__"


def wrap_angle(deg: float) -> float:
    """An angle in degrees within (-180, 180]."""
    deg = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if deg == -180.0 else round(deg, 3)


def fill_lut_combo(cb: QComboBox, base: list[tuple[str, str]], key_of=lambda path: path) -> None:
    """Fixed choices, then every library LUT, then an "Import LUT..." entry."""
    cb.blockSignals(True)
    cb.clear()
    for key, label in base:
        cb.addItem(label, key)
    luts = lutlib.library()
    if luts:
        cb.insertSeparator(cb.count())
        for name, path in luts:
            cb.addItem(f"LUT: {name}", key_of(path))
    cb.insertSeparator(cb.count())
    cb.addItem("Import LUT (.cube)…", IMPORT_LUT)
    cb.blockSignals(False)


def select_data(cb: QComboBox, key: str, label: str) -> None:
    """Select `key`, adding it first if the combo doesn't list it (e.g. a LUT from another machine)."""
    idx = cb.findData(key)
    if idx < 0:
        cb.insertItem(cb.count() - 2, label, key)
        idx = cb.findData(key)
    cb.setCurrentIndex(idx)


class Inspector(QScrollArea):
    # kind: "trim" | "grade" | "footage_color" | "reframe" | "audio"
    before_change = Signal(str)
    changed = Signal(str)
    # auto_color | auto_color_all | copy_grade_all | reset_color | detect_log | detect_log_all
    # | import_lut_log | import_lut_look
    request = Signal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.clip: Optional[TimelineClip] = None
        self.others: list[TimelineClip] = []          # the rest of the selection: edits apply to them too
        self.selection: Callable[[], list[str]] = list  # ids of the selected clips (set by the main window)
        self.layer_selection: Callable[[], list[str]] = list   # and of the selected clips on the video layers
        self.layer_others: list = []                  # those (but the shown one): color, speed, volume apply
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # fit the panel's width, never cut off
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 14, 6)    # room for the scroll bar on the right

        self.lbl_title = QLabel("Select a clip on the timeline")
        self.lbl_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        self.lbl_title.setWordWrap(True)
        lay.addWidget(self.lbl_title)
        # A clip on a video layer: its position, size and opacity are in the Layer tab.
        self.b_layer = QPushButton("🧩 Position, size && opacity on the layer…")
        self.b_layer.clicked.connect(lambda: self.request.emit("layer_tab"))
        self.b_layer.hide()
        lay.addWidget(self.b_layer)

        # --- Clip timing
        g = QGroupBox("Clip")
        f = QFormLayout(g)
        self.sp_in = QDoubleSpinBox(decimals=3, singleStep=0.04, suffix=" s")
        self.sp_out = QDoubleSpinBox(decimals=3, singleStep=0.04, suffix=" s")
        self.lbl_dur = QLabel("")
        self.sp_in.setRange(0, 1e6)
        self.sp_out.setRange(0, 1e6)
        self.sp_in.valueChanged.connect(lambda v: self._set_trim("in", v))
        self.sp_out.valueChanged.connect(lambda v: self._set_trim("out", v))
        f.addRow("Source in", self.sp_in)
        f.addRow("Source out", self.sp_out)
        f.addRow("Duration", self.lbl_dur)
        lay.addWidget(g)

        # --- Speed: slow / fast motion and ramps
        g = QGroupBox("Speed")
        f = QFormLayout(g)
        row = QHBoxLayout()
        row.setSpacing(3)
        for v in (0.25, 0.5, 1.0, 2.0, 4.0):
            b = QPushButton(f"{v:g}×")
            b.setToolTip("Normal speed" if v == 1 else (f"{v:g}× slow motion" if v < 1 else f"{v:g}× fast motion"))
            b.setFixedWidth(46)
            b.clicked.connect(lambda _=False, v=v: self.sp_speed.set_value(v, emit=True))
            row.addWidget(b)
        row.addStretch(1)
        f.addRow(row)
        self.sp_speed = ParamSlider(MIN_SPEED, MAX_SPEED, 1.0, 0.05, 2, "×")
        self.sp_speed.setToolTip("Below 1× is slow motion, above 1× fast motion. The clip gets longer or shorter "
                                 "on the timeline; its sound keeps its pitch.")
        self.sp_speed.changed.connect(self._set_speed)
        f.addRow("Speed", self.sp_speed)
        self.cb_ramp = QComboBox()
        self.cb_ramp.addItem("None (constant speed)", "")
        for key, (label, desc, _keys) in RAMPS.items():
            self.cb_ramp.addItem(label, key)
            self.cb_ramp.setItemData(self.cb_ramp.count() - 1, desc, Qt.ToolTipRole)
        self.cb_ramp.setToolTip("Speed ramp template: the speed changes along the clip (multiplied by the speed "
                                "above)")
        self.cb_ramp.currentIndexChanged.connect(lambda _: self._set_ramp())
        f.addRow("Ramp", self.cb_ramp)
        self.cb_sound = QComboBox()
        for key, (label, desc) in SOUND_MODES.items():
            self.cb_sound.addItem(label, key)
            self.cb_sound.setItemData(self.cb_sound.count() - 1, desc, Qt.ToolTipRole)
        self.cb_sound.setToolTip("What the clip's sound does at another speed: stay in sync with the picture "
                                 "(pitch kept or following the speed), play at normal speed, or be muted")
        self.cb_sound.currentIndexChanged.connect(lambda _: self._set_sound())
        f.addRow("Sound", self.cb_sound)
        self.grp_speed = g
        lay.addWidget(g)

        # --- Edit into this clip (from the previous one)
        g = QGroupBox("Edit into this clip")
        f = QFormLayout(g)
        self.cb_trans = QComboBox()
        self.cb_trans.addItem("None (cut)", "")
        for key, (label, _) in TRANSITIONS.items():
            self.cb_trans.addItem(label, key)
        self.sp_tdur = QDoubleSpinBox(decimals=2, singleStep=0.1, suffix=" s")
        self.sp_tdur.setRange(0.1, 3.0)
        self.sp_tdur.setToolTip("Centered on the cut; shortened automatically when the footage has no "
                                "spare frames before/after the clips")
        self.sp_split = QDoubleSpinBox(decimals=2, singleStep=0.1, suffix=" s")
        self.sp_split.setRange(-3.0, 3.0)
        self.sp_split.setToolTip("Move the audio cut:\n  negative = J-cut (this clip's sound starts early, "
                                 "under the previous picture)\n  positive = L-cut (the previous clip's sound "
                                 "runs on under this picture)")
        self.cb_trans.currentIndexChanged.connect(lambda _: self._set_edit("transition"))
        self.sp_tdur.valueChanged.connect(lambda _: self._set_edit("transition_dur"))
        self.sp_split.valueChanged.connect(lambda _: self._set_edit("audio_split"))
        f.addRow("Transition", self.cb_trans)
        f.addRow("Length", self.sp_tdur)
        f.addRow("Audio J / L", self.sp_split)
        self.grp_edit = g
        lay.addWidget(g)

        # --- Log conversion (per footage)
        g = QGroupBox("Log → Rec.709 (applies to the whole footage)")
        f = QFormLayout(g)
        self.lbl_camera = QLabel("")
        self.lbl_camera.setWordWrap(True)
        self.lbl_camera.setStyleSheet("color: #9aa4b2;")
        b_detect = QPushButton("Detect")
        b_detect.setToolTip("Identify the camera from the file's metadata and picture, and choose its log profile")
        b_detect.clicked.connect(lambda: self.request.emit("detect_log"))
        b_detect_all = QPushButton("Detect all")
        b_detect_all.setToolTip("Identify the camera of every imported file and choose each one's log profile")
        b_detect_all.clicked.connect(lambda: self.request.emit("detect_log_all"))
        cam_row = QVBoxLayout()     # the camera name above its buttons: fits a narrow panel
        cam_row.addWidget(self.lbl_camera)
        cam_btns = QHBoxLayout()
        cam_btns.addWidget(b_detect)
        cam_btns.addWidget(b_detect_all)
        cam_btns.addStretch(1)
        cam_row.addLayout(cam_btns)
        self.cb_log = QComboBox()
        self.cb_log.setToolTip("Built-in camera conversion, or an imported .cube LUT that converts log to Rec.709")
        self.cb_tone = QComboBox()
        for key, label in TONEMAPS:
            self.cb_tone.addItem(label, key)
        self.cb_log.currentIndexChanged.connect(lambda _: self._footage_color("log"))
        self.cb_tone.currentIndexChanged.connect(lambda _: self._footage_color("tone"))
        f.addRow("Camera", cam_row)
        f.addRow("Camera profile", self.cb_log)
        f.addRow("Conversion", self.cb_tone)
        lay.addWidget(g)

        # --- Color correction
        g = QGroupBox("Exposure && color")
        v = QVBoxLayout(g)
        btns = QGridLayout()
        b_auto = QPushButton("✨ Auto")
        b_auto.setObjectName("primary")
        b_auto.setToolTip("Analyze this clip and set exposure, white balance and contrast automatically")
        b_auto.clicked.connect(lambda: self.request.emit("auto_color"))
        b_auto_all = QPushButton("✨ Auto all clips")
        b_auto_all.setToolTip("Auto color every clip: the main track and the video on every layer")
        b_auto_all.clicked.connect(lambda: self.request.emit("auto_color_all"))
        b_reset = QPushButton("Reset")
        b_reset.clicked.connect(lambda: self.request.emit("reset_color"))
        b_copy = QPushButton("Copy to all clips")
        b_copy.setToolTip("Give every clip this clip's grade: the main track and the video on every layer "
                          "(pictures on the layers, such as logos, are left as they are)")
        b_copy.clicked.connect(lambda: self.request.emit("copy_grade_all"))
        btns.addWidget(b_auto, 0, 0)
        btns.addWidget(b_auto_all, 0, 1)
        btns.addWidget(b_reset, 1, 0)
        btns.addWidget(b_copy, 1, 1)
        v.addLayout(btns)
        form = QFormLayout()
        self.params: dict[str, ParamSlider] = {}
        for key, label, lo, hi, dflt, step, dec, suf in GRADE_PARAMS:
            ps = ParamSlider(lo, hi, dflt, step, dec, suf)
            ps.setToolTip((PARAM_TIPS[key] + ". " if key in PARAM_TIPS else "") + "Double-click to reset")
            ps.changed.connect(lambda val, k=key: self._set_grade(k, val))
            self.params[key] = ps
            form.addRow(label, ps)
        self.cb_look = QComboBox()
        self.cb_look.setToolTip("Creative color grading LUT (.cube), applied after the adjustments above")
        self.cb_look.currentIndexChanged.connect(self._set_look)
        self.look_mix = ParamSlider(0, 100, 100, 1, 0, " %")
        self.look_mix.setToolTip("How strongly the look LUT is applied. Double-click to reset")
        self.look_mix.changed.connect(lambda val: self._set_grade("look_mix", val))
        form.addRow("Look (LUT)", self.cb_look)
        form.addRow("Look strength", self.look_mix)
        v.addLayout(form)
        lay.addWidget(g)

        # --- Reframe
        g = QGroupBox("Reframe (crop position for the export ratio)")
        f = QFormLayout(g)
        self.rf_zoom = ParamSlider(1.0, 4.0, 1.0, 0.01, 2, "×")
        self.rf_x = ParamSlider(-1.0, 1.0, 0.0, 0.01, 2)
        self.rf_y = ParamSlider(-1.0, 1.0, 0.0, 0.01, 2)
        for ps, key in ((self.rf_zoom, "zoom"), (self.rf_x, "offset_x"), (self.rf_y, "offset_y")):
            ps.changed.connect(lambda val, k=key: self._set_attr(k, val, "reframe"))
        b_face = QPushButton("🙂 Zoom to face")
        b_face.setToolTip("Find the face in the clip and zoom in on it, to show off an expression or reaction.\n"
                          "Applies to every selected clip.")
        m = QMenu(b_face)
        for key, label in (("medium", "Medium shot"), ("close", "Close-up"), ("extreme", "Extreme close-up")):
            m.addAction(label, lambda k=key: self.request.emit("face_zoom:" + k))
        m.addSeparator()
        m.addAction("Reset framing", lambda: self.request.emit("reset_framing"))
        b_face.setMenu(m)
        f.addRow(b_face)
        f.addRow("Zoom", self.rf_zoom)
        f.addRow("Pan X", self.rf_x)
        f.addRow("Pan Y", self.rf_y)
        self.rf_rot = ParamSlider(-180.0, 180.0, 0.0, 0.5, 1, "°")
        self.rf_rot.setToolTip("Rotate the picture clockwise. Quarter turns (±90°, 180°) turn the footage before "
                               "it fills the frame, e.g. a phone clip recorded sideways. Double-click to reset")
        self.rf_rot.changed.connect(lambda val: self._set_attr("rotation", val, "reframe"))
        rot_row = QHBoxLayout()
        rot_row.setSpacing(4)
        rot_row.addWidget(self.rf_rot, 1)
        for text, step, tip in (("⟲", -90, "Rotate 90° left"), ("⟳", 90, "Rotate 90° right")):
            b = QToolButton(text=text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=step: self._rotate_by(k))
            rot_row.addWidget(b)
        f.addRow("Rotation", rot_row)
        self.grp_reframe = g
        lay.addWidget(g)

        # --- Audio
        g = QGroupBox("Audio")
        f = QFormLayout(g)
        self.vol = ParamSlider(0, 200, 100, 1, 0, " %")
        self.vol.changed.connect(lambda val: self._set_attr("volume", val / 100.0, "audio"))
        f.addRow("Volume", self.vol)
        lay.addWidget(g)
        lay.addStretch(1)
        self.reload_luts()

    def reload_luts(self):
        """Refresh the LUT lists (after an import) and re-select the current clip's values."""
        fill_lut_combo(self.cb_log, LOG_CHOICES, lut_profile)
        fill_lut_combo(self.cb_look, [("", "None")])
        self.set_clip(self.clip)

    def set_project(self, project: Project):
        self.project = project
        self.set_clip(None)

    @property
    def graded(self) -> list:
        """The clips a color, LUT, log, speed or volume change applies to: the targets and the selected clips on
        the video layers."""
        return self.targets + self.layer_others

    @property
    def targets(self) -> list[TimelineClip]:
        """The clips an edit applies to: the shown clip and the rest of the selection."""
        return ([self.clip] if self.clip is not None else []) + self.others

    def set_clip(self, clip):
        """Show a clip: one on the main track (TimelineClip), or on a video layer (OverlayClip)."""
        self.clip = clip
        layer = isinstance(clip, OverlayClip)
        sel = set(self.selection()) if clip is not None and not layer else set()
        self.others = [c for c in self.project.clips if c.id in sel and c is not clip] if clip and clip.id in sel else []
        lsel = set(self.layer_selection()) if clip is not None else set()
        self.layer_others = [o for o in self.project.overlays if o.id in lsel and o is not clip]
        self.widget().setEnabled(clip is not None)
        self.b_layer.setVisible(layer)
        self.grp_edit.setVisible(not layer)          # transitions and J / L cuts: between main-track clips
        self.grp_reframe.setVisible(not layer)       # a layer clip is placed and sized in the Layer tab
        if clip is None:
            self.lbl_title.setText("Select a clip on the timeline (main track or a video layer)")
            return
        f = self.project.footage[clip.footage_id]
        score = f"  ·  score {clip.score * 100:.0f}" if getattr(clip, "score", None) is not None else ""
        n = len(self.others) + len(self.layer_others) + 1
        many = f"\n{n} clips selected: changes apply to all of them" if n > 1 else ""
        where = f"  ·  layer V{clip.track + 2}" if layer else ""
        self.lbl_title.setText(f"{f.name}{where}{score}{many}")
        for w in (self.sp_in, self.sp_out, self.cb_log, self.cb_tone, self.cb_look, self.cb_trans, self.sp_tdur,
                  self.sp_split):
            w.blockSignals(True)
        if not layer:
            self.cb_trans.setCurrentIndex(max(0, self.cb_trans.findData(clip.transition)))
            self.sp_tdur.setValue(clip.transition_dur)
            self.sp_split.setValue(clip.audio_split)
            self.grp_edit.setEnabled(any(self.project.index_of(c.id) > 0 for c in self.targets))
        self.sp_in.setMaximum(f.duration)
        self.sp_out.setMaximum(f.duration)
        self.sp_in.setValue(clip.src_in)
        self.sp_out.setValue(clip.src_out)
        self.cb_ramp.blockSignals(True)
        self.cb_ramp.setCurrentIndex(max(0, self.cb_ramp.findData(clip.ramp if clip.ramp in RAMPS else "")))
        self.cb_ramp.blockSignals(False)
        self.cb_sound.blockSignals(True)
        self.cb_sound.setCurrentIndex(max(0, self.cb_sound.findData(sound_mode(clip.audio_mode))))
        self.cb_sound.blockSignals(False)
        self.sp_speed.set_value(clip.speed)
        self.grp_speed.setEnabled(not f.is_image)
        select_data(self.cb_log, f.log_profile, profile_label(f.log_profile))
        self.cb_tone.setCurrentIndex(max(0, self.cb_tone.findData(f.tonemap)))
        self.cb_tone.setEnabled(effective_profile(f.log_profile) in LOG_PROFILES)
        select_data(self.cb_look, clip.grade.look, f"LUT: {lutlib.lut_name(clip.grade.look)}")
        self.look_mix.setEnabled(bool(clip.grade.look))
        self.look_mix.set_value(clip.grade.look_mix)
        for w in (self.sp_in, self.sp_out, self.cb_log, self.cb_tone, self.cb_look, self.cb_trans, self.sp_tdur,
                  self.sp_split):
            w.blockSignals(False)
        self._show_camera(f)
        self.lbl_dur.setText(f"{clip.duration:.2f} s")
        for key, ps in self.params.items():
            ps.set_value(getattr(clip.grade, key))
        if not layer:
            self.rf_zoom.set_value(clip.zoom)
            self.rf_x.set_value(clip.offset_x)
            self.rf_y.set_value(clip.offset_y)
            self.rf_rot.set_value(clip.rotation)
        self.vol.set_value(clip.volume * 100.0)

    def _set_trim(self, which: str, v: float):
        c = self.clip
        if c is None:
            return
        f = self.project.footage[c.footage_id]
        self.before_change.emit("trim")
        if which == "in":
            c.src_in = min(max(0.0, v), c.src_out - 0.25)
        else:
            c.src_out = max(min(f.duration, v), c.src_in + 0.25)
        self.lbl_dur.setText(f"{c.duration:.2f} s")
        self.changed.emit("trim")

    def _set_speed(self, v: float):
        if self.clip is None:
            return
        self.before_change.emit("trim:speed")
        for c in self.graded:
            if not self.project.footage[c.footage_id].is_image:
                c.speed = clamp_speed(v)
        self.lbl_dur.setText(f"{self.clip.duration:.2f} s")
        self.changed.emit("trim")

    def _set_ramp(self):
        if self.clip is None:
            return
        self.before_change.emit("trim:ramp")
        for c in self.graded:
            if not self.project.footage[c.footage_id].is_image:
                c.ramp = self.cb_ramp.currentData() or ""
        self.lbl_dur.setText(f"{self.clip.duration:.2f} s")
        self.changed.emit("trim")

    def _set_sound(self):
        if self.clip is None:
            return
        self.before_change.emit("trim:audio_mode")
        for c in self.graded:
            if not self.project.footage[c.footage_id].is_image:
                c.audio_mode = self.cb_sound.currentData() or "sync"
        self.changed.emit("trim")

    def _set_edit(self, key: str):
        if self.clip is None:
            return
        val = {"transition": lambda: self.cb_trans.currentData() or "",
               "transition_dur": self.sp_tdur.value, "audio_split": self.sp_split.value}[key]()
        self.before_change.emit("trim:" + key)
        for c in self.targets:
            if self.project.index_of(c.id) > 0:     # the first clip has no edit into it
                setattr(c, key, val)
        self.changed.emit("trim")

    def _footage_color(self, which: str):
        c = self.clip
        if c is None:
            return
        if self.cb_log.currentData() == IMPORT_LUT:
            self.set_clip(c)   # put the previous choice back; the import selects the new LUT
            self.request.emit("import_lut_log")
            return
        self.before_change.emit("footage_color")
        for fid in dict.fromkeys(x.footage_id for x in self.graded):
            f = self.project.footage[fid]
            if which == "log" and f.log_profile != self.cb_log.currentData():
                f.log_profile = self.cb_log.currentData()
                f.log_note, f.log_pending = "Chosen manually", False
            elif which == "tone":
                f.tonemap = self.cb_tone.currentData()
        f = self.project.footage[c.footage_id]
        self.cb_tone.setEnabled(effective_profile(f.log_profile) in LOG_PROFILES)
        self._show_camera(f)
        self.changed.emit("footage_color")

    def _show_camera(self, f):
        self.lbl_camera.setText((f.camera or "Not identified") + (f"\n{f.log_note}" if f.log_note else ""))

    def _set_look(self):
        c = self.clip
        if c is None:
            return
        if self.cb_look.currentData() == IMPORT_LUT:
            self.set_clip(c)
            self.request.emit("import_lut_look")
            return
        self.before_change.emit("grade:look")
        for x in self.graded:
            x.grade.look = self.cb_look.currentData() or ""
        self.look_mix.setEnabled(bool(c.grade.look))
        self.changed.emit("grade")

    def _set_grade(self, key: str, val: float):
        if self.clip is None:
            return
        self.before_change.emit("grade:" + key)
        for c in self.graded:
            setattr(c.grade, key, float(val))
        self.changed.emit("grade")

    def _rotate_by(self, deg: float):
        """Quarter turn of every selected clip (kept within -180..180)."""
        if self.clip is None:
            return
        self.before_change.emit("reframe:rotate")
        for c in self.targets:
            c.rotation = wrap_angle(c.rotation + deg)
        self.rf_rot.set_value(self.clip.rotation)
        self.changed.emit("reframe")

    def _set_attr(self, key: str, val: float, kind: str):
        if self.clip is None:
            return
        self.before_change.emit(kind + ":" + key)
        for c in (self.graded if key == "volume" else self.targets):
            setattr(c, key, float(val))
        self.changed.emit(kind)
