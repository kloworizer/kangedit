"""Dialogs: auto edit, export, caption style, proxy and cache settings."""

from __future__ import annotations

import os
import re
import threading
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSpinBox,
                               QVBoxLayout, QWidget)

from .. import lut as lutlib
from ..autoedit import (FORMAT_PRESETS, ORDERS, STYLE_LABELS, TRANSITION_MODES, TRANSITION_STYLES, AutoEditSettings,
                        suggest_duration)
from ..captions import ANIMATIONS, LANGUAGES, WHISPER_MODELS, merged_transcript
from ..config import (ASPECT_RATIOS, CACHE_KINDS, FRAME_RATES, RESOLUTIONS, cache_root, default_cache_root,
                      fmt_bytes, folder_size, frame_size, open_path)
from ..export import CAPTION_MODES, CODECS, QUALITIES, ExportSettings
from ..fonts import ensure_qt_font, font_files, system_fonts
from ..models import Project
from .. import multicam
from ..polish import PACING

FIT_MODES = {"fill": "Fill (crop to frame)", "fit": "Fit (black bars)", "blur": "Fit with blurred background"}


def _combo(items: dict, current) -> QComboBox:
    cb = QComboBox()
    for k, v in items.items():
        cb.addItem(v, k)
    idx = cb.findData(current)
    cb.setCurrentIndex(max(0, idx))
    return cb


TWO_COLUMNS_WIDTH = 1360    # screen width (px) from which the Auto Edit options sit in two columns


class DurationSpinBox(QDoubleSpinBox):
    """Seconds, shown as minutes from two minutes on ("12 min 30 s"); typing "12:30", "12 min", "750" or
    "750 s" all work."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(0)
        self.setSingleStep(5)
        self.setAccelerated(True)

    def textFromValue(self, value: float) -> str:
        v = int(round(value))
        if v < 120:
            return f"{v} s"
        m, sec = divmod(v, 60)
        if m >= 60:
            h, m = divmod(m, 60)
            return f"{h} h {m} min" + (f" {sec} s" if sec else "")
        return f"{m} min" + (f" {sec} s" if sec else "")

    def valueFromText(self, text: str) -> float:
        v = parse_duration(text)
        return self.value() if v is None else v

    def validate(self, text: str, pos: int):
        from PySide6.QtGui import QValidator
        if parse_duration(text) is not None:
            return QValidator.Acceptable, text, pos
        return QValidator.Intermediate, text, pos

    def stepBy(self, steps: int):
        # bigger steps once the length is in minutes
        v = self.value()
        step = 5 if v < 120 else 30 if v < 1200 else 60
        self.setValue(v + steps * step)


def parse_duration(text: str) -> Optional[float]:
    """Seconds from "750", "750 s", "12:30", "1:02:30", "12 min", "12 min 30 s", "1 h 5 min" (None if not)."""
    t = text.strip().lower().replace(",", ".")
    if not t:
        return None
    if ":" in t:
        try:
            parts = [float(x) for x in t.split(":")]
        except ValueError:
            return None
        total = 0.0
        for x in parts:
            total = total * 60 + x
        return total
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*h(?:ours?)?)?\s*(?:(\d+(?:\.\d+)?)\s*m(?:in(?:utes?)?)?)?\s*"
                     r"(?:(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?)?", t)
    if m and any(m.groups()):
        h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mi * 60 + sec
    try:
        return float(t)
    except ValueError:
        return None


class AutoEditDialog(QDialog):
    def __init__(self, project: Project, selected_ids: list[str], last: AutoEditSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Edit — find the best moments")
        self.setMinimumWidth(520)
        self.project = project
        # The options scroll when the dialog is taller than the screen (the buttons stay in view).
        outer = QVBoxLayout(self)
        body = QWidget()
        # Two columns where the screen is wide enough (full HD and up), else one column that scrolls.
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        self.two_columns = screen.width() >= TWO_COLUMNS_WIDTH
        cols = QHBoxLayout(body)
        cols.setContentsMargins(0, 0, 0, 0)
        lay = QVBoxLayout()
        lay2 = QVBoxLayout() if self.two_columns else lay
        cols.addLayout(lay, 1)
        if self.two_columns:
            cols.addSpacing(10)
            cols.addLayout(lay2, 1)
        self._scroll = QScrollArea()
        self._scroll.setWidget(body)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll, 1)

        # --- Format
        g = QGroupBox("Video format")
        v = QVBoxLayout(g)
        self.fmt_group = QButtonGroup(self)
        row = QHBoxLayout()
        self.fmt_buttons = {}
        for key, preset in FORMAT_PRESETS.items():
            rb = QRadioButton(preset["label"])
            self.fmt_group.addButton(rb)
            self.fmt_buttons[key] = rb
            rb.toggled.connect(lambda on, k=key: on and self._apply_format(k))
            row.addWidget(rb)
        v.addLayout(row)
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color: #8b909a;")
        self.lbl_hint.setWordWrap(True)
        v.addWidget(self.lbl_hint)
        self.chk_aspect = QCheckBox("Also switch the project frame to the format's aspect ratio")
        self.chk_aspect.setChecked(True)
        v.addWidget(self.chk_aspect)
        lay.addWidget(g)

        # --- Timing
        g = QGroupBox("Timing")
        f = QFormLayout(g)
        self.sp_target = DurationSpinBox()
        self.sp_target.setRange(5, 6 * 3600)
        self.sp_min = QDoubleSpinBox(decimals=1, singleStep=0.5, suffix=" s")
        self.sp_min.setRange(0.5, 60)
        self.sp_max = QDoubleSpinBox(decimals=1, singleStep=0.5, suffix=" s")
        self.sp_max.setRange(0.5, 120)
        self.sp_gap = QDoubleSpinBox(decimals=1, singleStep=0.5, suffix=" s")
        self.sp_gap.setRange(0, 600)
        self.sp_gap.setToolTip("When the same footage is used more than once, its segments are at least this far "
                               "apart in the source and never overlap.")
        self.sp_uses = QSpinBox()
        self.sp_uses.setRange(0, 100)
        self.sp_uses.setSpecialValueText("Unlimited")
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.sp_min)
        clip_row.addWidget(QLabel("to"))
        clip_row.addWidget(self.sp_max)
        self.lbl_target = QLabel("")
        self.lbl_target.setStyleSheet("color: #8b909a;")
        self._target_auto = True     # follows the footage until the user types a length
        self.sp_target.valueChanged.connect(lambda _: setattr(self, "_target_auto", False))
        trow = QHBoxLayout()
        trow.addWidget(self.sp_target)
        trow.addWidget(self.lbl_target, 1)
        f.addRow("Target length", trow)
        f.addRow("Clip length", clip_row)
        f.addRow("Gap between reuses", self.sp_gap)
        f.addRow("Max uses per footage", self.sp_uses)
        lay.addWidget(g)

        # --- Selection
        g = QGroupBox("Selection")
        f = QFormLayout(g)
        self.cb_style = _combo(STYLE_LABELS, last.style)
        self.cb_order = _combo(ORDERS, last.order)
        self.chk_every = QCheckBox("Use every footage at least once")
        self.chk_every.setChecked(last.use_every_footage)
        self.chk_every.setToolTip("Every usable file gets a clip, shorter ones when that fits them in the target "
                                  "length; with more files than fit even at the shortest clip length, the edit "
                                  "runs longer than the target.")
        self.chk_replace = QCheckBox("Replace current timeline (otherwise append)")
        self.chk_replace.setChecked(last.replace_timeline)
        f.addRow("Prioritize", self.cb_style)
        f.addRow("Clip order", self.cb_order)
        f.addRow(self.chk_every)
        f.addRow(self.chk_replace)
        lay.addWidget(g)

        # --- Transitions & story
        g = QGroupBox("Transitions && story")
        f = QFormLayout(g)
        self.cb_trans = _combo(TRANSITION_MODES, last.transitions)
        self.cb_tstyle = _combo(TRANSITION_STYLES, last.transition_style)
        self.sp_tdur = QDoubleSpinBox(decimals=2, singleStep=0.1, suffix=" s")
        self.sp_tdur.setRange(0.1, 3.0)
        self.sp_tdur.setValue(last.transition_dur)
        trow = QHBoxLayout()
        trow.addWidget(self.cb_tstyle, 1)
        trow.addWidget(self.sp_tdur)
        self.cb_trans.currentIndexChanged.connect(
            lambda: (self.cb_tstyle.setEnabled(self.cb_trans.currentData() != "off"),
                     self.sp_tdur.setEnabled(self.cb_trans.currentData() != "off")))
        self.chk_jl = QCheckBox("J / L cuts: let sound lead into or run on past a cut (long-form)")
        self.chk_jl.setToolTip("Where the next scene starts with sound, it is heard just before the picture "
                               "changes (J-cut); where a voice is still going at a cut, it finishes over the next "
                               "picture (L-cut).")
        self.chk_jl.setChecked(last.jl_cuts)
        self.chk_bookends = QCheckBox("Find the opening and closing (file names, recording time, speech)")
        self.chk_bookends.setToolTip("Puts the clips in recording order and looks for the greeting that opens the "
                                     "video and the sign-off that closes it (\"hello…\", \"welcome…\", "
                                     "\"assalamualaikum…\", \"thanks for watching\", \"sampai jumpa\"…) in "
                                     "intro/outro-named files and the first/last recorded ones.\nFiles without a "
                                     "transcript get a quick listen to their first/last 2 minutes.")
        self.chk_bookends.setChecked(last.bookends)
        f.addRow("Transitions", self.cb_trans)
        f.addRow("Style / length", trow)
        f.addRow(self.chk_jl)
        f.addRow(self.chk_bookends)
        lay.addWidget(g)

        # --- Finishing
        g = QGroupBox("Finishing")
        f = QFormLayout(g)
        self.cb_pacing = _combo(PACING, last.pacing)
        self.cb_pacing.setToolTip("How the clip lengths change along the edit (within the clip lengths above)")
        self.chk_pauses = QCheckBox("Cut between words, never in the middle of one")
        self.chk_pauses.setToolTip("Each cut moves to the nearest pause (the transcript's word timings, else a "
                                   "quieter moment), up to 0.6 s")
        self.chk_pauses.setChecked(last.cut_on_pauses)
        self.chk_sentences = QCheckBox("Whole sentences (long form): every clip with speech holds complete sentences")
        self.chk_sentences.setToolTip("In a long-form edit each clip that has speech starts and ends on whole "
                                      "sentences of the transcript (a sentence cut into is taken in whole, or "
                                      "left out when the clip would get too long). Uses the transcripts: footage "
                                      "without one is cut between words as usual (Media panel → Transcribe).")
        self.chk_sentences.setChecked(last.whole_sentences)
        self.chk_beats = QCheckBox("Cut on the beat of the music (when the timeline has music)")
        self.chk_beats.setToolTip("Add music first (Sounds tab → Background music): the cuts then move to its nearest beat, up to "
                                  "half a beat")
        self.chk_beats.setChecked(last.beat_sync)
        self.chk_fillers = QCheckBox("Remove filler words, long pauses and false starts")
        self.chk_fillers.setToolTip("\"um\", \"eh\", \"anu\", silences over a second and words said twice in a "
                                    "row are cut out of the clips (jump cuts). Uses the transcripts: footage "
                                    "without one is transcribed first.")
        self.chk_fillers.setChecked(last.remove_fillers)
        self.chk_variety = QCheckBox("Vary the shot sizes (no three close-ups or wide shots in a row)")
        self.chk_variety.setChecked(last.shot_variety)
        self.chk_loud = QCheckBox("Even out the loudness of the clips")
        self.chk_loud.setChecked(last.even_loudness)
        f.addRow("Pacing", self.cb_pacing)
        for w in (self.chk_sentences, self.chk_pauses, self.chk_beats, self.chk_fillers, self.chk_variety,
                  self.chk_loud):
            f.addRow(w)
        lay2.addWidget(g)

        # --- Footage
        g = QGroupBox("Footage to use")
        v = QVBoxLayout(g)
        self.list = QListWidget()
        self.list.setMaximumHeight(130)
        use_ids = set(selected_ids) if len(selected_ids) > 1 else set(project.footage)
        for f_ in project.footage.values():
            if f_.is_image:      # pictures have no best moments: add them to the timeline by hand
                continue
            it = QListWidgetItem(f"{f_.name}  ({f_.duration:.0f}s)")
            it.setData(Qt.UserRole, f_.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if f_.id in use_ids else Qt.Unchecked)
            self.list.addItem(it)
        self.list.itemChanged.connect(lambda _: self._suggest_target())
        v.addWidget(self.list)
        self.grp_footage = g
        lay2.addWidget(g)
        lay2.addWidget(self._multicam_box(project))
        lay.addStretch(1)
        if self.two_columns:
            lay2.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Create edit")
        bb.button(QDialogButtonBox.Ok).setObjectName("primary")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)
        self._fit_screen()

        self.fmt_buttons.get(last.video_format, self.fmt_buttons["short"]).setChecked(True)
        # Restore the user's last values (they may have fine-tuned the preset).
        self.sp_target.setValue(last.target_duration)
        self.sp_min.setValue(last.clip_min)
        self.sp_max.setValue(last.clip_max)
        self.sp_gap.setValue(last.min_gap)
        self.sp_uses.setValue(last.max_uses)
        self.cb_order.setCurrentIndex(max(0, self.cb_order.findData(last.order)))
        self.cb_trans.setCurrentIndex(max(0, self.cb_trans.findData(last.transitions)))
        self.cb_tstyle.setCurrentIndex(max(0, self.cb_tstyle.findData(last.transition_style)))
        self.sp_tdur.setValue(last.transition_dur)
        self.chk_jl.setChecked(last.jl_cuts)
        self.chk_bookends.setChecked(last.bookends)
        self._target_auto = last.video_format == "long"   # long-form length follows the footage
        self._suggest_target()

    # ------------------------------------------------------------------ multicam
    ANGLE_LABELS = ["Main angle", "Angle 2", "Angle 3", "Angle 4", "Not used"]

    def _layout_changed(self):
        key = self.cb_layout.currentData()
        self.lbl_layout.setText(multicam.TEMPLATE_HINTS[key])
        # Switching shows one angle at a time: the sound stays on the main angle.
        self.cb_audio.setEnabled(key != "switch")
        self.cb_audio.setToolTip("Switching keeps the main angle's sound" if key == "switch" else "")

    def _multicam_box(self, project: Project) -> QGroupBox:
        g = QGroupBox("Multicam — several angles of the same moments")
        v = QVBoxLayout(g)
        self.chk_multi = QCheckBox("Sync the angles by their sound and show them together")
        self.chk_multi.setToolTip("For two or more cameras (or a phone and a camera) recording the same scene: "
                                  "their sound lines them up to the millisecond, the main angle is edited as "
                                  "usual, and the other angles follow it on the video layers, cut at the same "
                                  "moments, in the layout you choose.")
        v.addWidget(self.chk_multi)
        self.multi_body = QGroupBox()
        self.multi_body.setFlat(True)
        f = QFormLayout(self.multi_body)
        f.setContentsMargins(0, 0, 0, 0)
        self.angle_rows: dict[str, QComboBox] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        prev = (project.settings.multicam or {}).get("angles") or {}
        videos = [x for x in project.footage.values() if not x.is_image and x.has_audio]
        cams = list(dict.fromkeys(x.camera for x in videos if x.camera))
        for r, x in enumerate(videos):
            lbl = QLabel(x.name)
            lbl.setToolTip(f"{x.path}\n{x.camera or 'unknown camera'}")
            cb = QComboBox()
            cb.addItems(self.ANGLE_LABELS)
            if x.id in prev:
                k = prev[x.id]
            elif len(cams) > 1 and x.camera in cams:     # a guess: one angle per camera
                k = min(4, cams.index(x.camera) + 1)
            else:
                k = 1
            cb.setCurrentIndex(k - 1 if 1 <= k <= 4 else 4)
            cb.currentIndexChanged.connect(self._angles_changed)
            grid.addWidget(lbl, r, 0)
            grid.addWidget(cb, r, 1)
            self.angle_rows[x.id] = cb
        f.addRow(grid)
        last = project.settings.multicam or {}
        self.cb_layout = _combo(multicam.TEMPLATES, last.get("template", "pip"))
        self.cb_layout.currentIndexChanged.connect(self._layout_changed)
        self.lbl_layout = QLabel(multicam.TEMPLATE_HINTS[self.cb_layout.currentData()])
        self.lbl_layout.setWordWrap(True)
        self.lbl_layout.setStyleSheet("color: #8b909a;")
        self.cb_audio = QComboBox()
        self._last_audio = last.get("audio", "main")
        self.chk_covered = QCheckBox("Only use moments every angle recorded")
        self.chk_covered.setChecked(last.get("covered", True))
        self.chk_whole = QCheckBox("Use the whole recording (no best-moment cuts)")
        self.chk_whole.setToolTip("Put the main angle's whole recording on the timeline instead of choosing its "
                                  "best moments, like editing a full multicam event: with Switching, the angles "
                                  "are cut all the way through.")
        self.chk_whole.setChecked(bool(last.get("whole", False)))
        self.chk_whole.toggled.connect(lambda _: self._suggest_target())
        f.addRow("Layout", self.cb_layout)
        f.addRow(self.lbl_layout)
        f.addRow("Sound", self.cb_audio)
        f.addRow(self.chk_covered)
        f.addRow(self.chk_whole)
        self._layout_changed()
        self.lbl_angles = QLabel("")
        self.lbl_angles.setWordWrap(True)
        self.lbl_angles.setStyleSheet("color: #e0b050;")
        f.addRow(self.lbl_angles)
        v.addWidget(self.multi_body)
        self.chk_multi.toggled.connect(self._multi_toggled)
        self.chk_multi.setChecked(bool(last.get("angles")) and len(videos) > 1)
        self.chk_multi.setEnabled(len(videos) > 1)
        if len(videos) < 2:
            self.chk_multi.setToolTip("Import the recordings of two or more angles (with sound) first.")
        self._multi_toggled(self.chk_multi.isChecked())
        self._angles_changed()
        return g

    def _multi_toggled(self, on: bool):
        self.multi_body.setVisible(on)
        self.grp_footage.setVisible(not on)     # with multicam, the main angle's files are the ones edited
        self._suggest_target()
        self._fit_screen()

    def _fit_screen(self):
        """As tall as the options need, up to the screen's height (then they scroll)."""
        if not hasattr(self, "_scroll"):
            return
        body = self._scroll.widget()
        body.adjustSize()
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        want = body.sizeHint().height() + 70
        self._scroll.setMinimumHeight(min(want - 60, max(200, screen.height() - 160)))
        self.resize(max(self.width(), body.sizeHint().width() + 30), min(want, screen.height() - 60))

    def _angles(self) -> dict[str, int]:
        return {fid: cb.currentIndex() + 1 for fid, cb in self.angle_rows.items() if cb.currentIndex() < 4}

    def _angles_changed(self):
        angles = self._angles()
        used = sorted(set(angles.values()))
        n = max(used, default=1)
        cur = self.cb_audio.currentData() or self._last_audio
        self.cb_audio.blockSignals(True)
        self.cb_audio.clear()
        for k, label in multicam.audio_choices(n).items():
            self.cb_audio.addItem(label, k)
        self.cb_audio.setCurrentIndex(max(0, self.cb_audio.findData(cur)))
        self.cb_audio.blockSignals(False)
        problems = []
        if 1 not in used:
            problems.append("Choose at least one file as the main angle.")
        if len(used) < 2:
            problems.append("Give at least one file another angle (Angle 2…).")
        elif used != list(range(1, n + 1)):
            problems.append("Number the angles without gaps (Angle 2 before Angle 3).")
        self.lbl_angles.setText(" ".join(problems))
        self._suggest_target()

    def multicam(self):
        """The multicam setup, or None when it is off: {"angles", "template", "audio", "covered", "whole"}."""
        if not self.chk_multi.isChecked():
            return None
        angles = self._angles()
        if 1 not in angles.values() or len(set(angles.values())) < 2:
            return None
        return {"angles": angles, "template": self.cb_layout.currentData(),
                "audio": self.cb_audio.currentData() or "main", "covered": self.chk_covered.isChecked(),
                "whole": self.chk_whole.isChecked()}

    def accept(self):
        if self.chk_multi.isChecked() and self.lbl_angles.text():
            QMessageBox.information(self, "Multicam", self.lbl_angles.text())
            return
        super().accept()

    def _apply_format(self, key: str):
        preset = FORMAT_PRESETS[key]
        self.lbl_hint.setText(preset["hint"])
        self.chk_aspect.setEnabled(preset["aspect"] is not None)
        s = preset["settings"]
        if not s:
            return
        self.sp_target.setValue(s["target_duration"])
        self.sp_min.setValue(s["clip_min"])
        self.sp_max.setValue(s["clip_max"])
        self.sp_gap.setValue(s["min_gap"])
        self.cb_order.setCurrentIndex(max(0, self.cb_order.findData(s["order"])))
        if hasattr(self, "cb_trans"):
            self.cb_trans.setCurrentIndex(max(0, self.cb_trans.findData(s["transitions"])))
            self.cb_tstyle.setCurrentIndex(max(0, self.cb_tstyle.findData(s["transition_style"])))
            self.sp_tdur.setValue(s["transition_dur"])
            self.chk_jl.setChecked(s["jl_cuts"])
            self.chk_bookends.setChecked(s["bookends"])
            if "pacing" in s:
                self.cb_pacing.setCurrentIndex(max(0, self.cb_pacing.findData(s["pacing"])))
            self._target_auto = True
            self._suggest_target()

    def _suggest_target(self):
        """Long-form: target length from the total length of the checked footage."""
        if not hasattr(self, "list"):
            return
        total = sum(self.project.footage[fid].duration for fid in self.footage_ids() if fid in self.project.footage)
        if self.whole_recording():
            # The edit is the main angle's whole recording: its length is the target (not a choice).
            if self.sp_target.isEnabled():
                self._target_before = self.sp_target.value()
            self.sp_target.setEnabled(False)
            self.sp_target.blockSignals(True)
            self.sp_target.setValue(max(self.sp_target.minimum(), round(total)))
            self.sp_target.blockSignals(False)
            self.lbl_target.setText("the main angle's whole recording" if total else "")
            return
        if not self.sp_target.isEnabled():       # whole recording turned off: the length chosen before
            self.sp_target.setEnabled(True)
            self.sp_target.blockSignals(True)
            self.sp_target.setValue(getattr(self, "_target_before", self.sp_target.value()))
            self.sp_target.blockSignals(False)
        d = suggest_duration(self.format_key(), total)
        self.lbl_target.setText(f"of {total / 60:.1f} min footage" if total else "")
        if d is not None and self._target_auto and self.format_key() == "long":
            self.sp_target.blockSignals(True)
            self.sp_target.setValue(d)
            self.sp_target.blockSignals(False)

    def whole_recording(self) -> bool:
        """Multicam with "Use the whole recording" on."""
        multi, whole = getattr(self, "chk_multi", None), getattr(self, "chk_whole", None)
        return bool(multi is not None and whole is not None and multi.isChecked() and whole.isChecked())

    def format_key(self) -> str:
        for k, b in self.fmt_buttons.items():
            if b.isChecked():
                return k
        return "custom"

    def footage_ids(self) -> list[str]:
        """The footage to edit: the checked files, or with multicam the main angle's files."""
        if getattr(self, "chk_multi", None) is not None and self.chk_multi.isChecked():
            return [fid for fid, k in self._angles().items() if k == 1]
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def aspect(self):
        a = FORMAT_PRESETS[self.format_key()]["aspect"]
        return a if (a and self.chk_aspect.isChecked()) else None

    def settings(self) -> AutoEditSettings:
        lo, hi = sorted((self.sp_min.value(), self.sp_max.value()))
        return AutoEditSettings(
            video_format=self.format_key(),
            target_duration=self.sp_target.value(),
            clip_min=lo,
            clip_max=hi,
            min_gap=self.sp_gap.value(),
            max_uses=self.sp_uses.value(),
            use_every_footage=self.chk_every.isChecked(),
            order=self.cb_order.currentData(),
            style=self.cb_style.currentData(),
            replace_timeline=self.chk_replace.isChecked(),
            transitions=self.cb_trans.currentData(),
            transition_style=self.cb_tstyle.currentData(),
            transition_dur=self.sp_tdur.value(),
            jl_cuts=self.chk_jl.isChecked(),
            bookends=self.chk_bookends.isChecked(),
            pacing=self.cb_pacing.currentData(),
            cut_on_pauses=self.chk_pauses.isChecked(),
            whole_sentences=self.chk_sentences.isChecked(),
            beat_sync=self.chk_beats.isChecked(),
            remove_fillers=self.chk_fillers.isChecked(),
            shot_variety=self.chk_variety.isChecked(),
            even_loudness=self.chk_loud.isChecked(),
        )


class ExportDialog(QDialog):
    def __init__(self, project: Project, has_captions: bool, parent=None, use_gpu: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Export video")
        self.setMinimumWidth(500)
        self.project = project
        s = project.settings
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.ed_out = QLineEdit()
        base = os.path.splitext(project.path)[0] if project.path else os.path.join(os.path.expanduser("~"), "Videos", "kangedit_export")
        self.ed_out.setText(base + ".mp4")
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.ed_out, 1)
        row.addWidget(b)
        form.addRow("Save to", row)

        self.cb_aspect = _combo({k: k for k in ASPECT_RATIOS}, s.aspect)
        self.cb_res = _combo({k: k for k in RESOLUTIONS}, s.resolution)
        self.cb_fps = QComboBox()
        for r in FRAME_RATES:
            self.cb_fps.addItem(f"{r:g} fps", r)
        self.cb_fps.setCurrentIndex(min(range(len(FRAME_RATES)), key=lambda i: abs(FRAME_RATES[i] - s.fps)))
        self.cb_fit = _combo(FIT_MODES, s.fit_mode)
        self.cb_codec = _combo({k: v[0] for k, v in CODECS.items()}, "h264")
        self.cb_quality = _combo({k: k for k in QUALITIES}, "High")
        self.cb_caps = _combo(CAPTION_MODES, "burn" if has_captions and s.captions.enabled else "none")
        self.cb_caps.setEnabled(has_captions)
        self.chk_proxy = QCheckBox("Draft export from proxies (fast, lower quality)")
        self.chk_gpu = QCheckBox("Encode on the graphics card when possible (faster)")
        self.chk_gpu.setToolTip("Uses the GPU's H.264/H.265 encoder (NVIDIA NVENC, AMD AMF or Intel Quick Sync).\n"
                                "Falls back to the CPU automatically if the card can't do it. ProRes always uses the CPU.\n"
                                "GPU files are a little larger at the same quality setting.")
        self.chk_gpu.setChecked(use_gpu)
        self.chk_blend = QCheckBox("Smooth slow motion (blend frames)")
        self.chk_blend.setToolTip("Slowed-down clips blend neighbouring frames instead of showing each one twice or "
                                  "more:\nsmoother movement, slightly softer. Off: crisp repeated frames.")
        self.chk_blend.setChecked(True)
        slow = [c for c in project.clips + project.overlays if c.ramp or c.speed < 1.0]
        self.chk_blend.setVisible(bool(slow))
        self.lbl_size = QLabel("")
        self.lbl_size.setStyleSheet("color: #8b909a;")
        for cb in (self.cb_aspect, self.cb_res):
            cb.currentIndexChanged.connect(self._update)
        self.cb_codec.currentIndexChanged.connect(self._codec_changed)
        form.addRow("Aspect ratio", self.cb_aspect)
        form.addRow("Resolution", self.cb_res)
        form.addRow("", self.lbl_size)
        form.addRow("Frame rate", self.cb_fps)
        form.addRow("Reframe", self.cb_fit)
        form.addRow("Format", self.cb_codec)
        form.addRow("Quality", self.cb_quality)
        form.addRow("Captions", self.cb_caps)
        form.addRow(self.chk_proxy)
        form.addRow(self.chk_gpu)
        form.addRow(self.chk_blend)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export")
        bb.button(QDialogButtonBox.Ok).setObjectName("primary")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._update()

    def _update(self):
        w, h = frame_size(self.cb_aspect.currentData(), self.cb_res.currentData())
        self.lbl_size.setText(f"Output {w} × {h}, duration {self.project.duration:.1f} s")

    def _codec_changed(self):
        ext = CODECS[self.cb_codec.currentData()][1]
        self.ed_out.setText(os.path.splitext(self.ed_out.text())[0] + ext)
        self.cb_quality.setEnabled(self.cb_codec.currentData() != "prores")
        self.chk_gpu.setEnabled(self.cb_codec.currentData() != "prores")

    def _browse(self):
        ext = CODECS[self.cb_codec.currentData()][1]
        path, _ = QFileDialog.getSaveFileName(self, "Export video", self.ed_out.text(), f"Video (*{ext})")
        if path:
            self.ed_out.setText(path)

    def _accept(self):
        if not self.ed_out.text().strip():
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.ed_out.text())), exist_ok=True)
        self.accept()

    def settings(self) -> ExportSettings:
        return ExportSettings(
            output=self.ed_out.text().strip(),
            aspect=self.cb_aspect.currentData(),
            resolution=self.cb_res.currentData(),
            fps=float(self.cb_fps.currentData()),
            fit_mode=self.cb_fit.currentData(),
            codec=self.cb_codec.currentData(),
            quality=self.cb_quality.currentData(),
            captions=self.cb_caps.currentData() if self.cb_caps.isEnabled() else "none",
            use_proxies=self.chk_proxy.isChecked(),
            use_gpu=self.chk_gpu.isChecked(),
            blend_slowmo=self.chk_blend.isChecked(),
        )


def font_combo(current: str) -> QComboBox:
    """Drop-down list of the installed font families (type a name to jump to it), each shown in its own font."""
    cb = QComboBox()
    cb.setMaxVisibleItems(20)
    families = list(system_fonts()) or QFontDatabase.families()
    for fam in families:
        cb.addItem(fam)
        cb.setItemData(cb.count() - 1, QFont(fam, 11), Qt.FontRole)
    select_font(cb, current)
    return cb


def select_font(cb: QComboBox, family: str) -> None:
    """Show `family` in a font_combo; a font that is not installed is added to the list so it stays chosen."""
    if not family:
        return
    idx = cb.findText(family, Qt.MatchFixedString)
    if idx < 0:
        cb.addItem(family)
        idx = cb.count() - 1
        cb.setItemData(idx, "Not installed: the export uses a fallback font", Qt.ToolTipRole)
    cb.setCurrentIndex(idx)


class CaptionStyleDialog(QDialog):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Captions")
        self.project = project
        st = project.settings.captions
        lay = QVBoxLayout(self)

        g = QGroupBox("Speech recognition (Whisper, offline)")
        f = QFormLayout(g)
        self.cb_model = QComboBox()
        for m, label in WHISPER_MODELS.items():
            self.cb_model.addItem(label, m)
        self.cb_model.setCurrentIndex(max(0, self.cb_model.findData(project.settings.whisper_model)))
        self.cb_model.setToolTip("large-v3 gives the most accurate captions; large-v3-turbo is almost as accurate and "
                                 "several times faster.\nThe model is downloaded once on first use.")
        self.cb_lang = QComboBox()
        for code, name in LANGUAGES:
            self.cb_lang.addItem(name, code)
        self.cb_lang.setCurrentIndex(max(0, self.cb_lang.findData(project.settings.whisper_language)))
        f.addRow("Model", self.cb_model)
        f.addRow("Language", self.cb_lang)
        lay.addWidget(g)

        g = QGroupBox("Style")
        f = QFormLayout(g)
        self.cb_font = font_combo(st.font)
        self.lbl_font = QLabel("Aa Bb 123 — the quick brown fox")
        self.lbl_font.setMinimumHeight(34)
        self.cb_font.currentTextChanged.connect(self._font_preview)
        self._font_preview(self.cb_font.currentText())
        self.sp_size = QDoubleSpinBox(decimals=1, singleStep=0.5, suffix=" % of height")
        self.sp_size.setRange(2, 15)
        self.sp_size.setValue(st.size_pct)
        self.cb_pos = _combo({"bottom": "Bottom", "middle": "Middle", "top": "Top"}, st.position)
        self.sp_chars = QSpinBox()
        self.sp_chars.setRange(8, 80)
        self.sp_chars.setValue(st.max_chars)
        self.sp_lines = QSpinBox()
        self.sp_lines.setRange(1, 3)
        self.sp_lines.setValue(st.max_lines)
        self.sp_dur = QDoubleSpinBox(decimals=1, suffix=" s")
        self.sp_dur.setRange(0.5, 10)
        self.sp_dur.setValue(st.max_duration)
        self.chk_upper = QCheckBox("UPPERCASE")
        self.chk_upper.setChecked(st.uppercase)
        self.chk_outline = QCheckBox("Outline")
        self.chk_outline.setChecked(st.outline)
        self.chk_box = QCheckBox("Background box")
        self.chk_box.setChecked(st.box)
        self.color = st.color
        self.b_color = QPushButton()
        self.b_color.clicked.connect(lambda: self._pick_color("color"))
        self.highlight = st.highlight
        self.b_highlight = QPushButton()
        self.b_highlight.clicked.connect(lambda: self._pick_color("highlight"))
        self._paint_color()
        self.cb_anim = _combo(ANIMATIONS, st.animation)
        self.cb_anim.setToolTip("Word-by-word animation, timed to the speech (like CapCut). Shown in the viewer "
                                "and burned into the export.")
        self.chk_auto = QCheckBox("Automatic size and position for the frame ratio and format")
        self.chk_auto.setToolTip("Vertical 9:16: lower middle, clear of the app buttons and text at the bottom.\n"
                                 "Square / 4:5: lower third.  Landscape: subtitles at the bottom "
                                 "(bigger and shorter lines for short-form).")
        self.chk_auto.setChecked(st.auto_layout)
        self.chk_auto.toggled.connect(self._auto_toggled)
        presets = QHBoxLayout()
        presets_buttons = []
        for name, cfg in (("Social (big, 1 line)", dict(size=8.0, chars=18, lines=1, dur=1.6, upper=True, pos="middle")),
                          ("Classic subtitles", dict(size=5.0, chars=42, lines=2, dur=5.0, upper=False, pos="bottom"))):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, c=cfg: self._preset(c))
            presets.addWidget(b)
            presets_buttons.append(b)
        f.addRow("Animation", self.cb_anim)
        f.addRow("Highlight color", self.b_highlight)
        f.addRow(self.chk_auto)
        f.addRow("Presets", presets)
        f.addRow("Font", self.cb_font)
        f.addRow("", self.lbl_font)
        f.addRow("Size", self.sp_size)
        f.addRow("Position", self.cb_pos)
        f.addRow("Max chars / line", self.sp_chars)
        f.addRow("Max lines", self.sp_lines)
        f.addRow("Max duration", self.sp_dur)
        f.addRow("Text color", self.b_color)
        self._manual = [self.sp_size, self.cb_pos, self.sp_chars, self.sp_lines, self.sp_dur] + presets_buttons
        row = QHBoxLayout()
        row.addWidget(self.chk_upper)
        row.addWidget(self.chk_outline)
        row.addWidget(self.chk_box)
        f.addRow(row)
        lay.addWidget(g)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._auto_toggled(self.chk_auto.isChecked())

    def _font_preview(self, family: str):
        ensure_qt_font(family)
        font = QFont(family)
        font.setPixelSize(22)
        font.setBold(True)
        self.lbl_font.setFont(font)
        installed = bool(font_files(family))
        self.lbl_font.setToolTip("" if installed else "This font is not installed: the export uses a fallback font.")

    def _preset(self, c):
        self.sp_size.setValue(c["size"])
        self.sp_chars.setValue(c["chars"])
        self.sp_lines.setValue(c["lines"])
        self.sp_dur.setValue(c["dur"])
        self.chk_upper.setChecked(c["upper"])
        self.cb_pos.setCurrentIndex(self.cb_pos.findData(c["pos"]))

    def _auto_toggled(self, on: bool):
        for w in self._manual:
            w.setEnabled(not on)

    def _pick_color(self, which: str):
        c = QColorDialog.getColor(QColor(getattr(self, which)), self, "Caption color")
        if c.isValid():
            setattr(self, which, c.name())
            self._paint_color()

    def _paint_color(self):
        for b, col in ((self.b_color, self.color), (self.b_highlight, self.highlight)):
            b.setText(col)
            b.setStyleSheet(f"background: {col}; color: black;")

    def apply(self):
        s = self.project.settings
        s.whisper_model = self.cb_model.currentData()
        s.whisper_language = self.cb_lang.currentData()
        st = s.captions
        st.font = self.cb_font.currentText()
        st.size_pct = self.sp_size.value()
        st.position = self.cb_pos.currentData()
        st.max_chars = self.sp_chars.value()
        st.max_lines = self.sp_lines.value()
        st.max_duration = self.sp_dur.value()
        st.uppercase = self.chk_upper.isChecked()
        st.outline = self.chk_outline.isChecked()
        st.box = self.chk_box.isChecked()
        st.color = self.color
        st.highlight = self.highlight
        st.animation = self.cb_anim.currentData()
        st.auto_layout = self.chk_auto.isChecked()


class ProxyDialog(QDialog):
    def __init__(self, project: Project, use_gpu: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Proxies")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Proxies are small H.264 copies used for smooth playback and faster\n"
                             "analysis. Export always uses the original files unless you pick a draft export."))
        f = QFormLayout()
        self.cb_res = QComboBox()
        for h in (360, 540, 720, 1080):
            self.cb_res.addItem(f"{h}p", h)
        self.cb_res.setCurrentIndex(max(0, self.cb_res.findData(project.settings.proxy_height)))
        self.chk_all = QCheckBox("Generate for all footage without a proxy now")
        self.chk_all.setChecked(True)
        self.chk_gpu = QCheckBox("Use the graphics card (decode and encode)")
        self.chk_gpu.setToolTip("Decode the originals and encode the proxies on the GPU (NVIDIA, AMD, Intel or Apple),\n"
                                "falling back to the CPU for footage the GPU can't decode. Older graphics cards may\n"
                                "not be faster than a many-core CPU.")
        self.chk_gpu.setChecked(use_gpu)
        f.addRow("Proxy resolution", self.cb_res)
        f.addRow(self.chk_all)
        f.addRow(self.chk_gpu)
        lay.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class CacheDialog(QDialog):
    """Where caches live and clearing them. The main window does the work through the two callbacks,
    since it has to release open proxies and fix footage paths: change_folder(path or "") -> bool and
    clear(kinds) -> None."""

    def __init__(self, change_folder, clear, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cache")
        self.setMinimumWidth(560)
        self._change_folder = change_folder
        self._clear = clear
        lay = QVBoxLayout(self)

        g = QGroupBox("Cache folder")
        v = QVBoxLayout(g)
        info = QLabel("Proxies, analysis data and temporary files are stored here and reused across projects. "
                      "Pick a fast drive with plenty of free space.")
        info.setWordWrap(True)
        v.addWidget(info)
        self.ed_path = QLineEdit(readOnly=True)
        row = QHBoxLayout()
        row.addWidget(self.ed_path, 1)
        b_change = QPushButton("Change…")
        b_change.clicked.connect(self._choose)
        self.b_default = QPushButton("Use default")
        self.b_default.clicked.connect(lambda: self._change_folder("") and self.refresh())
        b_open = QPushButton("Open")
        b_open.clicked.connect(lambda: open_path(str(cache_root())))
        for b in (b_change, self.b_default, b_open):
            row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)

        g = QGroupBox("Clear cache")
        v = QVBoxLayout(g)
        self.checks: dict[str, QCheckBox] = {}
        for kind in CACHE_KINDS:
            cb = QCheckBox()
            cb.setChecked(True)
            self.checks[kind] = cb
            v.addWidget(cb)
        hint = QLabel("Cleared proxies and analysis are recreated when needed. Imported LUTs are not affected.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa4b2;")
        v.addWidget(hint)
        row = QHBoxLayout()
        self.lbl_total = QLabel()
        row.addWidget(self.lbl_total, 1)
        b_clear = QPushButton("Clear selected")
        b_clear.clicked.connect(self._clear_selected)
        row.addWidget(b_clear)
        v.addLayout(row)
        lay.addWidget(g)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.refresh()

    def refresh(self):
        root = cache_root()
        self.ed_path.setText(str(root))
        self.b_default.setEnabled(root != default_cache_root())
        total = 0
        for kind, cb in self.checks.items():
            size = folder_size(root / kind)
            total += size
            cb.setText(f"{CACHE_KINDS[kind]}  —  {fmt_bytes(size)}")
        self.lbl_total.setText(f"Total: {fmt_bytes(total)}")

    def _choose(self):
        path = QFileDialog.getExistingDirectory(self, "Choose cache folder", str(cache_root()))
        if path and self._change_folder(os.path.normpath(path)):
            self.refresh()

    def _clear_selected(self):
        kinds = [k for k, cb in self.checks.items() if cb.isChecked()]
        if not kinds:
            return
        names = ", ".join(CACHE_KINDS[k].split(" (")[0].lower() for k in kinds)
        if QMessageBox.question(self, "Clear cache", f"Delete cached {names}?") != QMessageBox.Yes:
            return
        self._clear(kinds)
        self.refresh()


class OfficialLutDialog(QDialog):
    """Which manufacturer LUT converts each camera log profile (instead of the built-in curve).

    mapping: {log profile: library LUT path}; import_files(paths) -> imported library paths.
    """

    def __init__(self, mapping: dict, auto: bool, import_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Official camera LUTs")
        self.setMinimumWidth(760)
        self.import_files = import_files
        lay = QVBoxLayout(self)
        intro = QLabel("Camera makers publish their own log → Rec.709 conversion LUTs. Download them from the "
                       "official pages below, then import the .cube or .zip files: they are recognised by name and "
                       "used instead of KangEdit's built-in conversion for that log profile. DJI's LUTs (every "
                       "camera on dji.com/lut) download straight from DJI with <b>Download</b>.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.combos: dict[str, QComboBox] = {}
        for row, e in enumerate(lutlib.OFFICIAL_LUTS):
            name = QLabel(f"<b>{e.maker}</b>  {e.label}")
            if e.hint:
                name.setToolTip(e.hint)
            cb = QComboBox()
            cb.setToolTip(e.hint)
            self.combos[e.profile] = cb
            dji = [d for d in lutlib.DJI_LUTS if d.profile == e.profile]
            btn = QPushButton("Download…" if dji else "Download page…" if e.url else "Where?")
            if dji:
                btn.setToolTip(f"Download {e.maker}'s {e.label} LUT for your camera from dji.com")
                btn.setMenu(self._dji_menu(btn, e, dji))
            elif e.url:
                btn.setToolTip(e.url)
                btn.clicked.connect(lambda _=False, u=e.url: QDesktopServices.openUrl(QUrl(u)))
            else:
                btn.clicked.connect(lambda _=False, h=e.hint: QMessageBox.information(self, "Official LUT", h))
            grid.addWidget(name, row, 0)
            grid.addWidget(cb, row, 1)
            grid.addWidget(btn, row, 2)
        lay.addLayout(grid)
        row = QHBoxLayout()
        b_imp = QPushButton("Import LUT files (.cube / .zip)…")
        b_imp.clicked.connect(self._import)
        row.addWidget(b_imp)
        self.resolve_dir = next((d for d in lutlib.RESOLVE_LUT_DIRS if os.path.isdir(d)), None)
        b_res = QPushButton("Import from DaVinci Resolve")
        b_res.setToolTip("Import the LUTs DaVinci Resolve installs (including several cameras' official ones)"
                         + (f"\n{self.resolve_dir}" if self.resolve_dir else "\nDaVinci Resolve's LUT folder was not found"))
        b_res.setEnabled(self.resolve_dir is not None)
        b_res.clicked.connect(lambda: self._import_paths([self.resolve_dir]))
        row.addWidget(b_res)
        row.addStretch(1)
        lay.addLayout(row)
        self.chk_auto = QCheckBox("Use these LUTs automatically when a camera's log profile is detected")
        self.chk_auto.setChecked(auto)
        self.chk_apply = QCheckBox("Also switch footage already in the project to these LUTs")
        self.chk_apply.setChecked(True)
        lay.addWidget(self.chk_auto)
        lay.addWidget(self.chk_apply)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._fill(mapping)

    def _dji_menu(self, parent, entry, items) -> QMenu:
        m = QMenu(parent)
        m.setToolTipsVisible(True)
        for d in items:
            a = m.addAction(d.product if d.file else f"{d.product}  (download page)")
            a.setToolTip(d.file or d.page_url)
            a.triggered.connect(lambda _=False, d=d: self._download_dji(entry, d))
        m.addSeparator()
        a = m.addAction("All DJI LUTs on dji.com/lut…")
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(entry.url)))
        return m

    def _download_dji(self, entry, item):
        """Download one DJI camera's LUT into the library and select it for its log profile."""
        if not item.file:
            QDesktopServices.openUrl(QUrl(item.page_url))
            return
        result: dict = {}

        def work():
            try:
                result["paths"] = lutlib.download_official(item.file)
            except Exception as e:  # noqa: BLE001
                result["error"] = e

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            t = threading.Thread(target=work, daemon=True)
            t.start()
            while t.is_alive():
                QApplication.processEvents()
                t.join(0.03)
        finally:
            QApplication.restoreOverrideCursor()
        if "error" in result:
            r = QMessageBox.warning(self, "Official LUT", f"Could not download the {item.product} LUT from DJI:\n"
                                    f"{result['error']}\n\nOpen its download page instead?",
                                    QMessageBox.Yes | QMessageBox.No)
            if r == QMessageBox.Yes:
                QDesktopServices.openUrl(QUrl(item.page_url))
            return
        paths = result.get("paths") or []
        mapping = self.mapping()
        best = lutlib.official_matches(entry, [(lutlib.lut_name(p), p) for p in paths])
        if best or paths:
            mapping[entry.profile] = (best or paths)[0]
        self._fill(mapping)

    def _fill(self, mapping: dict):
        luts = lutlib.library()
        for e in lutlib.OFFICIAL_LUTS:
            cb = self.combos[e.profile]
            current = mapping.get(e.profile, cb.currentData() or "")
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("Best match in the LUT library" if e.base else "Built-in conversion", "")
            matches = lutlib.official_matches(e, luts)
            for path in matches:
                cb.addItem(f"★ {lutlib.lut_name(path)}", path)
            others = [(n, p) for n, p in luts if p not in matches]
            if others:
                cb.insertSeparator(cb.count())
                for n, path in others:
                    cb.addItem(n, path)
            if not current and matches:
                current = matches[0]   # pick the recognised official file by default
            cb.setCurrentIndex(max(0, cb.findData(current)))
            cb.blockSignals(False)

    def _import(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import official LUTs", "",
                                                "LUT or archive (*.cube *.zip);;All files (*)")
        if paths:
            self._import_paths(paths)

    def _import_paths(self, paths: list[str]):
        if self.import_files(paths):
            self._fill(self.mapping())

    def mapping(self) -> dict:
        return {k: cb.currentData() for k, cb in self.combos.items() if cb.currentData()}


class TranscriptDialog(QDialog):
    """The transcripts of the footage merged into one text in recording order, to read, copy or save."""

    def __init__(self, project: Project, ids: Optional[list[str]] = None, parent=None,
                 transcribe=None):
        super().__init__(parent)
        self.setWindowTitle("Full transcript")
        self.resize(760, 640)
        self.project = project
        self.ids = ids
        self._transcribe = transcribe        # called with the ids of the files without a transcript
        lay = QVBoxLayout(self)
        self.lbl = QLabel()
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self.chk_headers = QCheckBox("File names and recording times")
        self.chk_headers.setChecked(True)
        self.chk_headers.toggled.connect(self.refresh)
        lay.addWidget(self.chk_headers)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        lay.addWidget(self.text, 1)
        row = QHBoxLayout()
        self.b_rest = QPushButton("Transcribe the rest")
        self.b_rest.setToolTip("Transcribe the files without a transcript (in the background); open this again "
                               "when it's done")
        self.b_rest.clicked.connect(self._transcribe_rest)
        b_copy = QPushButton("Copy all")
        b_copy.setObjectName("primary")
        b_copy.clicked.connect(self.copy_all)
        b_save = QPushButton("Save as .txt…")
        b_save.clicked.connect(self._save)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        row.addWidget(self.b_rest)
        row.addStretch(1)
        row.addWidget(b_copy)
        row.addWidget(b_save)
        row.addWidget(b_close)
        lay.addLayout(row)
        self.refresh()

    def footage(self) -> list:
        ids = self.ids if self.ids is not None else list(self.project.footage)
        return [self.project.footage[i] for i in ids if i in self.project.footage
                and not self.project.footage[i].is_image]

    def missing(self) -> list[str]:
        return [f.id for f in self.footage() if f.transcript is None and f.has_audio]

    def refresh(self):
        foot = self.footage()
        done = sum(1 for f in foot if f.transcript)
        rest = self.missing()
        self.text.setPlainText(merged_transcript(foot, self.chk_headers.isChecked()))
        msg = f"{done} of {len(foot)} file(s) transcribed, in recording order."
        if rest:
            msg += f" {len(rest)} file(s) with sound have no transcript yet."
        self.lbl.setText(msg)
        self.b_rest.setVisible(bool(rest) and self._transcribe is not None)

    def copy_all(self):
        QApplication.clipboard().setText(self.text.toPlainText())
        self.lbl.setText(self.lbl.text().split(" Copied")[0] + " Copied to the clipboard.")

    def _transcribe_rest(self):
        if self._transcribe is not None:
            self._transcribe(self.missing())
            self.b_rest.setEnabled(False)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript", "transcript.txt", "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.text.toPlainText() + "\n")

