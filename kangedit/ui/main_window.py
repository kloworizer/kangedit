"""Main window: wires media bin, viewer, timeline, inspector, captions and background tasks."""

from __future__ import annotations

import copy
import itertools
import logging
import os
import shutil
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QDockWidget,
                               QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSplitter,
                               QTabWidget, QTextEdit, QToolButton, QWidget)

from .. import APP_NAME, autosave, camera, version_string
from .. import lut as lutlib
from .. import keying, smart, spherical
from ..analysis import (Analysis, analysis_path_for, analyze_many, load_or_analyze, load_plain, make_smart,
                        needs_smart, quick_look, quick_path_for)
from ..autoedit import (HOOK_SUGGESTIONS, AutoEditSettings, apply_edit_style, hook_plan, hook_suggestions,
                        natural_length, score_samples, select)
from .. import images, multicam, music, reframe, scriptedit, sfx, updater
from ..sync import SyncResult, sync_files, sync_to_longest
from ..bookends import find_bookends, story_order
from .. import bins as binlib
from .. import health, motion, polish, scenes, stabilize, voice
from ..captions import (Caption, build_captions, cached_transcript, delete_captions, edit_caption, listen,
                        load_model, model_note, project_layout, resolve_model, shift_captions, shift_group,
                        transcribe, transcribed_with, write_srt)
from ..color import Grade, auto_correct, lut_profile, missing_lut, profile_label
from ..decoder import DecoderPool, FrameDecoder
from ..config import (ASPECT_RATIOS, CACHE_KINDS, FRAME_RATES, RESOLUTIONS, VIDEO_EXTENSIONS, cache_dir, cache_root,
                      frame_size,
                      clear_cache, default_cache_root, fmt_bytes, folder_size, legacy_cache_root, lut_dir,
                      open_path, set_cache_root, settings, sfx_dir)
from ..export import export, render_audio
from ..ffmpeg import Cancelled, probe
from ..models import Footage, OverlayClip, Project, SoundClip, TextClip, TimelineClip, new_id
from ..proxy import generate_proxy, proxy_path_for
from ..speed import MAX_SPEED, MIN_SPEED, RAMPS, SOUND_MODES, sound_mode
from ..speed import clamp as clamp_speed
from ..speed import label as speed_label
from ..tasks import TaskQueue
from ..titles import ANIM_IN, ANIM_OUT, TEXT_PRESETS, anim_lengths, make_text
from .captions_panel import CaptionsPanel
from .dialogs import (FIT_MODES, AutoEditDialog, CacheDialog, CaptionStyleDialog, ExportDialog, OfficialLutDialog,
                      ProxyDialog, TranscriptDialog)
from .edit_transcript import EditTranscript
from .flow import FlowBar
from .inspector import Inspector, wrap_angle
from .layer_panel import PLACEMENTS, LayerPanel
from .media_bin import MediaBin
from .sfx_panel import SoundPanel
from .text_panel import TextPanel
from .theme import compact_combos, fmt_time
from .timeline import MAX_LAYERS, OV_Y, TimelinePanel
from .tour import TourOverlay, TourStep
from .tray import Tray
from .platform_overlay import PLATFORM_OVERLAYS
from .viewer import ADJUST_MODES, Viewer

log = logging.getLogger(__name__)

AUDIO_SLOW_MS = 2500       # a preview mix taking longer than this is shown in the status bar
REFINE_MIN_WIDTH = 300     # the Refine panel can be dragged this narrow, or to its tabs' width when wider
FACE_SAMPLES = 9           # frames of a clip searched for its main face (Zoom to face)
PROJECT_FILTER = f"{APP_NAME} project (*{Project.FILE_EXT})"


class RefineTabs(QTabWidget):
    """The Refine panel's tabs. Panels that don't scroll themselves get a scroll area, so a short window can
    still reach the bottom of every tab instead of the tallest tab setting the window's minimum height."""

    def __init__(self):
        super().__init__()
        self._pages: dict[QWidget, QWidget] = {}
        # Every tab stays in sight (no scroll arrows hiding some): the panel is never narrower than its tabs,
        # which are kept compact for it.
        self.setUsesScrollButtons(False)
        self.tabBar().setElideMode(Qt.ElideNone)
        self.setStyleSheet("QTabBar::tab { padding: 6px 7px; }")

    def tabs_width(self) -> int:
        """The width every tab needs (the panel's narrowest)."""
        return self.tabBar().sizeHint().width() + 6

    def add(self, panel: QWidget, label: str) -> None:
        page = panel
        if not isinstance(panel, QScrollArea):
            page = QScrollArea()
            page.setWidget(panel)
            page.setWidgetResizable(True)
            page.setFrameShape(QFrame.NoFrame)
        # Narrower than a tab's content (the panel dragged small), the tab scrolls sideways instead of
        # holding the panel wide.
        page.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._pages[panel] = page
        self.addTab(page, label)

    def setCurrentWidget(self, panel: QWidget) -> None:   # noqa: N802 (Qt name) — accepts the panel itself
        super().setCurrentWidget(self._pages.get(panel, panel))

    def shows(self, panel: QWidget) -> bool:
        """The tab of this panel is the one showing."""
        return self.currentWidget() is self._pages.get(panel, panel)


class MainWindow(QMainWindow):
    audio_ready = Signal(int, str)
    audio_failed = Signal(int, str)        # the preview sound couldn't be made: the reason
    update_checked = Signal(object, str, bool)   # updater.Release or None, error, asked by the user
    analysis_ready = Signal(str, object)   # footage id, Analysis (from worker threads)
    refine_status = Signal(str)            # the full analysis running in the background (from its thread)

    def __init__(self):
        super().__init__()
        self.project = Project()
        self.analyses: dict[str, Analysis] = {}
        self._stale_thumbs: set[str] = set()
        self._analysis_refresh = QTimer(self, singleShot=True, interval=500)
        self._analysis_refresh.timeout.connect(self._refresh_after_analysis)
        self._refine_lock = threading.Lock()
        self._refine_queue: list = []
        self._refine_cancel = threading.Event()
        self._refine_thread: Optional[threading.Thread] = None
        self.captions: list[Caption] = []
        self.qs = settings()
        lutlib.set_official_choices(self._official_luts())
        self._init_cache_root()
        self.tasks = TaskQueue()
        self.last_auto = self._load_auto_settings()
        self._auto_ids: list[str] = []     # footage the last auto edit used (for regenerate)
        self._tried_hooks: list[TimelineClip] = []   # hooks "New hook" already showed (not offered again)
        self._tried_alts: dict[str, list[TimelineClip]] = {}   # clip id: moments Other moment showed
        self._auto_history: list[list[TimelineClip]] = []      # recent auto edits (new ones pick other moments)
        self._progress_text = ""
        self._last_ckpt: tuple = ("", 0.0)
        self._audio_version = 0
        self._audio_cancel = threading.Event()

        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)

        # --- widgets
        self.viewer = Viewer(self.project)
        self.timeline = TimelinePanel(self.project)
        self.tl = self.timeline.view
        self.bin = MediaBin(self.project)
        self.inspector = Inspector(self.project)
        self.caps_panel = CaptionsPanel()
        self.sfx_panel = SoundPanel()
        self.text_panel = TextPanel(self.project)
        self.layer_panel = LayerPanel(self.project)
        self.edit_text = EditTranscript(self.project)
        self._exported = False
        self._tour: Optional[TourOverlay] = None

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.viewer)
        split.addWidget(self.timeline)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([560, 280])
        self.setCentralWidget(split)
        self.split = split

        d_bin = QDockWidget("Media", self)
        d_bin.setObjectName("media")
        d_bin.setWidget(self.bin)
        d_bin.setFeatures(QDockWidget.NoDockWidgetFeatures)   # fixed in place; the border still resizes it
        self.addDockWidget(Qt.LeftDockWidgetArea, d_bin)
        tabs = RefineTabs()
        tabs.add(self.inspector, "🎨 Clip")
        tabs.add(self.layer_panel, "🧩 Layer")
        tabs.add(self.caps_panel, "💬 Captions")
        tabs.add(self.text_panel, "🔤 Text")
        tabs.add(self.sfx_panel, "🔊 Sounds")
        tabs.add(self.edit_text, "🗒 Transcript")
        tabs.setTabToolTip(0, "Colour, log conversion, reframe, rotation, speed, volume and transition of the selected clip "
                              "(on the main track or a video layer)")
        tabs.setTabToolTip(1, "Video layers: position, size, crop, rotation, opacity, fades and volume of the clip "
                              "selected on a layer above the main track (picture in picture, logos, images, "
                              "multicam angles), and how to add one")
        tabs.setTabToolTip(2, "Generate, edit, style and delete captions")
        tabs.setTabToolTip(3, "Animated titles and text")
        tabs.setTabToolTip(4, "Sound effects, music and auto ducking")
        tabs.setTabToolTip(5, "What is said in the edit, clip by clip: click to go there, drag to reorder the clips")
        tabs.currentChanged.connect(lambda _i: self._refresh_edit_text())
        self.right_tabs = tabs
        for panel in (self.inspector, self.layer_panel, self.caps_panel, self.text_panel, self.sfx_panel):
            compact_combos(panel)
        compact_combos(self.viewer, 6)          # its controls row sets the window's narrowest
        d_right = QDockWidget("3. Refine", self)
        d_right.setObjectName("clip")
        d_right.setWidget(tabs)
        d_right.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, d_right)
        # It starts wide enough for the widest tab (plus its scroll bar), and can be dragged narrower, but never
        # so narrow that a tab is hidden: the pages then scroll sideways. The width chosen is remembered.
        need = max(p.minimumSizeHint().width() for p in (self.inspector.widget(), self.text_panel.widget(),
                                                          self.layer_panel.widget(), self.caps_panel,
                                                          self.sfx_panel)) + 24
        d_right.setMinimumWidth(max(REFINE_MIN_WIDTH, tabs.tabs_width()))
        self._right_width = max(370, need, d_right.minimumWidth())
        saved = self.qs.value("refine_width", 0, type=int)
        self.resizeDocks([d_bin, d_right], [430, saved if saved >= d_right.minimumWidth() else self._right_width],
                         Qt.Horizontal)
        self.d_bin, self.d_right = d_bin, d_right

        self._build_actions()
        self._build_toolbar()
        self._build_statusbar()
        self._connect()
        self._sync_project_controls()
        self.tray: Optional[Tray] = None
        if Tray.available():
            self.tray = Tray(self, self.tasks)
            self.tray.show()

        self._slip_shown = None          # what the Slip transcript shows (see _update_slip_text)
        self._measuring: set = set()     # (footage id, from, to) whose shake is being measured (stabilize.py)
        self._audio_done = True          # the latest preview mix is made (or failed)
        self._audio_said = False         # the status bar says a mix is being made
        self._audio_timer = QTimer(self, singleShot=True, interval=700)
        self._audio_timer.timeout.connect(self._render_audio)
        QTimer.singleShot(800, self._offer_legacy_cache)
        self._autosave_timer = QTimer(self, interval=autosave.AUTOSAVE_SECONDS * 1000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()
        QTimer.singleShot(600, self._offer_recovery)
        QTimer.singleShot(900, self._first_run_tour)
        heights = self.qs.value("timeline_heights", None)
        if isinstance(heights, dict):
            self.tl.set_heights(heights)
            self.timeline._sync_heights()
        self._updater = None
        if self.a_auto_update.isChecked() and not os.environ.get("KANGEDIT_SETTINGS_DIR"):
            QTimer.singleShot(4000, lambda: self.check_updates(manual=False))

    # ================================================================== construction
    def _act(self, text, slot, shortcut=None, tip=None) -> QAction:
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        return a

    def _build_actions(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self._act("New project", self.new_project, "Ctrl+N"))
        m.addAction(self._act("Open project…", self.open_project, "Ctrl+O"))
        self.recent_menu = m.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self.recent_menu.setEnabled(bool(self._recent_projects()))
        m.addAction(self._act("Save project", self.save_project, "Ctrl+S"))
        m.addAction(self._act("Save project as…", lambda: self.save_project(True), "Ctrl+Shift+S"))
        self.a_autosave = QAction("Auto-save every minute", self, checkable=True)
        self.a_autosave.setToolTip("Save the project by itself while you work, keeping rolling backups of "
                                   "earlier versions (untitled projects get a recovery copy)")
        self.a_autosave.setChecked(self.qs.value("autosave", True, type=bool))
        self.a_autosave.toggled.connect(lambda v: self.qs.setValue("autosave", v))
        m.addAction(self.a_autosave)
        m.addAction(self._act("Restore a backup…", self.restore_backup, None,
                              "Open an earlier version of this project from its automatic backups"))
        m.addSeparator()
        self.a_import = self._act("Import footage && images…", self.import_dialog, "Ctrl+I")
        m.addAction(self.a_import)
        self.a_export = self._act("Export video…", self.export_dialog, "Ctrl+E")
        m.addAction(self.a_export)
        m.addSeparator()
        m.addAction(self._act("Quit", self.close, "Ctrl+Q"))

        m = mb.addMenu("&Edit")
        m.addAction(self._act("Undo", self.undo, "Ctrl+Z"))
        m.addAction(self._act("Redo", self.redo, "Ctrl+Shift+Z"))
        m.addSeparator()
        m.addAction(self._act("Split at playhead", self.split_at_playhead, "S"))
        m.addAction(self._act("Select all", self._select_all, "Ctrl+A"))
        m.addAction(self._act("Delete selected clips / sound", self._delete_selected, "Del"))
        m.addAction(self._act("Duplicate clip", lambda: self.clip_action("duplicate", self.tl.selected), "Ctrl+D"))
        m.addAction(self._act("Link / unlink camera angles", lambda: self.toggle_link(), "Ctrl+L",
                              "Multicam: unlink the selected clip's camera angles to edit them on their own, or "
                              "link them again (back in sync)"))
        m.addAction(self._act("Other moment", lambda: self.clip_action("alt_moment", self._moment_clip()), "R",
                              "Swap the clip for the best moment close by; its length follows the new moment"))
        m.addSeparator()
        m.addAction(self._act("Add title at playhead", lambda: self.add_text("title"), "T",
                              "Add an animated title where the playhead is (more templates in the Text tab)"))

        m = mb.addMenu("&Tools")
        self.a_auto = self._act("Auto Edit…", self.auto_edit_dialog, "Ctrl+Shift+A",
                                "Pick the best moments from your footage automatically")
        m.addAction(self.a_auto)
        m.addAction(self._act("Auto Edit from script…", self.script_edit_dialog, None,
                              "Find each line of a script in the footage's speech and lay the clips out in order"))
        m.addAction(self._act("Regenerate auto edit shorter (−25%)", lambda: self.regenerate_auto_edit(0.75),
                              "Ctrl+Shift+-", "Rebuild the automatic edit about a quarter shorter than the timeline"))
        m.addAction(self._act("Regenerate auto edit longer (+25%)", lambda: self.regenerate_auto_edit(1.25),
                              "Ctrl+Shift+=", "Rebuild the automatic edit about a quarter longer than the timeline"))
        m.addAction(self._act("Regenerate auto edit at length…", lambda: self.regenerate_auto_edit(None), None,
                              "Rebuild the automatic edit with the same settings at a length you choose"))
        m.addAction(self._act("Mark in (Source)", lambda: self._source_mark("in"), "I",
                              "In the viewer's Source mode: the part to use starts at the playhead"))
        m.addAction(self._act("Mark out (Source)", lambda: self._source_mark("out"), "O",
                              "In the viewer's Source mode: the part to use ends at the playhead"))
        m.addAction(self._act("Choose the hook…", lambda: self.new_hook(), "H",
                              "Choose the opening hook from the strongest moments of the whole video, as thumbnails "
                              "(and how many clips it has)"))
        fin = m.addMenu("Finish the edit")
        for key, label, tip in (
                ("beats", "Cut on the music's beat", "Move each cut to the nearest beat of the music on the timeline"),
                ("sentences", "Cut on whole sentences",
                 "Make every clip with speech start and end on whole sentences of the transcript (transcribes the "
                 "footage first when needed)"),
                ("pauses", "Cut between words", "Move cuts that fall inside a word to the pause next to it"),
                ("fillers", "Remove fillers, long pauses and false starts",
                 "Take \"um / eh / anu\", long silences and words said twice out of the clips (transcribes the "
                 "footage first when needed)"),
                ("variety", "Vary the shot sizes", "Swap a shot where three of the same size follow each other"),
                ("loudness", "Even out the loudness", "Bring every clip's sound to about the same loudness")):
            fin.addAction(self._act(label, lambda _=False, k=key: self.finish_timeline(k), None, tip))
        m.addAction(self._act("Make a highlight reel (9:16)…", self.highlight_reel, None,
                              "Turn the edit into a 30-60 s vertical reel of its best moments, framed on the faces "
                              "(Undo brings the edit back)"))
        m.addAction(self._act("Make a thumbnail…", lambda: self.thumbnail_dialog(), None,
                              "A thumbnail from the edit's best stills (or the frame at the playhead), framed on the "
                              "face, with big text, saved as a JPG"))
        m.addAction(self._act("YouTube chapters…", lambda: self.chapters_dialog(), None,
                              "Chapter timestamps for the video's description, from the edit's scenes and what is "
                              "said; title cards for them"))
        self.a_proxies = self._act("Generate proxies…", self.proxy_dialog, None,
                                   "Create small copies of the footage for faster editing")
        m.addAction(self.a_proxies)
        self.a_auto_proxy = QAction("Generate proxies automatically on import", self, checkable=True)
        self.a_auto_proxy.setChecked(self.qs.value("auto_proxy", False, type=bool))
        self.a_auto_proxy.toggled.connect(lambda v: self.qs.setValue("auto_proxy", v))
        m.addAction(self.a_auto_proxy)
        self.a_gpu_analysis = QAction("Use GPU to speed up analysis", self, checkable=True)
        self.a_gpu_analysis.setToolTip("Decode footage on the graphics card (NVIDIA, AMD or Intel) while the CPU "
                                       "analyzes other clips")
        self.a_gpu_analysis.setChecked(self.qs.value("gpu_analysis", True, type=bool))
        self.a_gpu_analysis.toggled.connect(lambda v: self.qs.setValue("gpu_analysis", v))
        m.addAction(self.a_gpu_analysis)
        self.a_smart = QAction("Smarter analysis (AI: what shots show, what is heard)", self, checkable=True)
        self.a_smart.setToolTip("Two small neural networks describe what each shot shows (repeated views are found "
                                "across files) and what is heard (speech, laughter, cheering, music, wind…), "
                                "for better moments, scenic shots and hooks.\nTurn off on a slow computer: the "
                                "lighter analysis is used.")
        self.a_smart.setChecked(self.qs.value("smart_analysis", True, type=bool) and smart.available())
        self.a_smart.setEnabled(smart.available())
        smart.set_enabled(self.a_smart.isChecked())
        self.a_smart.toggled.connect(self._set_smart)
        m.addAction(self.a_smart)
        self.a_auto_log = QAction("Detect camera log profile on import", self, checkable=True)
        self.a_auto_log.setToolTip("Identify the camera of imported footage and choose its log → Rec.709 "
                                   "conversion automatically")
        self.a_auto_log.setChecked(self.qs.value("auto_log", True, type=bool))
        self.a_auto_log.toggled.connect(lambda v: self.qs.setValue("auto_log", v))
        m.addAction(self.a_auto_log)
        m.addAction(self._act("Detect log profile for all footage", lambda: self.detect_log(list(self.project.footage)),
                              None, "Identify the camera of every imported file and choose its log → Rec.709 conversion"))
        sub = m.addMenu("Multicam layout")
        sub.setToolTipsVisible(True)
        for key, label in multicam.TEMPLATES.items():
            a = sub.addAction(label, lambda k=key: self.multicam_layout(k))
            a.setToolTip(multicam.TEMPLATE_HINTS[key])
        m.addSeparator()
        self.a_color_all = self._act("Auto color all clips", lambda: self.inspector_request("auto_color_all"))
        m.addAction(self.a_color_all)
        m.addAction(self._act("Import LUT (.cube)…", lambda: self.import_lut("library"), None,
                              "Add log-conversion or color-grading LUTs to the LUT library"))
        m.addAction(self._act("Official camera LUTs…", self.official_luts_dialog, None,
                              "Use the camera makers' own log → Rec.709 LUTs (DJI D-Log M, Sony, Panasonic…)"))
        m.addAction(self._act("Open LUT folder", lambda: open_path(str(lut_dir()))))
        self.a_captions = self._act("Generate captions", self.generate_captions, None,
                                    "Transcribe speech and create captions")
        m.addAction(self.a_captions)
        m.addAction(self._act("Caption style…", self.caption_style_dialog))
        m.addAction(self._act("Full transcript…", lambda: self.transcript_dialog(), None,
                              "All the footage's transcripts merged in recording order, to read, copy or save"))
        self.a_ask_transcribe = QAction("Offer to transcribe after importing", self, checkable=True)
        self.a_ask_transcribe.setChecked(self.qs.value("ask_transcribe_on_import", True, type=bool))
        self.a_ask_transcribe.toggled.connect(lambda v: self.qs.setValue("ask_transcribe_on_import", v))
        m.addAction(self.a_ask_transcribe)
        m.addSeparator()
        m.addAction(self._act("Re-analyze all footage", lambda: self.queue_analysis(list(self.project.footage), True)))
        m.addAction(self._act("Cache folder && clearing…", self.cache_dialog, None,
                              "Choose where proxies and analysis are stored, and clear them"))

        m = mb.addMenu("&View")
        m.addAction(self._act("Fit timeline", self.tl.fit, "Shift+Z", "Zoom so the whole edit fits the timeline"))
        m.addAction(self._act("Zoom timeline in", lambda: self.tl.zoom_by(1.5), "Ctrl+="))
        m.addAction(self._act("Zoom timeline out", lambda: self.tl.zoom_by(1 / 1.5), "Ctrl+-"))
        m.addAction(self._act("Fit timeline lanes to the panel height", self._fit_lane_heights, None,
                              "Make the video, layer and sound lanes fill the timeline panel"))
        m.addAction(self._act("Add a video layer", lambda: self._timeline_action("layer_add", ""), None,
                              "One more layer above the main track for picture in picture, logos and images"))
        m.addAction(self._act("Reset layout", self.reset_layout, "Ctrl+Shift+R",
                              "Put the panels, the viewer / timeline split and the timeline lanes back to their "
                              "usual sizes, and fit the timeline"))
        m.addSeparator()
        m.addAction(self._act("Full screen preview", self.viewer.toggle_fullscreen, "F11",
                              "Play the timeline on the whole screen (Esc to leave)"))
        sub = m.addMenu("Color in viewer")
        sub.setToolTipsVisible(True)
        group = QActionGroup(self)
        self.a_adjust: dict[str, QAction] = {}
        for key, label in ADJUST_MODES:
            a = QAction(label.split(": ", 1)[1].capitalize(), self, checkable=True)
            a.setToolTip("Log conversion, LUTs and grading in the viewer (export always includes them)")
            a.triggered.connect(lambda _=False, k=key: self.viewer.set_adjust_mode(k))
            group.addAction(a)
            sub.addAction(a)
            self.a_adjust[key] = a
        sub.addSeparator()
        sub.addAction(self._act("Toggle color on / off", self._toggle_adjustments, "Ctrl+Shift+B",
                                "Show the picture as recorded, without log conversion, LUTs or grading"))
        self.viewer.set_adjust_mode(self.qs.value("viewer_color", "on", type=str))
        self.a_adjust[self.viewer.adjust_mode].setChecked(True)
        self.viewer.adjust_mode_changed.connect(self._adjust_mode_changed)
        a = QAction("Rule-of-thirds guides", self, checkable=True)
        a.toggled.connect(lambda v: (setattr(self.viewer.canvas, "show_guides", v), self.viewer.canvas.update()))
        m.addAction(a)
        self.a_guides = a
        sub = m.addMenu("App overlay (Reels / TikTok)")
        sub.setToolTipsVisible(True)
        group = QActionGroup(self)
        self.a_platform: dict[str, QAction] = {}
        for key, label in PLATFORM_OVERLAYS.items():
            a = QAction(label, self, checkable=True)
            a.setToolTip("Show where the app's buttons and text cover a vertical video (guide only, not exported)")
            a.triggered.connect(lambda _=False, k=key: self._set_platform_overlay(k))
            group.addAction(a)
            sub.addAction(a)
            self.a_platform[key] = a
        self._set_platform_overlay(self.qs.value("platform_overlay", "", type=str))

        m = mb.addMenu("&Help")
        self.a_tour = self._act("Guided tour", lambda: self.show_tour(), "Shift+F1",
                                "A short walk through the app: import, auto edit, refine and export")
        m.addAction(self.a_tour)
        m.addAction(self._act("Keyboard shortcuts", self.show_shortcuts, "F1"))
        m.addSeparator()
        self.a_check_updates = self._act("Check for updates…", lambda: self.check_updates(manual=True), None,
                                         "Look for a newer KangEdit on GitHub and install it")
        m.addAction(self.a_check_updates)
        self.a_auto_update = QAction("Check for updates at startup", self, checkable=True)
        self.a_auto_update.setChecked(self.qs.value("update_check", True, type=bool))
        self.a_auto_update.toggled.connect(lambda v: self.qs.setValue("update_check", v))
        m.addAction(self.a_auto_update)
        m.addSeparator()
        m.addAction(self._act("Copy system info", self.copy_system_info, tip=(
            "Copy your app version, system, graphics card support and folders, to paste into a bug report")))
        m.addAction(self._act(f"About {APP_NAME}", self.show_about))

        for key, slot in (("Space", lambda: self.viewer.toggle_play()),
                          ("Left", lambda: self.viewer.step(-1)), ("Right", lambda: self.viewer.step(1)),
                          ("Shift+Left", lambda: self.viewer.step(-10)), ("Shift+Right", lambda: self.viewer.step(10)),
                          ("Up", self.viewer.prev_clip), ("Down", self.viewer.next_clip),
                          ("Home", lambda: self.viewer.seek(0)),
                          ("End", lambda: self.viewer.seek(self.viewer.project.duration)),
                          ("Ctrl+Y", self.redo)):
            QShortcut(QKeySequence(key), self, activated=slot)

    def _build_toolbar(self):
        """The four steps, left to right: 1. Import › 2. Auto Edit › 3. Refine › 4. Export."""
        tb = self.addToolBar("Main")
        tb.setObjectName("main")
        tb.setMovable(False)
        # One row when the window is wide enough, more rows when it isn't: nothing hides in a » menu.
        bar = FlowBar(hspacing=4, vspacing=4)

        def arrow():
            lbl = QLabel(" › ")
            lbl.setStyleSheet("color: #5d6068; font-size: 16px;")
            return lbl

        def group(*widgets) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            for x in widgets:
                h.addWidget(x)
            return w

        self.btn_import = QPushButton("1. Import")
        self.btn_import.setToolTip("Add your video files (Ctrl+I). You can also drag them onto the Media panel.")
        self.btn_import.clicked.connect(self.import_dialog)
        bar.add(group(self.btn_import, arrow()))

        self.btn_auto = QPushButton("2. ✨ Auto Edit")
        self.btn_auto.setObjectName("primary")
        self.btn_auto.setToolTip("Find the best moments and build the edit (Short or Long format)")
        self.btn_auto.clicked.connect(self.auto_edit_dialog)
        b_short, b_long = QPushButton("−"), QPushButton("+")
        for b, factor, tip in ((b_short, 0.75, "Regenerate the auto edit about 25% shorter"),
                               (b_long, 1.25, "Regenerate the auto edit about 25% longer")):
            b.setFixedWidth(30)
            b.setToolTip(tip + " (same settings and footage)")
            b.clicked.connect(lambda _=False, k=factor: self.regenerate_auto_edit(k))
        self.btn_script = QPushButton("📜 Script")
        self.btn_script.setToolTip("Auto edit from a script: each line is found in what the footage says (by text "
                                   "similarity) and the clips are laid out in the script's order")
        self.btn_script.clicked.connect(self.script_edit_dialog)
        self.grp_auto = group(self.btn_auto, b_short, b_long, self.btn_script)
        bar.add(group(self.grp_auto, arrow()))

        refine = QLabel("3. Refine")
        refine.setStyleSheet("color: #9aa0ab;")
        btns = [refine]
        for text, tab, tip in (("🎨 Color", self.inspector, "Colour, reframe and audio of the selected clip (main track or a video layer)"),
                               ("💬 Captions", self.caps_panel, "Generate and edit captions"),
                               ("🔤 Text", self.text_panel, "Add animated titles and text"),
                               ("🔊 Sounds", self.sfx_panel, "Add sound effects")):
            b = QToolButton(text=text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, w=tab: self.show_refine(w))
            btns.append(b)
        self.grp_refine = group(*btns)
        bar.add(self.grp_refine)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.add(spacer)

        lbl = QLabel("Frame")
        lbl.setStyleSheet("color: #9aa0ab;")
        self.cb_aspect = QComboBox()
        for k in ASPECT_RATIOS:
            self.cb_aspect.addItem(k, k)
        self.cb_res = QComboBox()
        for k in RESOLUTIONS:
            self.cb_res.addItem(k, k)
        self.cb_fps = QComboBox()
        for r in FRAME_RATES:
            self.cb_fps.addItem(f"{r:g} fps", r)
        self.cb_fit = QComboBox()
        for k, v in FIT_MODES.items():
            self.cb_fit.addItem(v, k)
        for cb, tip in ((self.cb_aspect, "Shape of the finished video"), (self.cb_res, "Export resolution"),
                        (self.cb_fps, "Frame rate"), (self.cb_fit, "How footage of another shape fills the frame")):
            cb.setToolTip(tip)
            cb.currentIndexChanged.connect(self._project_controls_changed)
        self.chk_proxy = QCheckBox("Proxies")
        self.chk_proxy.setToolTip("Play back from small proxy copies when available (Tools → Generate proxies)")
        self.chk_proxy.toggled.connect(self._toggle_proxies)
        self.grp_frame = group(lbl, self.cb_aspect, self.cb_res, self.cb_fps, self.cb_fit, self.chk_proxy)
        bar.add(group(self.grp_frame, arrow()))

        self.btn_export = QPushButton("4. ⬆ Export")
        self.btn_export.setObjectName("primary")
        self.btn_export.setToolTip("Render the finished video (Ctrl+E)")
        self.btn_export.clicked.connect(self.export_dialog)
        self.btn_help = QToolButton(text=" ? ")
        self.btn_help.setToolTip("Guided tour (Shift+F1)")
        self.btn_help.clicked.connect(lambda: self.show_tour())
        bar.add(group(self.btn_export, self.btn_help))
        tb.addWidget(bar)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.lbl_status = QLabel("Ready")
        self.lbl_next = QLabel("")
        self.lbl_next.setStyleSheet("color: #7fb2ff; padding: 0 8px;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(260)
        self.progress.setVisible(False)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.tasks.cancel_all)
        sb.addWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.lbl_next)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.btn_cancel)

    def _connect(self):
        t = self.tasks
        t.progress.connect(self._on_progress)
        t.started.connect(self._status_progress)
        t.pending_changed.connect(self._on_pending)
        # "Ready" replaces progress text only, not the result message a finished task just showed.
        t.idle.connect(lambda: self.lbl_status.text() == self._progress_text and self.lbl_status.setText("Ready"))

        self.viewer.time_changed.connect(lambda tt: self.tl.set_playhead(tt, follow=self.viewer.playing))
        self.viewer.play_state.connect(self._on_play_state)

        self.tl.seek.connect(self._timeline_seek)
        self.viewer.use_source.connect(self.use_source_part)
        self.viewer.transcribe_source.connect(lambda fid: self.queue_transcription([fid]))
        self.viewer.mode_changed.connect(self._viewer_mode_changed)
        self.tl.selection_changed.connect(self._on_select)
        self.tl.edit_started.connect(self.project_checkpoint)
        self.tl.edited.connect(self._on_timeline_edited)
        self.tl.edit_finished.connect(self._on_timeline_finished)
        self.tl.slip_preview.connect(self.viewer.show_source)
        self.tl.slip_end.connect(lambda: self.viewer.show_source(None))
        self.tl.slip_preview.connect(self._slip_follow)
        self.tl.slip_end.connect(self._update_slip_text)
        self.timeline.btn_slip.toggled.connect(lambda _on: self._update_slip_text())
        self.timeline.slip_text.slip_to.connect(self.slip_clip_to)
        self.timeline.slip_text.transcribe.connect(lambda fid: self.queue_transcription([fid]))
        self.edit_text.seek.connect(self._timeline_seek)
        self.edit_text.reordered.connect(self._reorder_by_transcript)
        self.edit_text.transcribe.connect(self._transcribe_edit)
        self.viewer.time_changed.connect(lambda t: self.edit_text.show_time(t)
                                         if self.right_tabs.shows(self.edit_text) else None)
        self.tl.sounds_changed.connect(self._sounds_changed)
        self.tl.sound_dropped.connect(lambda paths, t: self.add_sounds(paths, t))
        self.sfx_panel.add_sound.connect(lambda path, name: self.add_sounds([path], None, [name]))
        self.sfx_panel.add_music.connect(self.add_music)
        self.sfx_panel.volume_changed.connect(self._selected_sound_volume)
        self.sfx_panel.fade_changed.connect(
            lambda which, v: self.tl.selected_sound and self._sound_action(f"sound_fade_{which}:{v}",
                                                                           self.tl.selected_sound))
        self.sfx_panel.fit_requested.connect(
            lambda: self.tl.selected_sound and self._sound_action("sound_fit", self.tl.selected_sound))
        for sig in (self.tl.sound_selected, self.tl.text_selected, self.tl.selection_changed):
            sig.connect(lambda *_: self._sync_sound_panel())
        self.tl.context_action.connect(self._timeline_action)
        self.tl.overlay_selected.connect(self._on_overlay_selected)
        self.tl.overlays_changed.connect(self._overlay_moved)
        self.tl.footage_dropped.connect(self._footage_dropped)
        self.tl.heights_changed.connect(lambda h: self.qs.setValue("timeline_heights", h))
        self.layer_panel.before_change.connect(self.project_checkpoint)
        self.layer_panel.changed.connect(self._layer_edited)
        self.layer_panel.request.connect(self._overlay_action)
        self.sfx_panel.duck_changed.connect(
            lambda on: self.tl.selected_sound and self._sound_action(f"sound_duck:{int(on)}", self.tl.selected_sound))
        self.sfx_panel.duck_depth_changed.connect(self._duck_depth)
        self.tl.text_selected.connect(self._on_text_selected)
        self.tl.texts_changed.connect(lambda: self._texts_changed(panel_fields=True))
        cv = self.viewer.canvas
        cv.text_pressed.connect(self.tl.select_text)
        cv.text_dragged.connect(self._text_dragged)
        cv.text_released.connect(lambda tid, moved: moved and self._texts_changed(panel_fields=True))
        cv.text_activated.connect(lambda tid: self._text_action("text_edit", tid))
        cv.context_menu.connect(self._viewer_menu)
        cv.overlay_pressed.connect(lambda oid: self.tl.select_overlay(oid) if oid != self.tl.selected_overlay else None)
        cv.overlay_dragged.connect(self._overlay_dragged)
        cv.overlay_released.connect(lambda oid, moved: moved and self._overlays_changed())
        cv.overlay_scaled.connect(self._overlay_scaled)
        self.text_panel.add_requested.connect(self.add_text)
        self.text_panel.selected.connect(lambda tid: self.tl.select_text(tid) if tid != self.tl.selected_text else None)
        self.text_panel.before_change.connect(self.project_checkpoint)
        self.text_panel.changed.connect(lambda: self._texts_changed(panel=False))
        self.text_panel.delete_requested.connect(lambda tid: self._text_action("text_delete", tid))
        self.tl.captions_shifted.connect(self._captions_shifted)
        self.tl.caption_clicked.connect(lambda i: (self.right_tabs.setCurrentWidget(self.caps_panel),
                                                   self.caps_panel.select(i)))
        self.timeline.btn_split.clicked.connect(self.split_at_playhead)
        self.timeline.fades_changed.connect(self._fades_changed)
        self.timeline.btn_delete.clicked.connect(lambda: self.clip_action("delete", self.tl.selected))
        self.timeline.btn_hook.clicked.connect(lambda: self.new_hook())
        self.timeline.btn_undo.clicked.connect(self.undo)
        self.timeline.btn_redo.clicked.connect(self.redo)
        self.timeline.btn_alt.clicked.connect(lambda: self.clip_action("alt_moment", self._moment_clip()))
        self.timeline.btn_link.clicked.connect(lambda: self.toggle_link())
        for sig in (self.tl.selection_changed, self.tl.overlay_selected):
            sig.connect(lambda *_: self._sync_link_button())

        self.bin.import_requested.connect(self.import_dialog)
        self.bin.files_dropped.connect(self.import_files)
        self.bin.action.connect(self.bin_action)
        self.bin.log_profile.connect(self.set_log_profile)

        self.bin.edit_started.connect(self.project_checkpoint)
        self.bin.edited.connect(self._bins_edited)
        self.inspector.selection = self.tl.selected_ids
        self.inspector.layer_selection = self.tl.selected_overlay_ids
        self.text_panel.playhead = lambda: self.viewer.timeline_time()
        self.inspector.before_change.connect(self._inspector_before)
        self.inspector.changed.connect(self._inspector_changed)
        self.inspector.request.connect(self.inspector_request)

        self.caps_panel.generate.connect(self.generate_captions)
        self.caps_panel.style_requested.connect(self.caption_style_dialog)
        self.caps_panel.model_changed.connect(self._set_speech_model)
        self.caps_panel.cleanup_changed.connect(self._set_speech_cleanup)
        self.caps_panel.export_srt.connect(self.save_srt)
        self.caps_panel.text_edited.connect(self._caption_edited)
        self.caps_panel.delete_requested.connect(self._delete_captions)
        self.caps_panel.seek.connect(self._timeline_seek)
        self.caps_panel.toggled.connect(self._toggle_captions)

        self.audio_ready.connect(self._on_audio_ready)
        self.audio_failed.connect(self._on_audio_failed)
        self.update_checked.connect(self._on_update_checked)
        self.analysis_ready.connect(self._on_analysis_ready)
        self.refine_status.connect(self._on_refine_status)

    # ================================================================== project state
    def _sync_project_controls(self):
        s = self.project.settings
        for cb, val in ((self.cb_aspect, s.aspect), (self.cb_res, s.resolution), (self.cb_fit, s.fit_mode)):
            cb.blockSignals(True)
            cb.setCurrentIndex(max(0, cb.findData(val)))
            cb.blockSignals(False)
        self.cb_fps.blockSignals(True)
        self.cb_fps.setCurrentIndex(min(range(len(FRAME_RATES)), key=lambda i: abs(FRAME_RATES[i] - s.fps)))
        self.cb_fps.blockSignals(False)
        self.chk_proxy.blockSignals(True)
        self.chk_proxy.setChecked(s.use_proxies)
        self.chk_proxy.blockSignals(False)
        self.caps_panel.set_model(s.whisper_model, s.speech_cleanup)
        self.caps_panel.chk_show.blockSignals(True)
        self.caps_panel.chk_show.setChecked(s.captions.enabled)
        self.caps_panel.chk_show.blockSignals(False)
        self.timeline.sync_fades()
        self.sfx_panel.set_duck_depth(s.duck_db)

    def _fades_changed(self):
        self.project.dirty = True
        self.viewer.refresh()
        self._schedule_audio()
        self._update_title()

    def _project_controls_changed(self):
        s = self.project.settings
        s.aspect = self.cb_aspect.currentData()
        s.resolution = self.cb_res.currentData()
        s.fps = float(self.cb_fps.currentData())
        s.fit_mode = self.cb_fit.currentData()
        self.project.dirty = True
        self.viewer.refresh()
        self._update_title()

    def _set_platform_overlay(self, key: str):
        if key not in PLATFORM_OVERLAYS:
            key = ""
        self.a_platform[key].setChecked(True)
        self.viewer.canvas.platform_overlay = key
        self.viewer.canvas.update()
        self.qs.setValue("platform_overlay", key)
        if key and not self.project.settings.aspect.startswith("9:16"):
            self.lbl_status.setText(f"The {PLATFORM_OVERLAYS[key]} overlay is made for vertical 9:16 video.")

    def _toggle_adjustments(self):
        self.viewer.set_adjust_mode("off" if self.viewer.adjust_mode == "on" else "on")

    def _adjust_mode_changed(self, mode: str):
        self.qs.setValue("viewer_color", mode)
        self.a_adjust[mode].setChecked(True)
        self.lbl_status.setText({"on": "Viewer shows color adjustments",
                                 "playback": "Color adjustments are hidden while playing (export unaffected)",
                                 "off": "Viewer shows the picture as recorded (export unaffected)"}[mode])

    def _toggle_proxies(self, on: bool):
        self.project.settings.use_proxies = on
        self.viewer.refresh()
        self._schedule_audio()

    def _update_title(self):
        name = os.path.basename(self.project.path) if self.project.path else "Untitled"
        self.setWindowTitle(f"{name}{' *' if self.project.dirty else ''} — {APP_NAME}")
        self.timeline.btn_undo.setEnabled(self.project.can_undo)
        self.timeline.btn_redo.setEnabled(self.project.can_redo)
        self._update_flow()

    def _update_flow(self):
        """Tick the finished steps and say what to do next."""
        p = self.project
        steps = ((self.btn_import, "1. Import", bool(p.footage)), (self.btn_auto, "2. ✨ Auto Edit", bool(p.clips)),
                 (self.btn_export, "4. ⬆ Export", self._exported))
        for b, text, done in steps:
            b.setText(text + ("  ✓" if done else ""))
        if not p.footage:
            hint = "Next: 1. Import your footage (or drag files onto Media)"
        elif not p.clips:
            hint = "Next: 2. ✨ Auto Edit builds the edit for you"
        elif not self._exported:
            hint = "Next: 3. refine (right-click anything for options), then 4. Export"
        else:
            hint = ""
        self.lbl_next.setText(hint)

    def copy_system_info(self):
        from .. import sysinfo
        QApplication.setOverrideCursor(Qt.WaitCursor)   # the GPU checks run short test encodes
        try:
            text = sysinfo.report()
        finally:
            QApplication.restoreOverrideCursor()
        QApplication.clipboard().setText(text)
        box = QMessageBox(QMessageBox.Information, "System info copied",
                          "Your system info is on the clipboard. Paste it into your bug report or message.", parent=self)
        box.setDetailedText(text)
        box.exec()

    REPO_URL = f"https://github.com/{updater.REPO}"
    CONTACT = "esha@angkasa.dev"

    def show_about(self):
        def link(href: str, text: str) -> str:   # readable on the dark theme
            return f"<a href='{href}' style='color: #7fb2ff;'>{text}</a>"

        box = QMessageBox(QMessageBox.Information, f"About {APP_NAME}",
                          f"<b>{APP_NAME} {version_string()}</b><br>Automatic best-moment video editor.<br><br>"
                          f"Source, releases and issues:<br>{link(self.REPO_URL, self.REPO_URL)}<br><br>"
                          f"Contact: {link('mailto:' + self.CONTACT, self.CONTACT)}<br><br>"
                          "License: MIT with the Commons Clause — free to use, change and share, also for paid "
                          "video work; KangEdit itself may not be sold.<br><br>"
                          "Python · PySide6 · FFmpeg · OpenCV · faster-whisper", parent=self)
        box.setTextFormat(Qt.RichText)
        box.setTextInteractionFlags(Qt.TextBrowserInteraction)   # clickable links, selectable text
        box.setIconPixmap(self.windowIcon().pixmap(64, 64))
        box.exec()

    def reset_layout(self):
        """Panels, the viewer / timeline split, lane heights, tabs and list view back to how they start."""
        self.qs.remove("refine_width")
        from .timeline import DEFAULT_HEIGHTS
        if self.viewer.is_fullscreen():
            self.viewer.toggle_fullscreen()
        if self.isFullScreen():
            self.showMaximized()
        for d in (self.d_bin, self.d_right):
            d.show()
        self.resizeDocks([self.d_bin, self.d_right], [430, self._right_width], Qt.Horizontal)
        total = max(3, sum(self.split.sizes()))
        self.split.setSizes([total * 2 // 3, total - total * 2 // 3])
        self.tl.set_heights(DEFAULT_HEIGHTS, emit=True)
        self.timeline._sync_heights()
        self.timeline.area.verticalScrollBar().setValue(0)
        self.right_tabs.setCurrentIndex(0)
        self.bin.set_view_mode("details")
        QTimer.singleShot(0, self.tl.fit)      # once the timeline has its new width
        self.lbl_status.setText("Layout reset")

    def createPopupMenu(self):   # noqa: N802 (Qt name)
        """The right-click menu on toolbars and dock titles, without the entries that would hide the panels."""
        menu = super().createPopupMenu()
        if menu is not None:
            for d in (self.d_bin, self.d_right):
                menu.removeAction(d.toggleViewAction())
            if menu.isEmpty():
                menu.deleteLater()
                return None
        return menu

    def show_refine(self, panel: QWidget):
        self.d_right.show()
        self.right_tabs.setCurrentWidget(panel)

    def project_checkpoint(self, key: str = ""):
        now = time.monotonic()
        if key and self._last_ckpt[0] == key and now - self._last_ckpt[1] < 2.0:
            self._last_ckpt = (key, now)
            return
        self._last_ckpt = (key, now)
        self.project.checkpoint()
        self._update_title()

    def refresh_all(self, captions: bool = True):
        if self.viewer.source_fid and self.viewer.source_fid not in self.project.footage:
            self.viewer.close_source([self.viewer.source_fid])     # removed (or undone) while shown on its own
        # Synced camera angles follow every change to the main track (no-op without multicam).
        if multicam.follow(self.project):
            self._schedule_audio()
        self.bin.analyses = self.analyses
        self.tl.analyses = self.analyses
        self.tl.min_gap = self.last_auto.min_gap
        self.bin.refresh()
        self.tl.refresh()
        if captions:
            self.update_captions()
        self.viewer.refresh()
        self.inspector.set_clip(self._inspected())
        if not self.timeline.slip_text.isHidden():
            self._update_slip_text(slipping=True)
        self._refresh_edit_text()
        self.layer_panel.set_clip(self.selected_overlay())
        self.viewer.canvas.selected_overlay = self.tl.selected_overlay
        self.text_panel.refresh()
        self._update_title()

    def _link_targets(self, item: str = "") -> list[OverlayClip]:
        """The camera-angle clips Link / Unlink acts on: those of a given clip or angle, else of the selection
        (a selected angle, or the selected main clips)."""
        if item:
            o = self._overlay(item)
            return [o] if o is not None and o.link else multicam.angles_of(self.project, item)
        o = self.selected_overlay()
        if o is not None:
            return [o] if o.link else []
        return [a for cid in self.tl.selected_ids() for a in multicam.angles_of(self.project, cid)]

    def _sync_link_button(self):
        targets = self._link_targets()
        b = self.timeline.btn_link
        b.setEnabled(bool(targets))
        on = bool(targets) and all(o.linked for o in targets)
        b.setChecked(on)
        b.setText("🔗 Linked" if on else "⛓ Unlinked" if targets else "🔗 Link")

    def toggle_link(self, item: str = ""):
        """Unlink the camera angles (edit them on their own) when they are all linked, else link them again
        (they are re-timed from their main clip, back in sync)."""
        targets = self._link_targets(item)
        if not targets:
            self.lbl_status.setText("Select a multicam clip or one of its camera angles to link or unlink them.")
            self._sync_link_button()
            return
        on = not all(o.linked for o in targets)
        self.project_checkpoint()
        n = multicam.set_linked(self.project, targets, on)
        self._overlays_changed()
        self._sync_link_button()
        self.lbl_status.setText(f"Linked {n} camera angle(s): back in sync with the main clip" if on else
                                f"Unlinked {n} camera angle(s): they can be moved and trimmed on their own")

    def _moment_clip(self) -> str:
        """The clip Other moment works on: the selected clip, or the main clip a selected (linked) camera angle
        follows."""
        if not self.tl.selected and self.tl.selected_overlay:
            c = self.selected_overlay()
            if c is not None and c.link and c.linked and self.project.index_of(c.link) >= 0:
                return c.link
        return self.tl.selected

    def selected_overlay(self) -> Optional[OverlayClip]:
        oid = self.tl.selected_overlay
        return next((c for c in self.project.overlays if c.id == oid), None) if oid else None

    def _inspected(self):
        """The clip the Inspector refines: the selected main-track clip or the selected clip on a video layer."""
        return self.selected_clip() or self.selected_overlay()

    def selected_clip(self) -> Optional[TimelineClip]:
        i = self.project.index_of(self.tl.selected) if self.tl.selected else -1
        return self.project.clips[i] if i >= 0 else None

    def set_project(self, project: Project):
        self.tasks.cancel_all()
        autosave.clear_recovery()      # the previous project was saved or discarded
        self.project = project
        self.analyses = {}
        self._tried_alts = {}
        self._auto_history = []
        for w in (self.viewer, self.tl, self.bin, self.inspector, self.text_panel, self.layer_panel, self.edit_text):
            w.set_project(project)
        self._exported = False
        self._sync_project_controls()
        missing = []
        self._stop_refining()
        self._fill_transcripts(list(project.footage))
        for fid, f in project.footage.items():
            if f.is_image:
                continue
            a = load_plain(f)
            if a is not None:
                self.analyses[fid] = a
                if f.log_pending:
                    self._log_from_picture(f, list(a.thumbs))
            if os.path.exists(f.path) and (a is None or a.quick or needs_smart(f, a)):
                missing.append(fid)
        if missing:
            self.queue_analysis(missing)
        self.refresh_all()
        self.tl.fit()
        self._schedule_audio()

    # ================================================================== file ops
    def _confirm_discard(self) -> bool:
        if not self.project.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, "Save changes to the current project?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    def new_project(self):
        if self._confirm_discard():
            self.set_project(Project())

    def open_project(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", self.qs.value("last_dir", ""), PROJECT_FILTER)
        if not path:
            return
        self.load_project_file(path)

    def load_project_file(self, path: str):
        log.info("Opening project %s", path)
        try:
            p = Project.load(path)
        except Exception as e:  # noqa: BLE001
            log.exception("Could not open project %s", path)
            QMessageBox.critical(self, APP_NAME, f"Could not open project:\n{e}")
            return
        log.info("Parsed project: %d footage, %d clips", len(p.footage), len(p.clips))
        missing = [f.name for f in p.footage.values() if not os.path.exists(f.path)]
        if missing:
            log.warning("Missing footage: %s", missing)
            QMessageBox.warning(self, APP_NAME, "Some footage files are missing:\n" + "\n".join(missing[:15]))
        self.set_project(p)
        log.info("Project loaded into the window")
        self._add_recent(path)

    # ------------------------------------------------------------------ recent projects
    MAX_RECENT = 10

    def _recent_projects(self) -> list[str]:
        v = self.qs.value("recent_projects", [])
        if isinstance(v, str):   # QSettings returns a lone entry as a plain string on some platforms
            v = [v]
        return [p for p in (v or []) if p]

    def _set_recent(self, paths: list[str]):
        self.qs.setValue("recent_projects", paths[:self.MAX_RECENT])
        self.recent_menu.setEnabled(bool(paths))

    def _add_recent(self, path: str):
        path = os.path.abspath(path)
        key = os.path.normcase(path)
        self._set_recent([path] + [p for p in self._recent_projects() if os.path.normcase(p) != key])

    def _fill_recent_menu(self):
        m = self.recent_menu
        m.clear()
        paths = self._recent_projects()
        for k, path in enumerate(paths, 1):
            name, folder = (s.replace("&", "&&") for s in (os.path.basename(path), os.path.dirname(path)))
            a = m.addAction(f"&{k % 10}  {name}    —  {folder}")
            a.setToolTip(path)
            a.setStatusTip(path)
            a.triggered.connect(lambda _=False, p=path: self.open_recent(p))
            if not os.path.exists(path):
                a.setText(a.text() + "  (missing)")
        if paths:
            m.addSeparator()
            m.addAction(self._act("Clear recent projects", lambda: self._set_recent([])))

    def open_recent(self, path: str):
        if not os.path.exists(path):
            r = QMessageBox.question(self, APP_NAME, f"The project file no longer exists:\n{path}\n\n"
                                     "Remove it from the recent projects?")
            if r == QMessageBox.Yes:
                key = os.path.normcase(path)
                self._set_recent([p for p in self._recent_projects() if os.path.normcase(p) != key])
            return
        if self._confirm_discard():
            self.load_project_file(path)

    def save_project(self, save_as: bool = False) -> bool:
        path = self.project.path
        if save_as or not path:
            start = path or os.path.join(self.qs.value("last_dir", ""), "Untitled" + Project.FILE_EXT)
            path, _ = QFileDialog.getSaveFileName(self, "Save project", start, PROJECT_FILTER)
            if not path:
                return False
            if not path.endswith(Project.FILE_EXT):
                path += Project.FILE_EXT
        try:
            self._backup(path, min_interval=60.0)
            self.project.save(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Could not save project:\n{e}")
            return False
        autosave.clear_recovery()
        self._update_title()
        self._add_recent(path)
        self.lbl_status.setText(f"Saved {path}")
        return True

    def _backup(self, path: str, min_interval: float = autosave.BACKUP_EVERY):
        """Keep the version on disk before it is overwritten (a failed backup never stops a save)."""
        try:
            autosave.make_backup(path, min_interval)
        except OSError:
            pass

    def _autosave(self):
        """Save a changed project by itself: in place when it has a file, else as a recovery copy."""
        if not self.a_autosave.isChecked() or not self.project.dirty:
            return
        try:
            if self.project.path:
                self._backup(self.project.path)
                self.project.save(self.project.path)
                self._update_title()
                self.lbl_status.setText(f"Auto-saved {os.path.basename(self.project.path)} "
                                        f"at {time.strftime('%H:%M')}")
            elif self.project.footage:
                autosave.write_recovery(self.project)
        except Exception as e:  # noqa: BLE001
            self.lbl_status.setText(f"Auto-save failed: {e}")

    def _offer_recovery(self):
        """After a crash: offer the untitled project the last session was working on."""
        found = autosave.stale_recoveries()
        if not found:
            return
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(found[0].stat().st_mtime))
        r = QMessageBox.question(self, APP_NAME, f"{APP_NAME} closed before an untitled project was saved.\n"
                                                 f"Restore it (auto-saved {when})?")
        p = None
        if r == QMessageBox.Yes and not self.project.dirty:
            try:
                p = Project.load(str(found[0]))
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, APP_NAME, f"Could not restore the project:\n{e}")
        for f in found:
            try:
                f.unlink()
            except OSError:
                pass
        if p is not None:
            p.path, p.dirty = None, True
            self.set_project(p)
            self.lbl_status.setText("Restored the untitled project: save it to keep it")

    def restore_backup(self):
        start = str(autosave.backup_dir(self.project.path)) if self.project.path else str(autosave.backup_root())
        path, _ = QFileDialog.getOpenFileName(self, "Restore a backup", start, PROJECT_FILTER)
        if not path or not self._confirm_discard():
            return
        try:
            p = Project.load(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Could not open the backup:\n{e}")
            return
        # Saving puts the restored version back in the original file (after backing up the current one).
        p.path, p.dirty = autosave.backup_source(path), True
        self.set_project(p)
        self.lbl_status.setText(f"Restored the backup {os.path.basename(path)}: save to keep it")

    def import_dialog(self):
        vids = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
        imgs = " ".join(f"*{e}" for e in images.IMAGE_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(self, "Import footage and images", self.qs.value("last_dir", ""),
                                                f"Video and images ({vids} {imgs});;Video files ({vids});;"
                                                f"Images ({imgs});;All files (*)")
        if paths:
            self.import_files(paths)

    def import_files(self, paths: list[str]):
        self.qs.setValue("last_dir", os.path.dirname(paths[0]))
        existing = {os.path.normcase(os.path.abspath(f.path)) for f in self.project.footage.values()}
        added, errors = [], []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in sorted(paths):
                if os.path.normcase(os.path.abspath(path)) in existing:
                    continue
                if images.is_image(path):
                    try:
                        f = self._image_footage(path)
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{os.path.basename(path)}: {e}")
                        continue
                    self.project.footage[f.id] = f
                    added.append(f.id)
                    continue
                try:
                    info = probe(path)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{os.path.basename(path)}: {e}")
                    continue
                f = Footage(path=os.path.abspath(path), duration=info.duration, width=info.width, height=info.height,
                            fps=info.fps, has_audio=info.has_audio, codec=info.codec, pix_fmt=info.pix_fmt,
                            bit_depth=info.bit_depth, color_space=info.color_space, color_range=info.color_range,
                            color_transfer=info.color_transfer, rotation=info.rotation,
                            creation_time=info.creation_time, gps=scenes.parse_location(info.tags),
                            projection=spherical.detect(path, info.projection, info.width, info.height))
                self._identify_camera(f, info)
                for h in (self.project.settings.proxy_height, 360, 540, 720, 1080):
                    pp = proxy_path_for(f, h)
                    if os.path.exists(pp):
                        f.proxy_path = pp
                        break
                self.project.footage[f.id] = f
                added.append(f.id)
        finally:
            QApplication.restoreOverrideCursor()
        if errors:
            QMessageBox.warning(self, APP_NAME, "Some files could not be imported:\n\n" + "\n".join(errors[:12]))
        if not added:
            return
        videos = [fid for fid in added if not self.project.footage[fid].is_image]
        if not self.project.clips and videos:
            first = self.project.footage[videos[0]]
            self.project.settings.fps = min(FRAME_RATES, key=lambda r: abs(r - first.fps))
            self._sync_project_controls()
        self.project.dirty = True
        if self.a_auto_proxy.isChecked():
            self.queue_proxies(videos)
        self.queue_analysis(videos)
        self.refresh_all(captions=False)
        n_img = len(added) - len(videos)
        self.lbl_status.setText(f"Imported {len(added)} file(s)" + (f" ({n_img} image(s))" if n_img else "")
                                + (". Analyzing in the background…" if videos else ". Drag them onto the timeline "
                                   "or onto a video layer above it."))
        if videos:
            if self._fill_transcripts(videos):
                self.update_captions()
            self.offer_transcription()

    def _image_footage(self, path: str) -> Footage:
        """Footage for a picture (see images.py): any clip length fits in it, it has no sound."""
        info = images.probe(path)
        return Footage(path=os.path.abspath(path), duration=images.STILL_LENGTH, width=info.width, height=info.height,
                       fps=info.fps if info.frames > 1 else self.project.settings.fps, has_audio=False,
                       codec=info.codec, pix_fmt="rgba" if info.has_alpha else "rgb24", bit_depth=8, kind="image",
                       camera="Animated GIF" if info.frames > 1 else "Image")

    # ================================================================== background jobs
    def _status_progress(self, text: str):
        self._progress_text = text
        self.lbl_status.setText(text)

    def _on_progress(self, p: float, msg: str):
        self.progress.setValue(int(p * 1000))
        pending = self.tasks._pending
        self._status_progress(msg + (f"   (+{pending - 1} queued)" if pending > 1 else ""))

    def _on_pending(self, n: int):
        self.progress.setVisible(n > 0)
        self.btn_cancel.setVisible(n > 0)
        if n == 0:
            self.progress.setValue(0)

    def _task_error(self, name: str):
        def handler(msg: str):
            if msg:
                if self.tray:
                    self.tray.notify(f"{name} failed", msg.strip().splitlines()[-1][:200],
                                     icon=self.tray.Warning)
                QMessageBox.warning(self, APP_NAME, f"{name} failed:\n\n{msg[-1500:]}")
            else:
                self.lbl_status.setText(f"{name}: cancelled")
        return handler

    def queue_analysis(self, ids: list[str], force: bool = False):
        """Analyze footage in two passes: a quick first look from the keyframes (all files at once, in a few
        seconds each), then the full analysis in the background, replacing each quick look as it finishes.
        Analyses made without the smarter analysis get it added in the background too (when it's on), never
        holding up other tasks. `force` (Re-analyze) runs the full analysis at once."""
        todo = [self.project.footage[fid] for fid in ids if fid in self.project.footage
                and not self.project.footage[fid].is_image]
        if not force:
            todo = [f for f in todo if f.id not in self.analyses or self.analyses[f.id].quick
                    or needs_smart(f, self.analyses[f.id])]
        if not todo:
            return
        use_gpu = self.a_gpu_analysis.isChecked()

        def job(progress, cancel):
            failed: list[str] = []
            if force:
                for f in todo:
                    for path in (analysis_path_for(f), quick_path_for(f)):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                analyze_many(todo, progress, cancel,
                             on_result=lambda f, a: self.analysis_ready.emit(f.id, a),
                             on_error=lambda f, msg: failed.append(f"{f.name}: {msg}"), use_gpu=use_gpu)
                return [], failed
            full, first, upgrade = [], [], []
            for f in todo:
                a = self.analyses.get(f.id) or load_plain(f)     # (opening a project has loaded most)
                if a is None:
                    first.append(f)
                    full.append(f)
                    continue
                if f.id not in self.analyses:            # (opening a project loads them already)
                    self.analysis_ready.emit(f.id, a)
                if a.quick:
                    full.append(f)
                elif needs_smart(f, a):
                    upgrade.append(f)
            if first:
                done = [0]

                def look(f):
                    if cancel.is_set():
                        return
                    try:
                        q = quick_look(f, cancel)
                        q.save(quick_path_for(f))
                        f.analysis_path = quick_path_for(f)
                        self.analysis_ready.emit(f.id, q)
                    except Exception:  # noqa: BLE001 - the full analysis tries again (and says why)
                        traceback.print_exc()
                    done[0] += 1
                    progress(done[0] / len(first), f"Quick first look {done[0]}/{len(first)}")
                workers = max(1, min(len(first), (os.cpu_count() or 4) // 2))
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="quick-look") as ex:
                    list(ex.map(look, first))
            return full + upgrade, failed

        def done(res):
            later, failed = res
            if failed:
                QMessageBox.warning(self, APP_NAME, "Analysis failed:\n\n" + "\n".join(failed)[-1500:])
            if later:
                self._refine(later)

        name = f"Analyzing {todo[0].name}" if len(todo) == 1 else f"Analyzing {len(todo)} clips"
        self.tasks.submit(name, job, done, self._task_error("Analysis"))

    # The full analysis after the quick first look, in the background (not in the task queue: an auto edit
    # started meanwhile runs at once, on the quick looks of the files not done yet).
    def _refine(self, footage: list):
        with self._refine_lock:
            have = {f.id for f in self._refine_queue}
            self._refine_queue += [f for f in footage if f.id not in have]
            if self._refine_thread is not None and self._refine_thread.is_alive():
                return
            cancel = self._refine_cancel
            self._refine_thread = threading.Thread(target=self._refine_loop, args=(cancel,), daemon=True,
                                                   name="analysis-refine")
            self._refine_thread.start()

    def _refine_loop(self, cancel: threading.Event):
        use_gpu = self.a_gpu_analysis.isChecked()
        while not cancel.is_set():
            with self._refine_lock:
                batch, self._refine_queue = self._refine_queue, []
            if not batch:
                break
            failed: list[str] = []
            # Files with a full analysis only need the smarter analysis added (no decoding): those first, several
            # at once (each model run uses two threads).
            rest, upgrade = [], []
            for f in batch:
                a = self.analyses.get(f.id)
                if a is None or a.quick:
                    rest.append(f)
                elif needs_smart(f, a):
                    upgrade.append((f, a))
            if upgrade and not self._upgrade_smart(upgrade, cancel):
                return
            batch = rest
            if not batch:
                continue
            try:
                analyze_many(batch, lambda p, m="": self.refine_status.emit(f"Full analysis in the background: {m} "
                                                                             f"({p:.0%})"),
                             cancel, on_result=lambda f, a: self.analysis_ready.emit(f.id, a),
                             on_error=lambda f, msg: failed.append(f"{f.name}: {msg}"), use_gpu=use_gpu)
            except Cancelled:
                return
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            if failed:
                self.refine_status.emit("Full analysis failed for: " + "; ".join(failed)[:300])
        if not cancel.is_set():
            self.refine_status.emit("Full analysis done")

    def _upgrade_smart(self, items: list, cancel: threading.Event) -> bool:
        """Add the smarter analysis to full analyses made without it, in parallel. False when cancelled."""
        done = [0]
        lock = threading.Lock()

        def one(item):
            f, a = item
            if cancel.is_set():
                return
            try:
                if make_smart(f, a, cancel):
                    a.save(f.analysis_path or analysis_path_for(f))
                    self.analysis_ready.emit(f.id, a)
            except Cancelled:
                return
            except Exception:  # noqa: BLE001 - the plain analysis still works
                traceback.print_exc()
            with lock:
                done[0] += 1
                n = done[0]
            self.refine_status.emit(f"Smarter analysis in the background: {n}/{len(items)} {f.name}")

        workers = max(1, min(len(items), (os.cpu_count() or 4) // 2))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="smart-upgrade") as pool:
            list(pool.map(one, items))
        return not cancel.is_set()

    def _stop_refining(self):
        with self._refine_lock:
            self._refine_cancel.set()
            self._refine_cancel = threading.Event()
            self._refine_queue = []

    def _on_refine_status(self, msg: str):
        if not self.tasks.busy:
            self.lbl_status.setText(msg)

    def _set_smart(self, on: bool):
        self.qs.setValue("smart_analysis", on)
        smart.set_enabled(on)
        if on:      # add it to the analyses made without it
            self.queue_analysis([fid for fid, a in self.analyses.items()
                                 if fid in self.project.footage and needs_smart(self.project.footage[fid], a)])

    def _on_analysis_ready(self, fid: str, a: Analysis):
        if fid in self.project.footage:
            self.analyses[fid] = a
            self._stale_thumbs.add(fid)
            f = self.project.footage[fid]
            if f.log_pending:
                self._log_from_picture(f, list(a.thumbs))
            # Hundreds of files finish one after the other (opening a big project, an import): one refresh for
            # all that came in within half a second, not a full refresh of the window for each.
            if not self._analysis_refresh.isActive():
                self._analysis_refresh.start()

    def _refresh_after_analysis(self):
        stale, self._stale_thumbs = self._stale_thumbs, set()
        self.tl._thumb_cache = {k: v for k, v in self.tl._thumb_cache.items() if k[0] not in stale}
        self.refresh_all(captions=False)

    def queue_proxies(self, ids: list[str]):
        height = self.project.settings.proxy_height
        use_gpu = self.qs.value("gpu_proxies", True, type=bool)
        for fid in ids:
            f = self.project.footage.get(fid)
            if f is None or f.is_image:
                continue

            def done(path, fid=fid):
                ff = self.project.footage.get(fid)
                if ff:
                    ff.proxy_path = path
                    self.viewer.drop_decoders()
                    self.project.dirty = True
                    # (The preview sound is not mixed again: a proxy sounds the same, and each new mix cancels the
                    # one being made, so proxies finishing one after another kept a new mix from ever finishing.)
                    self.refresh_all(captions=False)

            self.tasks.submit(f"Proxy {f.name}", lambda p, c, f=f: generate_proxy(f, height, p, c, use_gpu), done,
                              self._task_error(f"Proxy for {f.name}"))

    def proxy_dialog(self):
        if not self.project.footage:
            QMessageBox.information(self, APP_NAME, "Import some footage first.")
            return
        dlg = ProxyDialog(self.project, self.qs.value("gpu_proxies", True, type=bool), self)
        if dlg.exec() != ProxyDialog.Accepted:
            return
        self.qs.setValue("gpu_proxies", dlg.chk_gpu.isChecked())
        self.project.settings.proxy_height = dlg.cb_res.currentData()
        if dlg.chk_all.isChecked():
            ids = [fid for fid, f in self.project.footage.items() if not f.has_proxy() and not f.is_image]
            self.queue_proxies(ids)
            if not self.chk_proxy.isChecked():
                self.chk_proxy.setChecked(True)

    # ================================================================== auto edit
    def _load_auto_settings(self) -> AutoEditSettings:
        s = AutoEditSettings()
        for k in s.to_dict():
            v = self.qs.value(f"auto/{k}")
            if v is not None:
                cur = getattr(s, k)
                try:
                    setattr(s, k, (v in (True, "true", "1")) if isinstance(cur, bool) else type(cur)(v))
                except (TypeError, ValueError):
                    pass
        return s

    def script_edit_dialog(self):
        """Auto edit from a script: transcribe what is missing, match each line, lay out the clips."""
        from .script_dialog import ScriptEditDialog
        if not any(not f.is_image and f.has_audio for f in self.project.footage.values()):
            QMessageBox.information(self, APP_NAME, "Import footage with sound first: the script is matched to "
                                                    "what it says.")
            return
        dlg = ScriptEditDialog(self.project, self.bin.selected_ids(), self)
        if dlg.exec() != ScriptEditDialog.Accepted:
            return
        st = dlg.settings()
        self.project.settings.script = st
        self.project.dirty = True
        foot = [self.project.footage[i] for i in st["footage"] if i in self.project.footage]
        if not foot or not scriptedit.split_script(st["script"]):
            QMessageBox.information(self, APP_NAME, "Type a script and choose the footage to search.")
            return
        todo = [f.id for f in foot if f.transcript is None]
        if todo:
            self.lbl_status.setText(f"Transcribing {len(todo)} file(s) for the script…")
            self.queue_transcription(todo)

        def job(progress, cancel):     # runs after the transcriptions (the queue is in order)
            progress(0.0, "Matching the script to the footage…")
            return scriptedit.script_edit(st["script"], foot, st["threshold"], st["prefer"], st["pad_in"],
                                          st["pad_out"], st["merge"])

        self.tasks.submit("Auto edit from script", job, self._apply_script_edit,
                          self._task_error("Auto edit from script"))

    def finish_timeline(self, kind: str):
        """One finishing step (polish.py) on the main track: "beats", "sentences", "pauses", "fillers",
        "variety" or "loudness"; in the background, applied in one undo step."""
        clips = [TimelineClip.from_dict(c.to_dict()) for c in self.project.clips]
        footage = {fid: self.project.footage[fid] for fid in {c.footage_id for c in clips}
                   if fid in self.project.footage}
        if not clips:
            self.lbl_status.setText("The timeline is empty")
            return
        sound = polish.music_track(self.project.sounds) if kind == "beats" else None
        if kind == "beats" and sound is None:
            QMessageBox.information(self, APP_NAME, "Add music to the timeline first (Sounds tab → Background "
                                                    "music): the cuts move to its beat.")
            return
        if kind in ("fillers", "sentences"):
            todo = [fid for fid, f in footage.items() if f.has_audio and f.transcript is None]
            if todo:
                self.queue_transcription(todo)
        use_gpu = self.a_gpu_analysis.isChecked()
        s = copy.copy(self.last_auto)
        for k in ("cut_on_pauses", "whole_sentences", "beat_sync", "remove_fillers", "shot_variety",
                  "even_loudness"):
            setattr(s, k, False)
        s.pacing = "steady"
        s.video_format = "long"          # whole sentences are a long-form step
        flag = {"beats": "beat_sync", "pauses": "cut_on_pauses", "fillers": "remove_fillers",
                "variety": "shot_variety", "loudness": "even_loudness", "sentences": "whole_sentences"}[kind]
        setattr(s, flag, True)

        def job(progress, cancel):
            progress(0.1, "Reading the footage…")
            analyses = {fid: load_or_analyze(f, cancel=cancel, use_gpu=use_gpu) for fid, f in footage.items()
                        if not f.is_image}
            beats = self._music_beats(sound, cancel) if sound is not None else None
            before = [(c.src_in, c.src_out, c.volume) for c in clips]
            out, notes = polish.finish(clips, footage, analyses, s, beats, 0.0)
            changed = len(out) != len(before) or any(
                (c.src_in, c.src_out, c.volume) != b for c, b in zip(out, before))
            return analyses, out, notes, changed, beats

        def done(res):
            analyses, out, notes, changed, beats = res
            self.analyses.update(analyses)
            if kind == "beats" and (beats is None or len(beats) < 2):
                self.lbl_status.setText("No clear beat found in the music")
                return
            if not changed:
                self.lbl_status.setText("Nothing to change: the edit already does that")
                return
            self.project_checkpoint()
            self.project.clips = out
            self.refresh_all()
            self._schedule_audio()
            label = {"beats": "Cuts moved to the beat", "pauses": "Cuts moved between words",
                     "sentences": "Clips set on whole sentences",
                     "fillers": "Fillers and pauses taken out", "variety": "Shot sizes varied",
                     "loudness": "Loudness evened out"}[kind]
            self.lbl_status.setText(label + (" · " + " · ".join(notes) if notes else ""))

        self.tasks.submit("Finish the edit", job, done, self._task_error("Finish the edit"))

    def highlight_reel(self, seconds: Optional[float] = None):
        """Turn the edit into a short vertical reel: its best moments (only what the edit shows), a hook first,
        quick cuts at a trailer pace on the music's beat, 9:16 framed on the faces. Undo brings the edit back."""
        src = [c for c in self.project.clips if c.footage_id in self.project.footage
               and not self.project.footage[c.footage_id].is_image]
        if not src:
            QMessageBox.information(self, APP_NAME, "Make an edit first: the reel is cut from its moments.")
            return
        if seconds is None:
            choice, ok = QInputDialog.getItem(self, "Highlight reel", "Length of the reel:", ["30 s", "45 s", "60 s"],
                                              0, False)
            if not ok:
                return
            seconds = float(choice.split()[0])
        footage = {c.footage_id: self.project.footage[c.footage_id] for c in src}
        order = list(dict.fromkeys(c.footage_id for c in src))
        # Only what the edit shows: everything else of each file is blocked.
        blockers = []
        for fid, f in footage.items():
            t = 0.0
            for a, b in sorted((c.src_in, c.src_out) for c in src if c.footage_id == fid):
                if a - t > 0.05:
                    blockers.append(TimelineClip(fid, t, a))
                t = max(t, b)
            if f.duration - t > 0.05:
                blockers.append(TimelineClip(fid, t, f.duration))
        position = {}
        for k, c in enumerate(src):
            position.setdefault((c.footage_id, c.src_in, c.src_out), k)
        s = copy.copy(self.last_auto)
        s.video_format, s.target_duration, s.clip_min, s.clip_max = "short", seconds, 1.2, 3.0
        s.min_gap, s.max_uses, s.order, s.use_every_footage = 0.3, 0, "hook", False
        s.transitions, s.jl_cuts, s.bookends, s.pacing, s.remove_fillers = "off", False, False, "trailer", False
        s.replace_timeline = True
        sound = polish.music_track(self.project.sounds) if s.beat_sync else None
        use_gpu = self.a_gpu_analysis.isChecked()

        def where(c: TimelineClip) -> int:
            """Position in the edit of the clip a reel moment comes from (the reel keeps the edit's story)."""
            for (fid, a, b), k in position.items():
                if fid == c.footage_id and a - 0.01 <= c.src_in <= b + 0.01:
                    return k
            return len(src)

        def job(progress, cancel):
            progress(0.1, "Reading the edit's footage…")
            analyses = {fid: load_or_analyze(f, cancel=cancel, use_gpu=use_gpu) for fid, f in footage.items()}
            progress(0.6, "Choosing the best moments…")
            clips = select(footage, analyses, s, order, existing=blockers)
            hooks = [c for c in clips if c.label == "hook"]
            body = sorted((c for c in clips if c.label != "hook"), key=where)
            clips = hooks + body
            beats = self._music_beats(sound, cancel) if sound is not None else None
            clips, notes = polish.finish(clips, footage, analyses, s, beats, 0.0, avoid=blockers)
            apply_edit_style(clips, footage, analyses, s)
            return analyses, clips, notes

        def done(res):
            analyses, clips, notes = res
            self.analyses.update(analyses)
            if not clips:
                QMessageBox.information(self, APP_NAME, "No moments for a reel were found in the edit.")
                return
            self.project_checkpoint()
            self.project.clips = clips
            self.project.settings.aspect = "9:16 (Vertical)"
            self.project.settings.video_format = "short"
            self._sync_project_controls()
            self.tl.selected = ""
            self.refresh_all()
            self.tl.fit()
            self._timeline_seek(0)
            self._schedule_audio()
            self.lbl_status.setText(f"Highlight reel: {len(clips)} clips, {self.project.duration:.0f} s, 9:16 · "
                                    "Undo (Ctrl+Z) brings the full edit back" + (" · " + " · ".join(notes)
                                                                                  if notes else ""))
            self._face_zoom(clips, "medium")      # frame each clip on its face for the vertical picture

        self.tasks.submit("Highlight reel", job, done, self._task_error("Highlight reel"))

    def _apply_script_edit(self, result):
        st = self.project.settings.script
        if not result.clips:
            QMessageBox.information(self, APP_NAME, "None of the script's lines were found in the footage. Lower "
                                                    "the similarity threshold, or check the footage's language "
                                                    "(Captions settings).")
            return
        self.project_checkpoint()
        if st.get("replace", True):
            self.project.clips = result.clips
        else:
            self.project.clips.extend(result.clips)
        self.tl.selected = ""
        self.refresh_all()
        self.tl.fit()
        self._timeline_seek(0)
        found = len(result.matches) - len(result.missing)
        self.lbl_status.setText(f"Script edit: {found} of {len(result.matches)} lines found, {len(result.clips)} "
                                f"clip(s), {sum(c.duration for c in result.clips):.1f} s")
        if result.missing:
            box = QMessageBox(QMessageBox.Information, "Auto edit from script",
                              f"{len(result.missing)} of {len(result.matches)} lines were not found (below "
                              f"{st.get('threshold', scriptedit.DEFAULT_THRESHOLD):.0%} similarity) and were "
                              "left out.", parent=self)
            box.setDetailedText("\n".join(f"{m.index + 1}. {m.line}  (closest: {m.score:.0%})"
                                           for m in result.missing))
            box.exec()

    def auto_edit_dialog(self):
        if not self.project.footage:
            QMessageBox.information(self, APP_NAME, "Import some footage first (File → Import, or drag files "
                                                    "onto the Media panel).")
            return
        dlg = AutoEditDialog(self.project, self.bin.selected_ids(), self.last_auto, self)
        if dlg.exec() != AutoEditDialog.Accepted:
            return
        s = dlg.settings()
        ids = dlg.footage_ids()
        if not ids:
            return
        self.last_auto = s
        for k, v in s.to_dict().items():
            self.qs.setValue(f"auto/{k}", v)
        self._run_auto_edit(s, ids, dlg.aspect(), fresh=True, multi=dlg.multicam())

    def regenerate_auto_edit(self, factor: Optional[float]):
        """Rebuild the automatic edit with the last settings and footage, shorter / longer than the timeline
        (factor), or at a length the user enters (None)."""
        ids = [fid for fid in self._auto_ids if fid in self.project.footage] \
            or list(dict.fromkeys(c.footage_id for c in self.project.clips)) or list(self.project.footage)
        ids = [fid for fid in ids if not self.project.footage[fid].is_image]
        if not ids:
            self.auto_edit_dialog()
            return
        current = self.project.duration or self.last_auto.target_duration
        if factor is None:
            v, ok = QInputDialog.getDouble(self, "Regenerate auto edit",
                                           f"New length in seconds (the timeline is {current:.0f} s):",
                                           round(current), 5, 3600, 0)
            if not ok:
                return
            target = v
        else:
            target = current * factor
        total = sum(self.project.footage[fid].duration for fid in ids)
        target = float(round(max(5.0, min(target, total * 0.9))))
        s = copy.copy(self.last_auto)
        s.target_duration = target
        s.replace_timeline = True
        s.clip_min = min(s.clip_min, target)
        self.last_auto.target_duration = target
        self.qs.setValue("auto/target_duration", target)
        mc = self.project.settings.multicam or {}
        multi = mc if mc.get("angles") and any(c.angle >= 2 for c in self.project.overlays) else None
        if multi:
            ids = [fid for fid, k in mc["angles"].items() if k == 1 and fid in self.project.footage] or ids
        self._run_auto_edit(s, ids, None, f"Regenerating the edit at {target:.0f} s", multi=multi)

    def _run_auto_edit(self, s: AutoEditSettings, ids: list[str], aspect: Optional[str], name: str = "Auto edit",
                       fresh: bool = False, multi: Optional[dict] = None):
        """`fresh`: a new edit, which prefers moments the timeline and recent auto edits did not show.
        `multi`: multicam setup (see AutoEditDialog.multicam): `ids` are the main angle's footage; the other
        angles are synced by sound and laid on the video layers in step with the edit."""
        self._auto_ids = list(ids)
        mc_angles = {fid: k for fid, k in (multi or {}).get("angles", {}).items() if fid in self.project.footage}
        mc_known = self._known_syncs(mc_angles) if multi else {}
        mc_footage = {fid: self.project.footage[fid] for fid in mc_angles}
        use_proxies = self.project.settings.use_proxies
        self.queue_analysis(ids)
        footage = {fid: self.project.footage[fid] for fid in ids}
        existing = [] if s.replace_timeline else list(self.project.clips)
        avoid: list[TimelineClip] = []
        if fresh:
            avoid = [c for edit in self._auto_history for c in edit]
            if s.replace_timeline:
                avoid += self.project.clips
        seed = int(time.time() * 1000) & 0x7FFFFFFF if fresh else None
        use_gpu = self.a_gpu_analysis.isChecked()
        lang = self.project.settings.whisper_language
        # Footage in the B-roll bin can open the scenes even when it isn't among the files edited.
        broll = {fid: f for fid, f in self.project.footage.items()
                 if f.bin == binlib.BROLL and not f.is_image and fid not in footage}
        music_clip = polish.music_track(self.project.sounds) if s.beat_sync else None
        t_start = 0.0 if s.replace_timeline else self.project.duration
        if s.remove_fillers and not multi:     # fillers are found in the transcripts: make the missing ones first
            todo = [fid for fid, f in footage.items() if f.has_audio and f.transcript is None]
            if todo:
                self.lbl_status.setText(f"Transcribing {len(todo)} file(s) to find the fillers…")
                self.queue_transcription(todo)

        def job(progress, cancel):
            syncs, faces, blockers = {}, {}, []
            if multi:
                syncs, faces = self._multicam_prepare(mc_angles, mc_footage, mc_known, use_proxies, progress, cancel)
                if multi.get("covered", True):
                    blockers = multicam.coverage_blockers(mc_footage, mc_angles, syncs)
            progress(0.3, "Analyzing footage…")
            analyses = {fid: load_or_analyze(f, cancel=cancel, use_gpu=use_gpu) for fid, f in footage.items()}
            quick = sum(1 for a in analyses.values() if getattr(a, "quick", False))
            if multi and multi.get("template") == "switch":       # switching scores every angle's picture
                progress(0.45, "Analyzing the camera angles…")
                for fid, f in mc_footage.items():
                    if fid not in analyses:
                        analyses[fid] = load_or_analyze(f, cancel=cancel, use_gpu=use_gpu)
            order = story_order(footage, ids)
            shot = (s.clip_min, s.clip_max)
            if multi and multi.get("whole"):
                # The whole recording, cut only between the angles (no best moments, hook or bookends). With
                # switching, the main track is cut at every angle change (shots within the clip lengths), so
                # the transitions play there and each angle shot follows its own clip.
                clips = multicam.whole_clips(mc_footage, mc_angles, syncs, order, multi.get("covered", True))
                plan, switches = None, set()
                if multi.get("template") == "switch":
                    progress(0.8, "Cutting between the angles…")
                    plan = multicam.switch_plan(self.project, analyses, mc_angles, syncs, shot)
                    clips = multicam.split_at_shots(clips, plan)
                    shown = [multicam.shot_angle(plan, c) for c in clips]
                    switches = {k for k in range(1, len(clips)) if shown[k] != shown[k - 1]}
                apply_edit_style(clips, footage, analyses, s, switches)
                note = f"Whole recording: {sum(c.duration for c in clips) / 60:.1f} min"
                return analyses, clips, [note], syncs, faces, plan
            intro = outro = None
            notes: list[str] = []
            if quick:
                notes.append(f"{quick} file(s) from the quick first look: regenerate when the full analysis is done "
                             "for the best edit")
            sel = s
            if s.bookends:
                progress(0.5, "Finding the opening and closing…")
                max_len = 20.0 if s.video_format == "long" else max(4.0, s.clip_max * 2)
                intro, outro, notes = find_bookends(
                    footage, order, max_len, s.clip_min,
                    listen=lambda f, a, d: listen(f, a, d, "small", lang, cancel),
                    progress=lambda msg: progress(0.5, msg), cancel=cancel)
                if cancel.is_set():
                    raise Cancelled()
                ends = [c for c in (intro, outro) if c is not None]
                sel = copy.copy(s)
                sel.target_duration = max(s.clip_min, s.target_duration - sum(c.duration for c in ends))
            # Scenes are time groups (and places) of the footage; in an edit in story order each scene after the
            # first opens with a scenic / travel shot from it, B-roll first (room for them is kept in the target).
            # A hook-first edit has one hook, at the start of the timeline: no opener plays like another one.
            everything = {**footage, **broll}
            scenes.read_locations(everything)
            for fid, f in broll.items():
                if fid not in analyses:
                    analyses[fid] = load_or_analyze(f, cancel=cancel, use_gpu=use_gpu)
            # The scenes come from the edited footage alone; B-roll joins the scene it was recorded in.
            groups = scenes.join_scenes(scenes.time_groups(footage, order), footage, broll)
            n_scenes = len({groups[f] for f in footage if f in groups})
            openers = s.order == "chronological"
            opener_len = round((s.clip_min + s.clip_max) / 2, 2)
            if openers and n_scenes > 1:
                sel = copy.copy(sel)
                sel.target_duration = max(s.clip_min, sel.target_duration - opener_len * (n_scenes - 1))
            progress(0.8, "Choosing the best moments…")
            used = existing + blockers + [c for c in (intro, outro) if c]
            body = select(footage, analyses, sel, order, existing=used, avoid=avoid, seed=seed)
            hooks = list(itertools.takewhile(lambda c: c.label == "hook", body))
            body = body[len(hooks):]
            # The hook grabs attention first, then the opening greets.
            clips = hooks + ([intro] if intro else []) + body + ([outro] if outro else [])
            if openers and n_scenes > 1:
                prefer = set(broll) | {f for f in footage if footage[f].bin == binlib.BROLL}
                n = scenes.add_scene_openers(clips, groups, analyses, everything, opener_len, s.min_gap, used,
                                             prefer=prefer)
                if n:
                    notes.append(f"{n} scene(s) open with a scenic shot")
            progress(0.9, "Finishing the edit…")
            beats = self._music_beats(music_clip, cancel) if music_clip is not None else None
            clips, more = polish.finish(clips, everything, analyses, s, beats, t_start, avoid=used)
            notes += more
            # Nothing shown twice: a clip that ended up on (or too near) a moment another one shows is left out.
            clips, dropped = polish.drop_repeats(clips, s.min_gap, used)
            if dropped:
                notes.append(f"{dropped} repeated moment(s) left out")
            apply_edit_style(clips, everything, analyses, s, groups=groups)
            plan = (multicam.switch_plan(self.project, analyses, mc_angles, syncs, shot)
                    if multi and multi.get("template") == "switch" else None)
            return analyses, clips, notes, syncs, faces, plan

        def done(res):
            analyses, clips, notes, syncs, faces, plan = res
            self.analyses.update(analyses)
            if not clips:
                QMessageBox.information(self, APP_NAME, "No suitable moments were found. Try shorter clip lengths "
                                                        "or a smaller gap.")
                return
            self.project_checkpoint()
            self._tried_hooks = []
            self._tried_alts = {}
            self._auto_history = (self._auto_history + [[TimelineClip.from_dict(c.to_dict()) for c in clips]])[-3:]
            if s.replace_timeline:
                self.project.clips = clips
            else:
                self.project.clips.extend(clips)
            self.project.settings.video_format = s.video_format
            if aspect:
                self.project.settings.aspect = aspect
                self._sync_project_controls()
            if multi:
                notes = self._multicam_place(multi, mc_angles, syncs, faces, clips, s.replace_timeline, plan,
                                             (s.clip_min, s.clip_max)) + notes
            elif s.replace_timeline:
                self.project.overlays = [c for c in self.project.overlays if c.angle == 0]
            self.tl.selected = ""
            self.refresh_all()
            self.tl.fit()
            self._timeline_seek(0)
            self._schedule_audio()
            msg = f"Auto edit: {len(clips)} clips, {self.project.duration:.1f} s"
            if (s.replace_timeline and self.project.duration < 0.85 * s.target_duration
                    and not (multi and multi.get("whole"))):
                msg += (f" (only this much distinct material for the {s.target_duration:.0f} s target — "
                        "lower 'Gap between reuses' or allow more uses per footage)")
            self.lbl_status.setText(msg + (" · " + " · ".join(notes) if notes else ""))

        self.tasks.submit(name, job, done, self._task_error("Auto edit"))

    @staticmethod
    def _music_beats(sound, cancel=None) -> np.ndarray:
        """(In the background.) The beats of a music sound clip as timeline times (empty when its file can't be
        read or has no clear beat)."""
        from ..sync import RATE, load_audio
        try:
            audio = load_audio(sound.path, cancel)
        except Cancelled:
            raise
        except Exception:  # noqa: BLE001
            return np.zeros(0)
        return polish.timeline_beats(sound, polish.detect_beats(audio, RATE))

    # ================================================================== multicam
    def _known_syncs(self, angles: dict[str, int]) -> dict[str, tuple[str, SyncResult]]:
        """Syncs found by an earlier multicam edit that still apply (same main-angle files)."""
        out = {}
        for fid, (ref, o, d, conf, ov) in ((self.project.settings.multicam or {}).get("syncs") or {}).items():
            if angles.get(fid, 0) >= 2 and angles.get(ref) == 1:
                out[fid] = (ref, SyncResult(o, d, conf, ov))
        return out

    def _multicam_prepare(self, angles: dict[str, int], footage: dict[str, Footage], known: dict,
                          use_proxies: bool, progress, cancel) -> tuple[dict, dict]:
        """(In the background.) Sync every other angle to the main angle by sound, and find each angle's main
        face (for cropping it into a layout box). Returns ({footage id: (main id, SyncResult)},
        {footage id: face})."""
        mains = [fid for fid, k in angles.items() if k == 1]
        todo = [fid for fid, k in angles.items() if k >= 2 and fid not in known]
        syncs = dict(known)
        if todo:
            ref_of = {footage[f].path: f for f in mains}
            # Everything is synced to the longest recording (it usually spans all the others), and each angle's
            # offset to the main angle worked out through it; what doesn't match it is synced directly.
            durations = {f.path: f.duration for f in footage.values()}
            by_path = sync_to_longest([footage[f].path for f in mains], [footage[f].path for f in todo],
                                      durations, lambda p, m: progress(0.2 * p, m), cancel)
            left = [f for f in todo if footage[f].path not in by_path]
            if left:
                by_path.update(sync_files([footage[f].path for f in mains], [footage[f].path for f in left],
                                          lambda p, m: progress(0.2 + 0.05 * p, m), cancel))
            for fid in todo:
                hit = by_path.get(footage[fid].path)
                if hit is not None:
                    syncs[fid] = (ref_of[hit[0]], hit[1])
        faces = {}
        pool = DecoderPool(limit=2)
        try:
            for k, (fid, f) in enumerate(footage.items()):
                if cancel.is_set():
                    raise Cancelled()
                if f.face is not None:
                    continue
                progress(0.25 + 0.05 * k / max(1, len(footage)), "Finding the speakers…")
                dec = pool.get(f, use_proxies, 480)
                frames = [img.copy() for i in range(7) if (img := dec.frame_at(f.duration * (i + 0.5) / 7)) is not None]
                face = reframe.find_face(frames)
                faces[fid] = list(face) if face is not None else []
        finally:
            pool.close()
        return syncs, faces

    def _multicam_place(self, multi: dict, angles: dict[str, int], syncs: dict, faces: dict,
                        clips: list[TimelineClip], replace: bool, plan: Optional[dict] = None,
                        shot: Optional[tuple[float, float]] = None) -> list[str]:
        """Lay the synced angles over the new clips (with switching, the shots of `plan`, else a new plan);
        returns status notes. shot: the shortest and longest shot (the auto edit's clip lengths)."""
        for fid, face in faces.items():
            if fid in self.project.footage:
                self.project.footage[fid].face = face
        if replace:
            self.project.overlays = [c for c in self.project.overlays if c.angle == 0]
        if plan is None:
            plan = (multicam.switch_plan(self.project, self.analyses, angles, syncs, shot)
                    if multi["template"] == "switch" else (self.project.settings.multicam or {}).get("switch"))
        new = multicam.build(self.project, clips, angles, syncs, multi["template"], multi.get("audio", "main"),
                             switch=plan)
        self.project.overlays += new
        top = max((c.track for c in self.project.overlays), default=0)
        self.project.settings.overlay_tracks = max(1, top + 1)
        self.project.settings.multicam = {
            "angles": dict(angles), "template": multi["template"], "audio": multi.get("audio", "main"),
            "covered": multi.get("covered", True), "whole": multi.get("whole", False), "switch": plan or {},
            "shot": list(shot or (self.project.settings.multicam or {}).get("shot") or
                         (multicam.MIN_SHOT, multicam.MAX_SHOT)),
            "syncs": {fid: [ref, r.offset, r.drift, r.confidence, r.overlap] for fid, (ref, r) in syncs.items()},
            "clips": [c.id for c in self.project.clips if angles.get(c.footage_id) == 1]}
        notes = []
        for fid, k in sorted(angles.items(), key=lambda it: it[1]):
            if k < 2:
                continue
            name = self.project.footage[fid].name
            if fid in syncs:
                ref, r = syncs[fid]
                drift = f", drift {r.drift * 1e6:+.0f} ppm" if abs(r.drift) > 2e-6 else ""
                notes.append(f"{name} synced to {self.project.footage[ref].name} at {r.offset:+.3f} s{drift}")
            else:
                notes.append(f"⚠ {name}: no matching sound found, left out")
        missing = [self.project.footage[f].name for f, k in angles.items() if k >= 2 and f not in syncs]
        if missing:
            QMessageBox.warning(self, APP_NAME, "These angles could not be synced to the main angle by their "
                                                "sound (no common sound, or no sound at all), so they were left "
                                                "out:\n\n" + "\n".join(missing))
        return notes

    def multicam_layout(self, template: str):
        """Move the synced angles to another layout template."""
        if not any(c.angle >= 2 for c in self.project.overlays):
            QMessageBox.information(self, APP_NAME, "There are no synced angles on the timeline yet: turn on "
                                                    "Multicam in ✨ Auto Edit.")
            return
        if template == "switch":
            self._multicam_switch()
            return
        self.project_checkpoint()
        n = multicam.relayout(self.project, template)
        self._overlays_changed()
        self.lbl_status.setText(f"Multicam layout: {multicam.TEMPLATES[template]} ({n} camera-angle clips)")

    def _multicam_switch(self):
        """Switch to the switching layout: analyse the angles (in the background), plan the cuts, rebuild."""
        mc = self.project.settings.multicam or {}
        footage = {fid: self.project.footage[fid] for fid in (mc.get("angles") or {}) if fid in self.project.footage}
        have = {fid: self.analyses[fid] for fid in footage if fid in self.analyses}
        use_gpu = self.a_gpu_analysis.isChecked()

        def job(progress, cancel):
            out = dict(have)
            todo = [fid for fid in footage if fid not in out]
            for k, fid in enumerate(todo):
                progress(k / max(1, len(todo)), f"Analyzing the camera angles: {k + 1}/{len(todo)}")
                out[fid] = load_or_analyze(footage[fid], cancel=cancel, use_gpu=use_gpu)
            return out

        def done(analyses: dict):
            self.analyses.update(analyses)
            self.project_checkpoint()
            st = self.project.settings
            st.multicam = {**st.multicam, "switch": multicam.switch_plan(self.project, self.analyses)}
            n = multicam.relayout(self.project, "switch")
            self._overlays_changed()
            cuts = sum(max(0, len(v) - 1) for v in st.multicam["switch"].values())
            self.lbl_status.setText(f"Multicam layout: switching ({n} camera-angle shots; the recording has "
                                    f"{cuts} angle changes to choose from)")

        self.tasks.submit("Multicam switching", job, done, self._task_error("Multicam switching"))

    def new_hook(self, choose: bool = True):
        """Replace the hook (the leading clips labelled "hook", else the first clip unless it is the opening)
        with other strong moments of the whole video: one clip in short-form, a 2-3 clip teaser in long-form (or
        as many as the auto edit's Hook setting asks). `choose`: offer the strongest moments as thumbnails to
        choose from (and how many clips the hook has); else take the next strongest ones not tried yet. On a
        timeline without a hook (or that starts with the opening) the hook is put in front."""
        clips = self.project.clips
        ids = [fid for fid in (self._auto_ids or list(self.project.footage))
               if fid in self.project.footage and fid in self.analyses]
        if not ids:
            QMessageBox.information(self, APP_NAME, "Analyze some footage first (or run ✨ Auto Edit).")
            return
        s = copy.copy(self.last_auto)
        s.video_format = self.project.settings.video_format or s.video_format
        n, length = hook_plan(s)
        old = list(itertools.takewhile(lambda c: c.label == "hook", clips))
        if not old and clips and clips[0].label != "opening" and n == 1:
            old = [clips[0]]
        footage = {fid: self.project.footage[fid] for fid in ids}
        analyses = {fid: self.analyses[fid] for fid in ids}
        rest = [c for c in clips if c not in old]

        def length_for(k: int) -> float:
            """Seconds each moment of a k-clip hook lasts (one clip keeps the length the hook has)."""
            if k == 1 and len(old) == 1:
                return old[0].duration
            ss = copy.copy(s)
            ss.hook_clips = k
            return hook_plan(ss)[1]

        if choose:
            from .hook_picker import HookPicker, card_name, strip

            def fetch(k: int, shown: list) -> list:
                return hook_suggestions(footage, analyses, s, rest + shown, max(HOOK_SUGGESTIONS, k), length_for(k))

            dlg = HookPicker(self, n, fetch,
                             lambda c: strip(self.analyses.get(c.footage_id), c.src_in, c.src_out),
                             lambda c: card_name(c, self.project.footage[c.footage_id].name), length_for,
                             self._moment_frames)
            if dlg.exec() != QDialog.Accepted or not dlg.chosen():
                return
            hooks = [TimelineClip.from_dict({**c.to_dict(), "id": new_id()}) for c in dlg.chosen()]
            if dlg.count != n:                       # the hook's length is remembered for the next auto edit
                self.last_auto.hook_clips = dlg.count
                self.qs.setValue("auto/hook_clips", dlg.count)
        else:
            for o in old:
                if not any(_same_moment(o, t) for t in self._tried_hooks):
                    self._tried_hooks.append(TimelineClip.from_dict(o.to_dict()))

            def find(avoid):
                return hook_suggestions(footage, analyses, s, avoid, n, length_for(n))

            hooks = find(rest + self._tried_hooks)
            if len(hooks) < n and self._tried_hooks:
                # Every strong moment has been shown once: start the round again (still avoiding the timeline).
                self._tried_hooks = [TimelineClip.from_dict(o.to_dict()) for o in old]
                hooks = find(rest + self._tried_hooks)
            if not hooks:
                self.lbl_status.setText("No other hook moment found (every strong moment is already in the edit).")
                return
            self._tried_hooks += [TimelineClip.from_dict(h.to_dict()) for h in hooks]
        self.project_checkpoint()
        for k, h in enumerate(hooks):
            h.label = "hook"
            if k < len(old):
                # Keep the clip's look and audio settings; only the moment changes.
                o = old[k]
                o.footage_id, o.src_in, o.src_out, o.score, o.label = h.footage_id, h.src_in, h.src_out, h.score, "hook"
                o.motion_face, o.voice_floor = None, 0.0      # (the old moment's face and noise: found again)
                hooks[k] = o
        self.project.clips = hooks + rest
        self._after_structure_change()
        self._measure_noise(hooks)
        self._find_motion_faces(hooks)
        self.measure_shake()
        self.tl.selected = hooks[0].id
        self.tl.update()
        self.inspector.set_clip(self._inspected())
        self._timeline_seek(0)
        f = self.project.footage[hooks[0].footage_id]
        if len(hooks) > 1:
            what = f"{len(hooks)} clips, {sum(h.duration for h in hooks):.1f} s"
        else:
            what = f"{f.name} at {hooks[0].src_in:.1f} s (score {(hooks[0].score or 0) * 100:.0f})"
        self.lbl_status.setText(f"New hook: {what}")

    def _moment_frames(self, c: TimelineClip, cancel: threading.Event) -> list:
        """A moment's frames for a preview: small, PREVIEW_FPS a second (on a background thread)."""
        from .hook_picker import PREVIEW_FPS
        f = self.project.footage[c.footage_id]
        if f.is_image:
            img = images.thumbnail(f.path, 240)
            return [img] if img is not None else []
        proxy = self.project.settings.use_proxies and f.has_proxy()
        dec = FrameDecoder(f.proxy_path if proxy else f.path, 0 if proxy else f.rotation,
                           "bt709" if proxy else f.color_space, 240)
        try:
            out, t = [], c.src_in
            while t < c.src_out and not cancel.is_set():
                img = dec.frame_at(t)
                if img is not None:
                    out.append(img)
                t += 1.0 / PREVIEW_FPS
            return out
        finally:
            dec.close()

    def _best_for(self, fid: str, duration: Optional[float] = None, existing=None,
                  relaxed: bool = False, free_length: bool = False) -> Optional[TimelineClip]:
        """The best moment of a footage that avoids `existing` clips. `relaxed` only rules out overlaps
        (no gap or look-alike rule), for footage the edit already uses closely. `free_length`: the moment
        takes its own best length (within the edit's clip lengths, stretched to include `duration`)
        instead of exactly `duration`."""
        a = self.analyses.get(fid)
        if a is None:
            return None
        s = copy.copy(self.last_auto)
        d = duration or s.clip_max
        if free_length:
            s.clip_min, s.clip_max = min(s.clip_min, d), max(s.clip_max, d)
            s.target_duration = s.clip_max
        else:
            s.target_duration = d
            s.clip_min = s.clip_max = d     # same length: the rest of the timeline stays where it is
        s.use_every_footage = True
        s.max_uses = 0
        if relaxed:
            s.min_gap = 0.0
            s.similarity = 1.01
        clips = select({fid: self.project.footage[fid]}, {fid: a}, s, [fid],
                       existing=self.project.clips if existing is None else existing)
        return clips[0] if clips else None

    @staticmethod
    def _slip_window(x: TimelineClip) -> float:
        """How far either side of where the clip first was Other moment looks, in seconds."""
        return max(6.0, 2.0 * x.duration)

    def _outside_window(self, x: TimelineClip) -> list[TimelineClip]:
        """Stand-in clips covering the footage outside the Other moment window around clip x."""
        w = self._slip_window(x)
        dur = self.project.footage[x.footage_id].duration
        lo, hi = max(0.0, x.src_in - w), min(dur, x.src_out + w)
        out = []
        if lo > 0.0:
            out.append(TimelineClip(x.footage_id, 0.0, lo))
        if hi < dur:
            out.append(TimelineClip(x.footage_id, hi, dur))
        return out

    def _slip_moment(self, x: TimelineClip, others: list[TimelineClip],
                     tried: list[TimelineClip]) -> Optional[TimelineClip]:
        """The best moment near where the clip first was (tried[0]), at that moment's own best length. The
        first press may just nudge it; later presses step to the next best nearby moments."""
        home = tried[0]
        others = [c for c in others if not _same_moment(c, x)] + self._outside_window(home)

        def best(avoid: list[TimelineClip]) -> Optional[TimelineClip]:
            return self._best_for(x.footage_id, home.duration, existing=avoid, relaxed=True, free_length=True)

        b = best(others)
        if b is not None and abs(b.src_in - x.src_in) >= 0.2 and not any(
                abs(b.src_in - t.src_in) < 0.2 for t in tried):
            return self._fit_length(b, home.duration, others)
        alt = best(others + tried)
        if alt is None and len(tried) > 1:
            del tried[1:]       # every nearby moment was shown: start the round again
            alt = best(others + [x])
        return None if alt is None else self._fit_length(alt, home.duration, others)

    def _fit_length(self, m: TimelineClip, duration: float, others: list[TimelineClip]) -> TimelineClip:
        """Give moment m its natural length (see autoedit.natural_length), within the edit's clip lengths
        (stretched to include `duration`) and without running into a later clip of the same footage."""
        fid = m.footage_id
        a, f = self.analyses[fid], self.project.footage[fid]
        sc = score_samples({fid: a}, {fid: f}, self.last_auto.style)[fid]
        lo, hi = min(self.last_auto.clip_min, duration), max(self.last_auto.clip_max, duration)
        limit = min((c.src_in for c in others if c.footage_id == fid and c.src_in > m.src_in + 0.05),
                    default=f.duration)
        length = natural_length(a, sc, f.duration, m.src_in, lo, hi, limit)
        if length > 0.2:
            m.src_out = m.src_in + length
        return m

    def _replace_moments(self, targets: list[TimelineClip]):
        """Other moment: swap each clip for the best moment close by in its footage; the clip takes that
        moment's length (the timeline ripples). Pressing again keeps offering new moments; once every
        nearby moment was shown the round starts again."""
        if not targets:
            self.lbl_status.setText("Select a clip on the timeline to replace it with another moment.")
            return
        missing = list(dict.fromkeys(x.footage_id for x in targets if x.footage_id not in self.analyses))
        if missing:
            self.queue_analysis(missing)
            self.lbl_status.setText("Analyzing the footage first: try again when the analysis is done.")
            return
        changes = []
        for x in targets:
            tried = self._tried_alts.setdefault(x.id, [])
            if not any(abs(t.src_in - x.src_in) < 0.05 for t in tried):
                tried.append(TimelineClip.from_dict(x.to_dict()))
            others = [c for c in self.project.clips if c.id != x.id and c.id not in {t.id for t, _ in changes}]
            others += [alt for _, alt in changes]
            alt = self._slip_moment(x, others, tried)
            if alt is not None:
                changes.append((x, alt))
        if not changes:
            self.lbl_status.setText("No other moment close by.")
            return
        self.project_checkpoint()
        for x, alt in changes:
            x.src_in, x.src_out, x.score = alt.src_in, alt.src_out, alt.score
        self._after_structure_change()
        if len(changes) == 1:
            x = changes[0][0]
            self._timeline_seek(self.project.clip_starts()[self.project.index_of(x.id)] + 0.01)
            self.lbl_status.setText(f"Other moment: {fmt_time(x.src_in, frames=False)} of "
                                    f"{self.project.footage[x.footage_id].name}, {x.duration:.1f} s "
                                    "(R again for another)")
        else:
            self.lbl_status.setText(f"Other moment for {len(changes)} clips")

    # ================================================================== bin actions
    def bin_action(self, action: str, ids: list[str]):
        if not ids:
            return
        if action.startswith("projection:"):     # 360° footage (or not): its clips show a view of the sphere
            self.project_checkpoint()
            for fid in ids:
                self.project.footage[fid].projection = action.split(":", 1)[1]
            self.project.dirty = True
            self.bin.refresh()
            self.tl.refresh()
            self.viewer.refresh()
            self.inspector.set_clip(self.inspector.clip)
            self._update_title()
            return
        if action == "play":
            self.viewer.open_source(ids[0])
            return
        if action == "remove":
            usage = self.project.usage()
            used = [fid for fid in ids if usage.get(fid)]
            if used and QMessageBox.question(self, APP_NAME, "Some of this footage is used on the timeline. "
                                                             "Remove it and its clips?") != QMessageBox.Yes:
                return
            self.project_checkpoint()
            self.project.clips = [c for c in self.project.clips if c.footage_id not in ids]
            self.project.overlays = [c for c in self.project.overlays if c.footage_id not in ids]
            for fid in ids:
                self.project.footage.pop(fid, None)
                self.analyses.pop(fid, None)
            self.refresh_all()
            self._schedule_audio()
        elif action == "append_whole":
            self.project_checkpoint()
            for fid in ids:
                self.project.clips.append(self._whole_clip(fid))
            self._after_structure_change()
        elif action == "overlay":
            self.add_overlays(ids)
        elif action == "append_best" and all(self.project.footage[fid].is_image for fid in ids):
            self.bin_action("append_whole", ids)
        elif action == "append_best":
            ids = [fid for fid in ids if not self.project.footage[fid].is_image] or ids
            missing = [fid for fid in ids if fid not in self.analyses]
            if missing:
                self.queue_analysis(missing)
                self.lbl_status.setText("Analyzing first — try again when analysis is finished.")
                return
            self.project_checkpoint()
            added = 0
            for fid in ids:
                c = self._best_for(fid)
                if c:
                    self.project.clips.append(c)
                    added += 1
            if not added:
                self.lbl_status.setText("No unused moment left in that footage (gap rule).")
            self._after_structure_change()
        elif action == "proxy":
            self.queue_proxies(ids)
        elif action == "delete_proxy":
            self.viewer.close_source(ids)          # (the file it plays is let go of: Windows can't delete an open one)
            for fid in ids:
                f = self.project.footage[fid]
                self.viewer.drop_decoders()
                if f.proxy_path and os.path.exists(f.proxy_path):
                    try:
                        os.remove(f.proxy_path)
                    except OSError as e:
                        QMessageBox.warning(self, APP_NAME, str(e))
                f.proxy_path = None
            self.refresh_all(captions=False)
        elif action == "analyze":
            self.queue_analysis(ids, force=True)
        elif action == "transcribe":
            self.transcribe_media(ids)
        elif action == "transcribe_force":
            self.transcribe_again(ids)
        elif action == "transcript":
            self.transcript_dialog(ids)
        elif action == "detect_log":
            self.detect_log(ids)
        elif action == "import_lut_log":
            self.import_lut("log", ids)

    def set_log_profile(self, ids: list[str], key: str):
        self.project_checkpoint()
        for fid in ids:
            f = self.project.footage.get(fid)
            if f is not None:
                f.log_profile, f.log_note, f.log_pending = key, "Chosen manually", False
        self.refresh_all(captions=False)
        entry = lutlib.LUT_ONLY.get(key)
        if entry is not None and missing_lut(key):
            r = QMessageBox.question(
                self, APP_NAME, f"{entry.menu_label} converts with {entry.maker}'s own LUT, which isn't in the "
                f"LUT library yet.\n\nUntil it is, the footage is shown and exported with the built-in "
                f"{profile_label(entry.base)} curve as an approximation.\n\nGet the LUT now?",
                QMessageBox.Yes | QMessageBox.No)
            if r == QMessageBox.Yes:
                self.official_luts_dialog()

    # ================================================================== camera / log detection
    def _identify_camera(self, f: Footage, info) -> None:
        """At import: name the camera and apply a log profile the file states outright. When only the camera
        is known, the profile is decided from the picture once analysis has thumbnails."""
        try:
            cam = camera.detect(f.path, info)
        except Exception:  # noqa: BLE001 - identification is best effort
            return
        f.camera = cam.label
        if not self.a_auto_log.isChecked():
            return
        if cam.log_profile is not None:
            f.log_profile, f.log_note = self._official_profile(cam.log_profile), f"Detected from {cam.source}"
        else:
            f.log_guess, f.log_pending = cam.guess, True

    def _log_from_picture(self, f: Footage, frames) -> None:
        profile, note = camera.decide_from_picture(f.log_guess, f.camera, frames)
        if profile is not None and profile != f.log_profile and f.log_profile == "none":
            self.project.set_detected_profile(f.id, self._official_profile(profile), f"Detected: {note}")
            self.lbl_status.setText(f"{f.name}: {profile_label(profile)} ({note})")
        else:
            f.log_note, f.log_pending = note, False

    def detect_log(self, ids: list[str]):
        """Re-run camera identification and choose the log profile for the given footage (in the background)."""
        ids = [fid for fid in ids if fid in self.project.footage and not self.project.footage[fid].is_image]
        if not ids:
            return
        footage = {fid: self.project.footage[fid] for fid in ids}
        before = {fid: f.log_profile for fid, f in footage.items()}   # a choice made meanwhile wins
        thumbs = {fid: list(self.analyses[fid].thumbs) for fid in ids
                  if fid in self.analyses and len(self.analyses[fid].thumbs)}
        use_proxies = self.project.settings.use_proxies
        name = "Detect log profile" + (f" ({len(ids)} files)" if len(ids) > 1 else "")

        def job(progress, cancel):
            pool = DecoderPool(limit=2)
            results = []
            try:
                for k, (fid, f) in enumerate(footage.items()):
                    if cancel.is_set():
                        raise Cancelled()
                    progress(k / len(footage), f"{name}: {f.name}")
                    try:
                        cam = camera.detect(f.path, probe(f.path))
                        if cam.log_profile is not None:
                            profile, note = cam.log_profile, f"from {cam.source}"
                        else:
                            frames = thumbs.get(fid)
                            if frames is None:   # not analyzed yet: sample the picture
                                dec = pool.get(f, use_proxies, 360)
                                frames = [img for img in (dec.frame_at(f.duration * (i + 0.5) / 5) for i in range(5))
                                          if img is not None]
                            profile, note = camera.decide_from_picture(cam.guess, cam.label, frames)
                        results.append((fid, cam.label, profile, note, None))
                    except Exception as e:  # noqa: BLE001
                        results.append((fid, None, None, None, str(e)))
                return results
            finally:
                pool.close()

        def done(results):
            self.project_checkpoint()
            lines = []
            for fid, label, profile, note, err in results:
                f = self.project.footage.get(fid)
                if f is None:
                    continue
                if err:
                    lines.append(f"{f.name}: {err}")
                    continue
                f.camera = label
                if f.log_profile != before.get(fid):
                    lines.append(f"{f.name}: kept {profile_label(f.log_profile)} (chosen while detecting)")
                    continue
                if profile is not None:
                    f.log_profile = self._official_profile(profile)
                f.log_note, f.log_pending = (f"Detected {note}" if profile else note), False
                lines.append(f"{f.name}: {f.camera or 'unknown camera'} → {profile_label(f.log_profile)}"
                             + ("" if profile else f"  ({note})"))
            self.project.dirty = True
            self.refresh_all(captions=False)
            self.inspector.set_clip(self._inspected())
            self.viewer.refresh()
            if len(lines) == 1:
                QTimer.singleShot(0, lambda: self.lbl_status.setText(lines[0]))
            elif lines:
                QMessageBox.information(self, APP_NAME, "Camera detection:\n\n" + "\n".join(lines[:25])
                                        + (f"\n… and {len(lines) - 25} more" if len(lines) > 25 else ""))

        self.tasks.submit(name, job, done, self._task_error(name))

    # ================================================================== LUTs
    def import_lut(self, target: str, ids: Optional[list[str]] = None):
        """Import .cube files into the LUT library. target: "library", "log" (conversion for footage `ids`, else
        the footage of every selected clip) or "look" (creative LUT on every selected clip)."""
        paths, _ = QFileDialog.getOpenFileNames(self, "Import LUT", self.qs.value("lut_import_dir", ""),
                                                "Cube LUT or .zip (*.cube *.zip);;All files (*)")
        if not paths:
            return
        imported = self._import_lut_files(paths)
        if not imported:
            return
        lut = imported[0]
        clip = self._inspected()
        targets = self.inspector.graded if clip is not None else []   # the shown clip and the rest of the selection
        if target == "log":
            ids = ids or list(dict.fromkeys(c.footage_id for c in targets))
            if ids:
                self.set_log_profile(ids, lut_profile(lut))
        elif target == "look" and targets:
            self.project_checkpoint()
            for c in targets:
                c.grade.look = lut
            self.inspector.set_clip(clip)
            self.viewer.refresh()
        self._update_title()
        self.lbl_status.setText(f"Imported {len(imported)} LUT(s)" if len(imported) > 1
                                else f"Imported LUT {lutlib.lut_name(lut)}")

    def _import_lut_files(self, paths: list[str]) -> list[str]:
        """Copy .cube files (or those inside .zip archives / folders) into the LUT library."""
        if os.path.isfile(paths[0]):
            self.qs.setValue("lut_import_dir", os.path.dirname(paths[0]))
        imported, errors = [], []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                try:
                    imported += lutlib.import_any(path)
                except (lutlib.LutError, OSError, zipfile.BadZipFile) as e:
                    errors.append(f"{os.path.basename(path)}: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        if errors:
            QMessageBox.warning(self, APP_NAME, "Some LUTs could not be imported:\n\n" + "\n".join(errors[:12]))
        self.inspector.reload_luts()
        if imported:   # LUT-only profiles (DJI D-Log M...) may have just got their official LUT
            self.bin.refresh()
            self.viewer.refresh()
        return imported

    def _official_luts(self) -> dict:
        """{built-in log profile: official LUT path} chosen in the official-LUT dialog (existing files only)."""
        out = {}
        for e in lutlib.OFFICIAL_LUTS:
            path = self.qs.value(f"official_lut/{e.profile}", "", type=str)
            if path and os.path.exists(path):
                out[e.profile] = path
        return out

    def _official_profile(self, profile: str) -> str:
        """The official LUT to use instead of a detected built-in profile, when the user set one up."""
        if not self.qs.value("official_lut_auto", True, type=bool) or profile in lutlib.LUT_ONLY:
            return profile   # LUT-only profiles find their official LUT by themselves
        path = self._official_luts().get(profile)
        return lut_profile(path) if path else profile

    def official_luts_dialog(self):
        dlg = OfficialLutDialog(self._official_luts(), self.qs.value("official_lut_auto", True, type=bool),
                                self._import_lut_files, self)
        accepted = dlg.exec() == OfficialLutDialog.Accepted
        self.inspector.reload_luts()   # the dialog may have downloaded LUTs either way
        self.bin.refresh()
        self.viewer.refresh()
        if not accepted:
            return
        mapping = dlg.mapping()
        for e in lutlib.OFFICIAL_LUTS:
            self.qs.setValue(f"official_lut/{e.profile}", mapping.get(e.profile, ""))
        self.qs.setValue("official_lut_auto", dlg.chk_auto.isChecked())
        lutlib.set_official_choices(self._official_luts())
        self.bin.refresh()
        self.viewer.refresh()
        if not dlg.chk_apply.isChecked():
            return
        # Footage converted with a built-in profile that now has an official LUT.
        changed = 0
        self.project_checkpoint()
        for f in self.project.footage.values():
            if f.log_profile in mapping and f.log_profile not in lutlib.LUT_ONLY:
                f.log_profile = lut_profile(mapping[f.log_profile])
                changed += 1
        self.project.dirty = changed > 0 or self.project.dirty
        self.refresh_all(captions=False)
        self.inspector.set_clip(self._inspected())
        self.viewer.refresh()
        self._update_title()
        self.lbl_status.setText(f"Official LUTs: {changed} footage file(s) switched" if changed
                                else "Official LUTs saved")

    # ================================================================== cache
    def _init_cache_root(self):
        path = self.qs.value("cache_dir", "", type=str)
        if not path:
            return
        set_cache_root(path)
        try:
            cache_root()
        except OSError:
            set_cache_root(None)   # the drive may just be disconnected: let the user decide
            QTimer.singleShot(0, lambda: self._cache_root_missing(path))

    def _cache_root_missing(self, path: str):
        box = QMessageBox(QMessageBox.Warning, APP_NAME,
                          f"The cache folder is not available:\n{path}\n\n"
                          f"The default cache folder is used for now:\n{cache_root()}", parent=self)
        b_choose = box.addButton("Choose another folder…", QMessageBox.AcceptRole)
        b_default = box.addButton("Always use the default", QMessageBox.DestructiveRole)
        box.addButton("Ask again next time", QMessageBox.RejectRole)
        box.setInformativeText("Keep asking if it is on a drive that is not connected right now.")
        box.exec()
        if box.clickedButton() is b_default:
            self.qs.remove("cache_dir")
            self.lbl_status.setText(f"Cache folder: {cache_root()}")
        elif box.clickedButton() is b_choose:
            new = QFileDialog.getExistingDirectory(self, "Choose cache folder", str(cache_root()))
            if not new:
                return
            if os.path.normcase(str(Path(new).resolve())) == os.path.normcase(str(cache_root().resolve())):
                self.qs.setValue("cache_dir", new)   # the folder already in use: just remember it
                self.lbl_status.setText(f"Cache folder: {cache_root()}")
            elif not self._change_cache_folder(new):
                self._cache_root_missing(path)

    def cache_dialog(self):
        CacheDialog(self._change_cache_folder, self._clear_cache, self).exec()

    def _change_cache_folder(self, path: str) -> bool:
        if self.tasks.busy:
            QMessageBox.information(self, APP_NAME, "Wait for the background tasks to finish (or cancel them) "
                                                    "before changing the cache folder.")
            return False
        old = cache_root()
        new = Path(path) if path else default_cache_root()
        if os.path.normcase(str(new.resolve())) == os.path.normcase(str(old.resolve())):
            return False
        if any(new.resolve().is_relative_to((old / kind).resolve()) for kind in CACHE_KINDS):
            QMessageBox.warning(self, APP_NAME, "Choose a folder outside the current cache's sub-folders.")
            return False
        try:
            new.mkdir(parents=True, exist_ok=True)
            test = new / ".kangedit_write_test"
            test.write_bytes(b"")
            test.unlink()
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"This folder cannot be used for the cache:\n{e}")
            return False
        copyable = folder_size(old / "proxies") + folder_size(old / "analysis")
        copy_files = False
        if copyable:
            r = QMessageBox.question(
                self, APP_NAME, f"Copy the files from the current cache folder to the new one?\n\n"
                                f"From: {old}\nTo: {new}\n\nProxies and analysis, {fmt_bytes(copyable)}. The old folder "
                                f"is left as it is (delete it yourself when you no longer need it). If you don't copy, "
                                f"they are recreated in the new folder when needed.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                return False
            copy_files = r == QMessageBox.Yes
        set_cache_root(path or None)
        if path:
            self.qs.setValue("cache_dir", path)
        else:
            self.qs.remove("cache_dir")
        self._schedule_audio()
        if copy_files:
            self._copy_cache(old, cache_root())
        self.lbl_status.setText(f"Cache folder: {cache_root()}")
        return True

    def _copy_cache(self, old: Path, new: Path):
        """Copy proxies and analysis from `old` into the cache folder `new` (in the background) and point the
        project's footage at the copies. The old folder is not touched."""

        def job(progress, cancel):
            files = [(kind, e) for kind in ("proxies", "analysis") if (old / kind).is_dir()
                     for e in os.scandir(old / kind) if e.is_file()]
            total = sum(e.stat().st_size for _, e in files) or 1
            done_bytes, copied = 0, 0
            for kind, e in files:
                if cancel.is_set():
                    break
                size = e.stat().st_size
                dst = new / kind / e.name
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not (dst.exists() and dst.stat().st_size == size):   # same cache key -> same content
                        part = dst.with_name(dst.name + ".part")
                        shutil.copy2(e.path, part)
                        os.replace(part, dst)
                    copied += 1
                except OSError:
                    pass   # unreadable: recreated when needed
                done_bytes += size
                progress(done_bytes / total, f"Copying cache {copied}/{len(files)}")
            return copied

        def finish(_=None):
            self.viewer.drop_decoders()
            self._relink_cache_paths(old)
            self.refresh_all(captions=False)
            self.viewer.refresh()

        def failed(msg: str):
            finish()
            self._task_error("Copying the cache")(msg)

        self.tasks.submit("Copying cache", job, finish, failed)

    def _offer_legacy_cache(self):
        """Caches used to live in the per-user data folder; offer once to copy them next to the app."""
        if self.qs.value("cache_dir", "", type=str) or self.qs.value("legacy_cache_offered", False, type=bool):
            return
        old, new = legacy_cache_root(), cache_root()
        if not old.is_dir() or os.path.normcase(str(old.resolve())) == os.path.normcase(str(new.resolve())):
            return
        size = folder_size(old / "proxies") + folder_size(old / "analysis")
        self.qs.setValue("legacy_cache_offered", True)
        if not size:
            return
        r = QMessageBox.question(
            self, APP_NAME, f"The cache folder is now next to the app:\n{new}\n\nCopy the proxies and analysis from "
                            f"the previous cache folder ({fmt_bytes(size)})?\n{old}\n\nThe old folder is left as it is.")
        if r == QMessageBox.Yes:
            self._copy_cache(old, new)

    def _relink_cache_paths(self, old: Path):
        """Point footage at its cache files in the current cache folder after they were moved from `old`."""
        old_key = os.path.normcase(str(old.resolve()))
        for f in self.project.footage.values():
            for attr in ("proxy_path", "analysis_path"):
                p = getattr(f, attr)
                if not p:
                    continue
                pp = Path(p)
                if os.path.normcase(str(pp.parent.parent.resolve())) != old_key:
                    continue
                cand = cache_root() / pp.parent.name / pp.name
                if cand.exists():
                    setattr(f, attr, str(cand))
                    self.project.dirty = True

    def _clear_cache(self, kinds: list[str]):
        if self.tasks.busy:
            QMessageBox.information(self, APP_NAME, "Wait for the background tasks to finish (or cancel them) "
                                                    "before clearing the cache.")
            return
        if "proxies" in kinds:
            self.viewer.drop_decoders()
        if "tmp" in kinds:
            self.viewer.stop()
            self.viewer.set_audio(None)   # releases the preview audio file
        freed, locked = clear_cache(kinds)
        for f in self.project.footage.values():
            if f.proxy_path and not os.path.exists(f.proxy_path):
                f.proxy_path = None
            if f.analysis_path and not os.path.exists(f.analysis_path):
                f.analysis_path = None
        self.refresh_all(captions=False)
        if "tmp" in kinds:
            self._schedule_audio()
        self.lbl_status.setText(f"Cleared {fmt_bytes(freed)} from the cache"
                                + (f" ({locked} item(s) in use were kept)" if locked else ""))

    # ================================================================== timeline editing
    def thumbnail_dialog(self):
        from .. import thumbnail as th
        from .thumbnail_dialog import ThumbnailDialog
        if not self.project.clips:
            QMessageBox.information(self, APP_NAME, "Make an edit first (✨ Auto Edit or clips on the timeline).")
            return
        idx, off = self.project.clip_at(self.viewer.timeline_time())
        here = None
        if idx is not None:
            c = self.project.clips[idx]
            here = th.Still(c.footage_id, c.src_at(off), 0.0, c.id)
        ThumbnailDialog(self.project, self.analyses, here, self, self.project.settings.use_proxies).exec()

    def chapters_dialog(self):
        from .chapters_dialog import ChaptersDialog
        if not self.project.clips:
            QMessageBox.information(self, APP_NAME, "Make an edit first (✨ Auto Edit or clips on the timeline).")
            return
        dlg = ChaptersDialog(self.project, self)
        dlg.exec()
        if dlg.changed:
            self.refresh_all()

    def _timeline_seek(self, t: float):
        """The timeline was clicked: the viewer shows the edit again (it may be in Source mode)."""
        if self.viewer.mode == "source":
            self.viewer.show_timeline()
        self.viewer.seek(t)

    def _viewer_mode_changed(self, mode: str):
        if mode == "source":
            f = self.project.footage.get(self.viewer.source_fid)
            self.lbl_status.setText(f"Playing {f.name if f else 'the file'} on its own: mark the part to use (I / O, "
                                    "or select its words), then Insert at playhead or Append. Click the timeline "
                                    "(or Timeline) to go back to the edit.")
        else:
            self.lbl_status.setText("Showing the edit")

    def _source_mark(self, kind: str):
        if self.viewer.mode == "source":
            self.viewer.mark(kind)
        else:
            self.lbl_status.setText("Mark in / out are for a file played on its own: double-click it in Media")

    def use_source_part(self, fid: str, a: float, b: float, how: str):
        """Put the part of a file marked in the viewer's Source mode on the timeline: at the timeline's playhead
        (between the clips nearest to it) or at the end. The viewer stays on the file, to mark the next part."""
        f = self.project.footage.get(fid)
        if f is None:
            self.lbl_status.setText("That file is no longer in the project")
            return
        if b - a < 0.05:
            return
        clip = TimelineClip(fid, round(a, 3), round(min(b, f.duration), 3))
        t = self.viewer.timeline_time()
        starts = self.project.clip_starts()
        if how == "insert" and self.project.clips:
            i = min(range(len(starts) + 1),
                    key=lambda k: abs((starts[k] if k < len(starts) else self.project.duration) - t))
        else:
            i = len(self.project.clips)
        self.project_checkpoint()
        self.project.clips.insert(i, clip)
        self._after_structure_change()
        self.tl.selected = clip.id
        self.tl.update()
        start = self.project.clip_starts()[i]
        self.viewer.set_timeline_time(start + clip.duration)     # the next part goes after this one
        self.tl.set_playhead(start + clip.duration)
        self.lbl_status.setText(f"{f.name} {fmt_time(a, frames=False)}–{fmt_time(b, frames=False)} put on the "
                                f"timeline at {fmt_time(start, frames=False)} ({clip.duration:.1f} s)")

    def _refresh_edit_text(self):
        """The edit's transcript follows the timeline while its tab shows."""
        if self.right_tabs.shows(self.edit_text):
            self.edit_text.refresh()
            self.edit_text.show_time(self.viewer.timeline_time())

    def _reorder_by_transcript(self, ids: list):
        """Clips dragged into another order in the edit's transcript: the timeline follows."""
        by_id = {c.id: c for c in self.project.clips}
        if sorted(ids) != sorted(by_id):
            self._refresh_edit_text()
            return
        self.project_checkpoint()
        self.project.clips = [by_id[i] for i in ids]
        multicam.follow(self.project)
        self._after_structure_change()
        self.lbl_status.setText("Clips reordered from the transcript")

    def _transcribe_edit(self):
        todo = list(dict.fromkeys(c.footage_id for c in self.project.clips
                                  if (f := self.project.footage.get(c.footage_id)) is not None and f.has_audio
                                  and not f.is_image and f.transcript is None))
        todo += [v.id for x in self.project.voice_sounds()
                 if (v := self.project.voice_for(x.path)).transcript is None and v.id not in todo]
        if todo:
            self.queue_transcription(todo)

    def _after_structure_change(self):
        self.refresh_all()
        self._schedule_audio()

    def _on_select(self, clip_id: str):
        self.inspector.set_clip(self._inspected())
        self._update_slip_text()
        if clip_id:
            self.right_tabs.setCurrentWidget(self.inspector)

    def _update_slip_text(self, slipping: bool = False):
        """In Slip mode (or while a clip is slipped), the transcript of the selected clip's footage, under the
        timeline's buttons."""
        panel = self.timeline.slip_text
        if not slipping:
            panel.show_length(None)                   # (a trim ended)
        show = slipping or self.timeline.btn_slip.isChecked()
        panel.setVisible(show)
        if not show:
            self._slip_shown = None
            return
        clip = self.selected_clip()
        f = self.project.footage.get(clip.footage_id) if clip is not None else None
        key = (clip.id if clip else "", f.id if f else "", len(f.transcript) if f and f.transcript else -1)
        if key != self._slip_shown:
            self._slip_shown = key
            panel.show_clip(clip, f)
        elif clip is not None:
            panel.follow(clip.id, clip.src_in, clip.src_out)

    def _slip_follow(self, clip_id: str, _src_t: float):
        """A clip is slipped or trimmed: its transcript follows (shown for the drag, also in Move mode)."""
        i = self.project.index_of(clip_id)
        if i < 0:
            return
        if self.timeline.slip_text.clip_id != clip_id or self.timeline.slip_text.isHidden():
            self.tl.selected = clip_id
            self._update_slip_text(slipping=True)
        c = self.project.clips[i]
        self.timeline.slip_text.follow(clip_id, c.src_in, c.src_out)
        d = self.tl._drag or {}
        trimming = d.get("clip") == clip_id and d.get("kind") in ("trim_l", "trim_r")
        self.timeline.slip_text.show_length(c.duration if trimming else None, d.get("dur0"))

    def slip_clip_to(self, clip_id: str, src_in: float):
        """Slip a clip to start at source time src_in (a sentence of its transcript), keeping its length."""
        i = self.project.index_of(clip_id)
        if i < 0:
            return
        c = self.project.clips[i]
        f = self.project.footage.get(c.footage_id)
        if f is None:
            return
        length = c.src_dur
        a = round(min(max(0.0, src_in), max(0.0, f.duration - length)), 3)
        if abs(a - c.src_in) < 1e-3:
            return
        self.project_checkpoint()
        c.src_in, c.src_out = a, round(a + length, 3)
        multicam.follow(self.project)
        self._on_timeline_edited()
        self._on_timeline_finished()
        self._timeline_seek(self.project.clip_starts()[i])
        self.lbl_status.setText(f"Slipped the clip to {fmt_time(a, frames=False)} in {f.name}")

    def _on_timeline_edited(self):
        if multicam.follow(self.project):      # the angles follow a trim / slip while it is dragged
            self.tl.update()
        self.viewer.invalidate_audio()
        self.captions = build_captions(self.project)
        self.tl.captions = self.captions
        self.viewer.captions = self.captions
        self.viewer.refresh()
        self.inspector.set_clip(self._inspected())

    def _on_timeline_finished(self):
        self.measure_shake()          # a stabilized clip trimmed or slipped onto a part not measured yet
        self.refresh_all()
        self._schedule_audio()

    def _ask_speed(self, cur: float) -> Optional[float]:
        v, ok = QInputDialog.getDouble(self, "Clip speed",
                                       f"Speed ({MIN_SPEED:g}× to {MAX_SPEED:g}×; below 1 is slow motion, "
                                       "above 1 fast motion):", cur, MIN_SPEED, MAX_SPEED, 2)
        return clamp_speed(v) if ok else None

    def set_speed(self, targets: list, action: str):
        """Slow / fast motion and speed ramps on main-track clips: 'speed:<x>', 'ramp:<key>' or 'speed_custom'
        (asks). The timeline closes up around the new lengths; linked camera angles follow."""
        if not targets:
            return
        if action == "speed_custom":
            v = self._ask_speed(targets[0].speed)
            if v is None:
                return
            action = f"speed:{v}"
        key, val = action.split(":", 1)
        self.project_checkpoint()
        for x in targets:
            f = self.project.footage.get(x.footage_id)
            if f is not None and f.is_image:
                continue                       # a still has no motion to slow down
            if key == "speed":
                x.speed = clamp_speed(float(val))
            elif key == "audio":
                x.audio_mode = sound_mode(val)
            else:
                x.ramp = val if val in RAMPS else ""
        self._after_structure_change()
        c = targets[0]
        if key == "audio":
            self.lbl_status.setText(f"Sound: {SOUND_MODES[c.audio_mode][0]} — {len(targets)} clip(s)")
            return
        what = speed_label(c.speed, c.ramp) or "normal speed"
        self.lbl_status.setText(f"Speed: {what} — {len(targets)} clip(s); "
                                f"{'the ramp' if c.ramp else 'the clip'} now runs {c.duration:.1f} s.")

    def split_at_playhead(self):
        idx, off = self.project.clip_at(self.viewer.timeline_time())
        if idx is None:
            return
        c = self.project.clips[idx]
        if off < 0.1 or off > c.duration - 0.1:
            return
        self.project_checkpoint()
        cut = c.src_at(off)
        new = TimelineClip.from_dict(c.to_dict())
        new.id = new_id()
        new.src_in = cut
        c.src_out = cut
        self.project.clips.insert(idx + 1, new)
        self._after_structure_change()
        if c.ramp:
            self.lbl_status.setText("Split a speed-ramped clip: each part now plays the whole ramp.")

    def clip_action(self, action: str, clip_id: str):
        if action == "select_all":
            self.tl.select_all()
            return
        if action.startswith("sound_"):
            self._sound_action(action, clip_id)
            return
        if action.startswith(("transition:", "split:")):
            key, val = action.split(":", 1)
            ids = set(self.tl.selected_ids()) if clip_id in self.tl.selected_ids() else {clip_id}
            self.project_checkpoint()
            for x in self.project.clips:
                if x.id in ids:
                    if key == "transition":
                        x.transition = val
                        if val:
                            x.audio_split = 0.0
                    else:
                        x.audio_split = float(val)
                        if x.audio_split:
                            x.transition = ""
            self._on_timeline_edited()
            self.tl.refresh()
            self.inspector.set_clip(self._inspected())
            self._schedule_audio()
            self._update_title()
            return
        if action.startswith("rotate:") and not clip_id:
            return
        i = self.project.index_of(clip_id) if clip_id else -1
        if i < 0:
            if action == "alt_moment":
                self._replace_moments([])
            return
        c = self.project.clips[i]
        # Act on the whole selection when the clip is part of it.
        sel = self.tl.selected_ids()
        targets = [x for x in self.project.clips if x.id in sel] if clip_id in sel else [c]
        if action == "delete":
            self.project_checkpoint()
            gone = {x.id for x in targets}
            self.project.clips = [x for x in self.project.clips if x.id not in gone]
            layer_gone = set(self.tl.selected_overlay_ids()) if clip_id in sel else set()
            if layer_gone:                    # (Ctrl+A) the selected clips on the video layers go too
                self.project.overlays = [o for o in self.project.overlays if o.id not in layer_gone]
                self.tl.select_overlay("")
            self.tl.selected = ""
            self._after_structure_change()
        elif action == "duplicate":
            self.project_checkpoint()
            for x in reversed(targets):
                d = TimelineClip.from_dict(x.to_dict())
                d.id = new_id()
                self.project.clips.insert(self.project.index_of(x.id) + 1, d)
            self._after_structure_change()
        elif action == "split":
            starts = self.project.clip_starts()
            if not (starts[i] < self.viewer.timeline_time() < starts[i] + c.duration):
                self._timeline_seek(starts[i] + c.duration / 2)
            self.split_at_playhead()
        elif action == "alt_moment":
            self._replace_moments(targets)
        elif action == "transcribe_force":
            self.transcribe_again(list(dict.fromkeys(x.footage_id for x in targets)))
        elif action == "face_zoom":
            self._face_zoom(targets)
        elif action.startswith(("speed:", "ramp:", "audio:")) or action == "speed_custom":
            self.set_speed(targets, action)
        elif action.startswith("rotate:"):
            self.project_checkpoint()
            for x in targets:
                x.rotation = wrap_angle(x.rotation + float(action.split(":", 1)[1]))
            self.inspector.set_clip(self._inspected())
            self.tl.update()
            self.viewer.refresh()
            self._update_title()
        elif action == "auto_color":
            self._auto_color(targets)
        elif action == "reset_color":
            self.project_checkpoint()
            for x in targets:
                x.grade = Grade()
            self.inspector.set_clip(self._inspected())
            self.viewer.refresh()
            self._update_title()
        elif action == "copy_grade":
            if len(targets) > 1:
                self.project_checkpoint()
                for x in targets:
                    x.grade = Grade.from_dict(c.grade.to_dict())
                self.viewer.refresh()
                self._update_title()
            else:
                self.tl.selected = clip_id
                self.inspector.set_clip(c)
                self.inspector_request("copy_grade_all")

    # ================================================================== sound effects
    def add_sounds(self, paths: list[str], t: Optional[float] = None, names: Optional[list[str]] = None):
        """Put audio files on the SFX layer at timeline time t (default: the playhead), one after another."""
        t = self.viewer.timeline_time() if t is None else t
        added, bad = [], []
        for k, path in enumerate(paths):
            if os.path.normcase(os.path.dirname(os.path.abspath(path))) != os.path.normcase(str(sfx_dir())):
                try:
                    path = sfx.import_file(path)     # keep projects working if the original moves
                except OSError as e:
                    bad.append(f"{os.path.basename(path)}: {e}")
                    continue
            dur = sfx.duration(path)
            if dur <= 0:
                bad.append(f"{os.path.basename(path)}: not a readable audio file")
                continue
            name = names[k] if names and k < len(names) else os.path.splitext(os.path.basename(path))[0]
            added.append(SoundClip(path, round(t, 3), 0.0, round(dur, 3), 1.0, name))
            t += dur
        if bad:
            QMessageBox.warning(self, APP_NAME, "Some sounds could not be added:\n\n" + "\n".join(bad[:10]))
        if not added:
            return
        self.project_checkpoint()
        self.project.sounds += added
        self.sfx_panel.refresh_library()
        self.tl.selected_sound = added[-1].id
        self.tl.refresh()
        self._sounds_changed()
        self.lbl_status.setText(f"Added sound “{added[0].name}” at {fmt_time(added[0].start, frames=False)}"
                                if len(added) == 1 else f"Added {len(added)} sounds")

    MUSIC_VOLUME = 0.5

    def add_music(self, path: str, title: str, url: str = ""):
        """Lay a music track under the whole edit from its start (cut to the edit's length), quieter than the
        clips' own sound, and copy the NCS credit for the video's description to the clipboard."""
        dur = sfx.duration(path)
        if dur <= 0:
            QMessageBox.warning(self, APP_NAME, f"“{title}” is not a readable audio file.")
            return
        end = min(dur, self.project.duration) if self.project.duration > 0 else dur
        x = SoundClip(path, 0.0, 0.0, round(end, 3), self.MUSIC_VOLUME, title, duck=True)
        self.project_checkpoint()
        self.project.sounds.append(x)
        self.tl.selected_sound = x.id
        self.tl.refresh()
        self._sounds_changed()
        QApplication.clipboard().setText(music.credit(title, url))
        self.lbl_status.setText(f"Added music “{title}” under the edit ({fmt_time(end, frames=False)}), auto-ducked "
                                "under the clips' sound. The NCS credit is on the clipboard: paste it into the "
                                "video's description.")

    def _sync_sound_panel(self):
        sid = self.tl.selected_sound
        self.sfx_panel.set_selected_sound(next((s for s in self.project.sounds if s.id == sid), None) if sid else None)

    def _selected_sound_volume(self, volume: float):
        if self.tl.selected_sound:
            self._sound_action(f"sound_volume:{volume}", self.tl.selected_sound)
            self._schedule_audio(150, keep_old=True)   # re-mix quickly so a volume drag is heard live

    def _sounds_changed(self, keep_old: bool = False):
        self._sync_sound_panel()
        self.project.dirty = True
        self._schedule_audio(keep_old=keep_old)
        self._update_title()

    def _sound_action(self, action: str, sound_id: str):
        x = next((s for s in self.project.sounds if s.id == sound_id), None)
        if x is None:
            return
        if action == "sound_preview":
            self.sfx_panel.preview_file(x.path)
            return
        if action == "sound_fit":
            self._fit_sound(x)
            return
        # one undo step per volume / fade drag, not per slider tick
        key = action.split(":", 1)[0]
        self.project_checkpoint(f"sound:{sound_id}:{key}"
                                if key in ("sound_volume", "sound_fade_in", "sound_fade_out") else "")
        if action == "sound_delete":
            self.project.sounds.remove(x)
            self.tl.selected_sound = ""
        elif action == "sound_to_playhead":
            x.start = round(self.viewer.timeline_time(), 3)
        elif action.startswith("sound_volume:"):
            x.volume = float(action.split(":", 1)[1])
        elif action.startswith("sound_fade_in:"):
            x.fade_in = max(0.0, float(action.split(":", 1)[1]))
        elif action.startswith("sound_fade_out:"):
            x.fade_out = max(0.0, float(action.split(":", 1)[1]))
        elif action.startswith("sound_caption:"):
            x.caption = action.endswith(":1")
            if x.caption:
                v = self.project.voice_for(x.path)
                if v.transcript is None:
                    self.queue_transcription([v.id])
                else:
                    self.lbl_status.setText(f"“{x.name or os.path.basename(x.path)}” is captioned (voice-over)")
            self.update_captions()
        elif action.startswith("sound_duck:"):
            x.duck = action.endswith(":1")
            if x.duck:
                self.lbl_status.setText(f"“{x.name or os.path.basename(x.path)}” is turned down "
                                        f"{self.project.settings.duck_db:g} dB while the clips have sound")
        self.tl.refresh()
        self._sounds_changed(keep_old=key in ("sound_volume", "sound_fade_in", "sound_fade_out", "sound_duck"))

    def _duck_depth(self, db: float):
        self.project_checkpoint("duck_depth")
        self.project.settings.duck_db = float(db)
        self._schedule_audio(300, keep_old=True)
        self._update_title()

    def _fit_sound(self, x: SoundClip):
        """Lay the sound over the whole timeline: repeated with crossfades, or cut with a fade out."""
        total = self.project.duration
        file_len = sfx.duration(x.path)
        if total <= 0 or file_len <= 0:
            self.lbl_status.setText("Nothing to fit: the timeline is empty or the sound can't be read.")
            return
        copies = sfx.fit_to_timeline(x, file_len, total)
        self.project_checkpoint()
        # earlier copies of the same file would play over the fitted ones
        self.project.sounds = [s for s in self.project.sounds if s.path != x.path] + copies
        self.tl.selected_sound = copies[0].id
        self.tl.refresh()
        self._sounds_changed()
        label = x.name or os.path.basename(x.path)
        y = copies[0]
        self.lbl_status.setText(f"“{label}” fits the timeline: it loops {y.repeats()}× with crossfades, on one row"
                                if y.looping else f"“{label}” fits the timeline, with a fade out at the end")

    # ================================================================== video layers (overlays)
    def _whole_clip(self, fid: str) -> TimelineClip:
        """A main-track clip of the whole footage (DEFAULT_LENGTH seconds of an image)."""
        f = self.project.footage[fid]
        return TimelineClip(fid, 0.0, images.DEFAULT_LENGTH if f.is_image else f.duration)

    def _footage_dropped(self, ids: list[str], t: float, track: int):
        ids = [fid for fid in ids if fid in self.project.footage]
        if not ids:
            return
        if track >= 0:
            self.add_overlays(ids, t, track)
            return
        # Main track: in at the cut nearest the drop point.
        starts = self.project.clip_starts()
        at = sum(1 for s, c in zip(starts, self.project.clips) if s + c.duration / 2 < t)
        self.project_checkpoint()
        new = [self._whole_clip(fid) for fid in ids]
        self.project.clips[at:at] = new
        self.tl.selected = new[0].id
        self._after_structure_change()
        self._timeline_seek(self.project.clip_starts()[at] + 0.01)
        self.lbl_status.setText(f"Added {len(new)} clip(s) to the timeline")

    def _overlay_length(self, f: Footage) -> tuple[float, float]:
        """Source in / out of a new overlay: an image's usual length, a short video whole, else its best moment."""
        if f.is_image:
            return 0.0, images.DEFAULT_LENGTH
        if f.duration <= 15.0:
            return 0.0, f.duration
        best = self._best_for(f.id, relaxed=True) if f.id in self.analyses else None
        return (best.src_in, best.src_out) if best is not None else (0.0, min(f.duration, 10.0))

    def _free_track(self, start: float, end: float, first: int = 0) -> int:
        """The lowest layer from `first` up with nothing between start and end (a new layer if all are busy)."""
        for k in range(first, MAX_LAYERS):
            if not any(c.track == k and c.start < end - 1e-3 and start < c.end - 1e-3 for c in self.project.overlays):
                return k
        return min(MAX_LAYERS - 1, first)

    def add_overlays(self, ids: list[str], t: Optional[float] = None, track: Optional[int] = None):
        """Put footage / images on video layers at timeline time t (default: the playhead): videos as picture in
        picture in a corner, images in the middle at half size."""
        if not self.project.clips:
            QMessageBox.information(self, APP_NAME, "Put something on the main track first: video layers show on "
                                                    "top of it.")
            return
        t = self.viewer.timeline_time() if t is None else t
        added = []
        self.project_checkpoint()
        for fid in ids:
            f = self.project.footage.get(fid)
            if f is None:
                continue
            src_in, src_out = self._overlay_length(f)
            k = self._free_track(t, t + src_out - src_in, track or 0)
            x, y, scale = PLACEMENTS["center" if f.is_image else "br"]
            c = OverlayClip(fid, round(t, 3), src_in, src_out, k, x, y, scale)
            self.project.overlays.append(c)
            added.append(c)
        if not added:
            return
        top = max(c.track for c in self.project.overlays)
        self.project.settings.overlay_tracks = max(self.project.settings.overlay_tracks, top + 1)
        self.tl.refresh()
        self.tl.select_overlay(added[-1].id)
        self._overlays_changed()
        self._timeline_seek(min(self.project.duration - 1e-3, added[-1].start + 0.01))
        self.lbl_status.setText(f"Added {len(added)} clip(s) on video layer V{added[-1].track + 2}. Drag it in the "
                                "viewer to place it, the Layer tab sizes and rotates it.")

    def _overlay(self, oid: str) -> Optional[OverlayClip]:
        return next((c for c in self.project.overlays if c.id == oid), None)

    def _on_overlay_selected(self, oid: str):
        self.layer_panel.set_clip(self._overlay(oid))
        self.inspector.set_clip(self._inspected())
        self.viewer.canvas.selected_overlay = oid
        self.viewer.canvas.update()
        if oid:
            self.show_refine(self.layer_panel)

    def _overlay_moved(self):
        """An overlay is being dragged on the timeline."""
        self.layer_panel.set_clip(self.selected_overlay())
        self.viewer.refresh()

    def _overlays_changed(self):
        self.project.dirty = True
        self.layer_panel.set_clip(self.selected_overlay())
        self.tl.refresh()
        self.bin.refresh()
        self.viewer.refresh()
        self._schedule_audio()
        self._update_title()

    def _layer_edited(self):
        """The Layer tab changed the selected overlay."""
        self.project.dirty = True
        self.tl.refresh()
        self.viewer.refresh()
        c = self.selected_overlay()
        if c is not None and self.project.footage.get(c.footage_id) and self.project.footage[c.footage_id].has_audio:
            self._schedule_audio()
        self._update_title()

    def _overlay_dragged(self, oid: str, x: float, y: float):
        c = self._overlay(oid)
        if c is None:
            return
        self.project_checkpoint(f"layer:{oid}:drag")
        c.x, c.y = round(x, 4), round(y, 4)
        self.layer_panel.set_clip(self.selected_overlay())
        self.viewer.refresh()

    def _overlay_scaled(self, oid: str, factor: float):
        c = self._overlay(oid)
        if c is None:
            return
        self.project_checkpoint(f"layer:{oid}:scale")
        c.scale = round(max(0.02, min(3.0, c.scale * factor)), 4)
        self.layer_panel.set_clip(c)
        self.viewer.refresh()
        self._update_title()

    def _screen_colour(self, c) -> Optional[str]:
        """The green / blue screen colour of an overlay clip's picture (a frame from the middle of it)."""
        f = self.project.footage.get(c.footage_id)
        if f is None:
            return None
        t = c.src_in + c.src_dur / 2
        a = self.analyses.get(f.id)
        if a is not None and len(a.thumbs):
            k = int(np.clip(np.searchsorted(a.thumb_times, t), 0, len(a.thumbs) - 1))
            return keying.guess(np.asarray(a.thumbs[k]))
        try:
            dec = FrameDecoder(f.proxy_path if f.has_proxy() else f.path, f.rotation, f.color_space, 360)
            try:
                img = dec.frame_at(t)
            finally:
                dec.close()
            return keying.guess(img) if img is not None else None
        except Exception:  # noqa: BLE001 - unreadable: the caller uses chroma green
            return None

    def _overlay_action(self, action: str, oid: str):
        if action == "link_toggle":
            self.toggle_link(oid)
            return
        c = self._overlay(oid)
        if c is None:
            return
        if action == "ov_edit":
            self.tl.select_overlay(oid)
            self.show_refine(self.layer_panel)
            if not (c.start <= self.viewer.timeline_time() < c.end):
                self._timeline_seek(c.start + 0.01)
            return
        if action == "ov_auto_color":                 # in the background; one undo step when it's done
            self._auto_color([c])
            return
        if action == "ov_key_auto":                   # chroma key: the screen colour from the picture's edges
            self.project_checkpoint()
            c.key_color = self._screen_colour(c) or keying.PRESETS["green"]
            self.layer_panel.set_clip(c)
            self.viewer.refresh()
            self.lbl_status.setText(f"Chroma key: {c.key_color} is made see-through")
            return
        if action == "ov_color":
            self.tl.select_overlay(oid)
            self.show_refine(self.inspector)
            return
        key = action.split(":", 1)[0]
        self.project_checkpoint(f"layer:{oid}:{key}" if key in ("ov_fade_in", "ov_fade_out") else "")
        if action == "ov_delete":
            self.project.overlays.remove(c)
            if self.tl.selected_overlay == oid:
                self.tl.select_overlay("")
        elif action == "ov_to_playhead":
            c.start = round(self.viewer.timeline_time(), 3)
        elif action == "ov_duplicate":
            d = OverlayClip.from_dict(c.to_dict())
            d.id, d.start = new_id(), c.end
            self.project.overlays.append(d)
            self.tl.select_overlay(d.id)
        elif action == "ov_split":
            cut = self.viewer.timeline_time() - c.start
            if not (0.1 < cut < c.duration - 0.1):
                self.lbl_status.setText("Put the playhead inside the layer clip to split it.")
                return
            d = OverlayClip.from_dict(c.to_dict())
            d.id, d.start, d.src_in = new_id(), c.start + cut, c.src_at(cut)
            c.src_out = d.src_in
            c.fade_out, d.fade_in = 0.0, 0.0
            self.project.overlays.append(d)
        elif action.startswith("ov_place:"):
            c.x, c.y, c.scale = PLACEMENTS[action.split(":", 1)[1]]
        elif action.startswith("ov_rotate:"):
            v = float(action.split(":", 1)[1])
            c.rotation = 0.0 if v == 0 else wrap_angle(c.rotation + v)
        elif action.startswith("ov_fade_in:"):
            c.fade_in = max(0.0, float(action.split(":", 1)[1]))
        elif action.startswith("ov_fade_out:"):
            c.fade_out = max(0.0, float(action.split(":", 1)[1]))
        elif action == "ov_reset_color":
            c.grade = Grade()
        elif action == "ov_copy_grade":
            for x in self._graded_clips():
                if x is not c:
                    x.grade = Grade.from_dict(c.grade.to_dict())
            self.inspector.set_clip(self._inspected())
            self.viewer.refresh()
        elif action.startswith("ov_speed:"):
            c.speed = clamp_speed(float(action.split(":", 1)[1]))
        elif action.startswith("ov_ramp:"):
            c.ramp = action.split(":", 1)[1]
        elif action.startswith("ov_audio:"):
            c.audio_mode = sound_mode(action.split(":", 1)[1])
        elif action == "ov_speed_custom":
            v = self._ask_speed(c.speed)
            if v is None:
                return
            c.speed = v
        elif action.startswith("ov_layer:"):
            c.track = max(0, min(MAX_LAYERS - 1, c.track + (1 if action.endswith("up") else -1)))
            self.project.settings.overlay_tracks = max(self.project.settings.overlay_tracks, c.track + 1)
        self._overlays_changed()

    def _fit_lane_heights(self):
        self.tl.fit_heights(self.timeline.area.viewport().height())

    # ================================================================== titles & text
    def add_text(self, preset: str = "title", t: Optional[float] = None):
        """Add a title from a template at time t (default: the playhead)."""
        if not self.project.clips:
            QMessageBox.information(self, APP_NAME, "Put some clips on the timeline first (2. ✨ Auto Edit).")
            return
        t = self.viewer.timeline_time() if t is None else t
        tc = make_text(preset, t, self.project.duration)
        self.project_checkpoint()
        self.project.texts.append(tc)
        self._texts_changed()
        self.tl.select_text(tc.id)
        self.show_refine(self.text_panel)
        self._timeline_seek(min(self.project.duration - 1e-3, tc.start + anim_lengths(tc)[0] + 0.05))
        self.lbl_status.setText(f"Added \"{TEXT_PRESETS[preset]['label']}\" — type your text in the Text tab, "
                                "drag it in the viewer to place it")
        QTimer.singleShot(0, self.text_panel.focus_text)

    def _text(self, tid: str) -> Optional[TextClip]:
        return next((x for x in self.project.texts if x.id == tid), None)

    def _text_action(self, action: str, tid: str):
        tc = self._text(tid)
        if tc is None:
            return
        if action == "text_edit":
            self.tl.select_text(tid)
            self.show_refine(self.text_panel)
            if not (tc.start <= self.viewer.timeline_time() < tc.end):
                self._timeline_seek(min(tc.end - 1e-3, tc.start + anim_lengths(tc)[0] + 0.05))
            QTimer.singleShot(0, self.text_panel.focus_text)
            return
        self.project_checkpoint()
        if action == "text_delete":
            self.project.texts.remove(tc)
            if self.tl.selected_text == tid:
                self.tl.select_text("")
        elif action == "text_to_playhead":
            tc.start = round(self.viewer.timeline_time(), 3)
        elif action == "text_duplicate":
            d = TextClip.from_dict(tc.to_dict())
            d.id, d.start = new_id(), tc.end
            self.project.texts.append(d)
            self.tl.select_text(d.id)
        elif action.startswith("text_anim_in:"):
            tc.anim_in = action.split(":", 1)[1]
        elif action.startswith("text_anim_out:"):
            tc.anim_out = action.split(":", 1)[1]
        elif action.startswith("text_pos:"):
            tc.x, tc.y, tc.align = {"top": (0.5, 0.12, "center"), "middle": (0.5, 0.5, "center"),
                                    "bottom": (0.5, 0.82, "center"),
                                    "lower_left": (0.06, 0.8, "left")}[action.split(":", 1)[1]]
        self._texts_changed()

    def _texts_changed(self, panel: bool = True, panel_fields: bool = False):
        """After titles changed: redraw the timeline / viewer and (unless the panel made the change) the panel."""
        self.project.dirty = True
        if panel:
            self.text_panel.refresh()
        elif panel_fields:
            self.text_panel.refresh()
        self.tl.update()
        self.tl.view_changed.emit()
        self.viewer.canvas.texts = self.project.texts
        self.viewer.canvas.update()
        self._update_title()

    def _on_text_selected(self, tid: str):
        self.viewer.canvas.selected_text = tid
        self.viewer.canvas.update()
        if (self.text_panel.text.id if self.text_panel.text else "") != tid:
            self.text_panel.refresh(select=tid)
        if tid:
            self.show_refine(self.text_panel)

    def _text_dragged(self, tid: str, x: float, y: float):
        tc = self._text(tid)
        if tc is None:
            return
        self.project_checkpoint(f"text:{tid}:drag")
        tc.x, tc.y = x, y
        self.viewer.canvas.update()

    # ================================================================== right-click menus
    def _timeline_action(self, key: str, payload: str):
        """Context-menu commands from the timeline (clips, titles, captions, empty space)."""
        if key == "show_media":                      # payload: a clip id (main track) or a footage id (layers)
            c = next((x for x in self.project.clips if x.id == payload), None)
            self.show_in_media(c.footage_id if c is not None else payload)
        elif key.startswith("text_add:"):
            self.add_text(key.split(":", 1)[1], float(payload))
        elif key.startswith("text_"):
            self._text_action(key, payload)
        elif key == "open_sounds":
            self.show_refine(self.sfx_panel)
        elif key == "seek":
            self._timeline_seek(float(payload))
        elif key == "split_playhead":
            self.split_at_playhead()
        elif key == "fit":
            self.tl.fit()
        elif key == "captions_generate":
            self.generate_captions()
        elif key == "layer_add":
            if self.tl.layers < MAX_LAYERS:
                self.project_checkpoint()
                self.project.settings.overlay_tracks = self.tl.layers + 1
                self.tl.refresh()
                self.lbl_status.setText(f"Added video layer V{self.tl.layers + 1}: drag footage or images onto it")
        elif key == "layer_remove":
            self.project_checkpoint()
            self.project.settings.overlay_tracks = max(1, self.tl.layers - 1)
            self.tl.refresh()
        elif key == "ov_add_media":
            t, track = payload.split(":")
            ids = self.bin.selected_ids()
            if not ids:
                self.lbl_status.setText("Select footage or images in the Media panel first.")
            else:
                self.add_overlays(ids, float(t), int(track))
        elif key == "link_toggle":
            self.toggle_link(payload)
        elif key.startswith("multicam_layout:"):
            self.multicam_layout(key.split(":", 1)[1])
        elif key.startswith("ov_"):
            self._overlay_action(key, payload)
        elif key == "caption_style":
            self.caption_style_dialog()
        elif key == "caption_edit":
            i = int(payload)
            self.show_refine(self.caps_panel)
            self.caps_panel.select(i)
            item = self.caps_panel.table.item(i, 2)
            if item is not None:
                self.caps_panel.table.editItem(item)
        elif key == "caption_delete":
            self._delete_captions([int(payload)])
        elif key == "caption_delete_all":
            self._delete_captions(list(range(len(self.captions))))
        else:
            self.clip_action(key, payload)

    def _viewer_menu(self, gpos, tid: str):
        m = QMenu(self)
        tc = self._text(tid) if tid else None
        if tc is not None:
            for key, label in (("text_edit", "Edit text…"), ("text_duplicate", "Duplicate text")):
                m.addAction(label, lambda k=key: self._text_action(k, tid))
            for title, table, prefix in (("Animation in", ANIM_IN, "text_anim_in"),
                                         ("Animation out", ANIM_OUT, "text_anim_out")):
                sub = m.addMenu(title)
                for k, label in table.items():
                    a = sub.addAction(label, lambda k=k, pre=prefix: self._text_action(f"{pre}:{k}", tid))
                    a.setCheckable(True)
                    a.setChecked(getattr(tc, prefix[5:]) == k)
            m.addAction("Delete text", lambda: self._text_action("text_delete", tid))
            m.addSeparator()
        if self.project.clips:
            sub = m.addMenu("Add title here")
            for key, cfg in TEXT_PRESETS.items():
                sub.addAction(cfg["label"], lambda k=key: self.add_text(k))
            m.addAction("Pause" if self.viewer.playing else "Play", self.viewer.toggle_play)
            m.addSeparator()
        sub = m.addMenu("Color in viewer")
        for a in self.a_adjust.values():
            sub.addAction(a)
        m.addAction(self.a_guides)
        sub = m.addMenu("Frame shape")
        for k in ASPECT_RATIOS:
            a = sub.addAction(k, lambda k=k: self.cb_aspect.setCurrentIndex(self.cb_aspect.findData(k)))
            a.setCheckable(True)
            a.setChecked(self.project.settings.aspect == k)
        if self.viewer.canvas.image is not None:
            m.addSeparator()
            m.addAction("Save this frame as an image…", self._save_frame)
        m.exec(gpos)

    def _save_frame(self):
        cv = self.viewer.canvas
        rect = cv._frame_rect.toAlignedRect()
        if rect.isEmpty():
            return
        start = os.path.splitext(self.project.path)[0] + f"_{self.viewer.timeline_time():.1f}s.png" if self.project.path else "frame.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save frame", start, "PNG image (*.png);;JPEG image (*.jpg)")
        if path:
            cv.grab(rect).save(path)
            self.lbl_status.setText(f"Saved {path}")

    # ================================================================== guided tour
    def _first_run_tour(self):
        if not self.qs.value("tour_done", False, type=bool):
            self.show_tour()

    def show_tour(self, start: int = 0):
        if self._tour is not None:
            return
        tl = self.tl
        steps = [
            TourStep("Welcome to KangEdit 👋",
                     "Turn your footage into a finished edit in four steps:<br>"
                     "<b>1. Import › 2. Auto Edit › 3. Refine › 4. Export</b><br><br>"
                     "This tour takes a minute. Open it again any time with the <b>?</b> button or "
                     "<b>Help → Guided tour</b>."),
            TourStep("1. Import your footage",
                     "Drag video files onto the <b>Media</b> panel, or click <b>1. Import</b>.<br>"
                     "<b>Right-click</b> footage for more: add it to the timeline, camera log profile, proxies…",
                     lambda: self.d_bin),
            TourStep("2. Let Auto Edit build the edit",
                     "<b>✨ Auto Edit</b> picks the best moments and puts them on the timeline.<br>"
                     "<b>− / +</b> rebuild it shorter or longer. Running it again on the same footage picks "
                     "<b>new moments</b>, and moments with dead silence are left out.",
                     lambda: self.grp_auto),
            TourStep("Swap moments",
                     "<b>🎣 Hook</b> chooses the opening hook from the strongest moments of the whole video, "
                     "as thumbnails (1 quick clip, or a teaser of up to 5): point at one to play it, tick them "
                     "in the order the hook should play them.<br>"
                     "<b>🔀 Other moment</b> (R) swaps the selected clip for the best moment close by in its "
                     "footage; the clip takes that moment's length. Press again for the next one; "
                     "<b>↶ Undo</b> brings the previous back.",
                     lambda: self._moment_buttons()),
            TourStep("Preview",
                     "<b>Space</b> plays and pauses. Titles can be <b>dragged here</b> to place them.<br>"
                     "<b>Double-click</b> the picture (or press <b>F11</b>) for full screen, <b>Esc</b> to leave.<br>"
                     "<b>View → App overlay</b> shows where Reels / TikTok buttons cover a vertical video.<br>"
                     "<b>Right-click</b> the picture to add a title, change the colour view or save a frame.<br>"
                     "<b>Double-click footage in Media</b> to play it here on its own (<b>Source</b>): mark the part "
                     "to use (<b>I</b> / <b>O</b>, or select its words under the picture) and <b>Insert</b> it; "
                     "<b>Timeline</b> shows the edit again.<br>"
                     "<b>♪ updating</b> next to the time: the preview sound is being mixed again after a "
                     "change (music just added is heard once it is ready).",
                     lambda: self.viewer.canvas),
            TourStep("The timeline",
                     "From the top: <b>titles</b>, <b>video layers</b> (V2, V3… for picture in picture and "
                     "images on top), the <b>main video track</b>, <b>sound effects</b> and <b>captions</b>.<br>"
                     "Drag footage or images from Media onto them. Drag a lane's bottom edge to make it taller.<br>"
                     "Drag clips to reorder, drag their edges to trim, <b>Alt-drag</b> to slide which moment "
                     "a clip shows: in <b>⇋ Slip</b> mode the clip's transcript shows, click a sentence to slip "
                     "to it.<br>Icons on a clip say what it is and what is done to it: 🎣 hook, 🏞 scene opener, "
                     "👋 opening / closing, 🔍 dynamic zoom, 〰 stabilized, 🎙 voice cleaned up.",
                     lambda: self.timeline),
            TourStep("Video layers",
                     "The lanes <b>V2, V3…</b> above the main track lay pictures <b>on top</b> of the edit: picture "
                     "in picture, a logo, B-roll, a second camera.<br>Drag footage or an image from <b>Media</b> "
                     "onto a layer, then drag it in the <b>viewer</b> to place it (mouse wheel resizes). The "
                     "<b>🧩 Layer</b> tab sets size, crop, rotation, opacity and fades.<br>Filmed with two "
                     "cameras? <b>✨ Auto Edit → Multicam</b> syncs them by their sound and lays them out "
                     "(picture in picture, talking cam, stacked, side by side…).",
                     lambda: (tl, QRect(0, OV_Y, tl.width(), tl.ov_h))),
            TourStep("Right-click anything",
                     "Every part of the timeline has its own menu: <b>clips</b> (transitions, J/L cuts, another "
                     "moment, rotate), <b>layer clips</b> (position, fades, layer, multicam layout), <b>empty "
                     "space</b> (add a title, a sound or a video layer), <b>captions</b> (edit, delete) and "
                     "<b>titles</b> (animations, position).",
                     lambda: (tl, tl.rect().adjusted(0, 24, 0, 0))),
            TourStep("3. Refine",
                     "Fine-tune here: <b>🎨 Clip</b> colour & reframe, <b>🧩 Layer</b> for clips on the video "
                     "layers, <b>💬 Captions</b>, <b>🔤 Text</b> for animated titles, and <b>🔊 Sounds</b>. The <b>3. Refine</b> buttons in the toolbar jump "
                     "to each tab.<br>Select <b>several clips</b> (Ctrl / Shift+click) and every change here "
                     "applies to all of them. <b>🙂 Zoom to face</b> under Reframe punches in on a reaction; "
                     "<b>Sharpness</b> and <b>Halation</b> sit below Vibrance.<br>"
                     "In <b>🎨 Clip</b> (also for layer clips): <b>Dynamic zoom</b> (a slow zoom or pan, or in on "
                     "the face), <b>Stabilize</b> for shaky handheld shots and, under Audio, <b>Clean up the "
                     "voice</b> (noise, wind). <b>🗒 Transcript</b> shows what is said in the edit: click to go "
                     "there, drag to reorder the clips.",
                     lambda: self.d_right),
            TourStep("Frame",
                     "The shape of the finished video (9:16 for Reels / TikTok, 16:9 for YouTube…), its "
                     "resolution and frame rate.",
                     lambda: self.grp_frame),
            TourStep("For YouTube and socials",
                     "<b>Tools → Make a thumbnail…</b>: the edit's best stills, framed on the face, with big text, "
                     "saved as a JPG.<br><b>Tools → YouTube chapters…</b>: chapter timestamps for the description "
                     "from the edit's scenes, and title cards for them.<br><b>Tools → Make a highlight reel</b>: "
                     "a vertical 30-60 s reel of the best moments.",
                     lambda: self.menuBar()),
            TourStep("4. Export",
                     "Render the finished video. Captions can be burned in or saved as an .srt file.<br>"
                     "First a <b>check</b> lists what could spoil it (a missing file, missing music, captions "
                     "without transcripts, a draft from proxies…), each with a fix.",
                     lambda: self.btn_export),
            TourStep("That's it!",
                     "Start with <b>1. Import</b>. If you get stuck, the hint at the bottom right says what to do "
                     "next, and the <b>?</b> button brings this tour back.<br><br>Your project is "
                     "<b>auto-saved</b> every minute with rolling backups: <b>File → Restore a backup…</b> "
                     "brings back an earlier version.",
                     lambda: self.btn_help),
        ]
        self._tour = TourOverlay(self, steps)

        def done(_completed: bool):
            self._tour = None
            self.qs.setValue("tour_done", True)

        self._tour.finished.connect(done)
        self._tour.start(start)

    def _moment_buttons(self):
        """The tour's target: the Hook / Other moment buttons above the timeline."""
        tp = self.timeline
        r = tp.btn_hook.geometry().united(tp.btn_alt.geometry())
        return tp, r.adjusted(-4, -2, 4, 2)

    def _bins_edited(self, message: str):
        self.project.dirty = True
        self._update_title()
        if message:
            self.lbl_status.setText(message)

    def _select_all(self):
        """Ctrl+A where the focus is: every item in the Media panel, the text of a text field, or else every clip
        on the timeline (main track and video layers)."""
        w = QApplication.focusWidget()
        if w is not None and self.bin.isAncestorOf(w):
            self.bin.select_all()
        elif isinstance(w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            w.selectAll()
        else:
            self.tl.select_all()

    def _delete_selected(self):
        layer = self.tl.selected_overlay_ids()
        if len(layer) > 1 and not self.tl.selected:     # several layer clips (Ctrl+click)
            self.project_checkpoint()
            self.project.overlays = [o for o in self.project.overlays if o.id not in set(layer)]
            self.tl.select_overlay("")
            self._overlays_changed()
        elif self.tl.selected and self.tl.selected_ids():
            self.clip_action("delete", self.tl.selected)
        elif self.tl.selected_overlay:
            self._overlay_action("ov_delete", self.tl.selected_overlay)
        elif self.tl.selected_text:
            self._text_action("text_delete", self.tl.selected_text)
        elif self.tl.selected_sound:
            self._sound_action("sound_delete", self.tl.selected_sound)
        else:
            self.clip_action("delete", self.tl.selected)

    def undo(self):
        if self.project.undo():
            self.refresh_all()
            self._schedule_audio()

    def redo(self):
        if self.project.redo():
            self.refresh_all()
            self._schedule_audio()

    # ================================================================== inspector
    def _inspector_before(self, kind: str):
        c = self._inspected()
        self.project_checkpoint(f"{kind}:{c.id if c else ''}")

    def _inspector_changed(self, kind: str):
        self.project.dirty = True
        if self.inspector.layer_others:       # the change reached selected clips on the layers too
            self.tl.refresh()
            self.viewer.refresh()
            self._schedule_audio()
        if kind == "motion":
            self._find_motion_faces(self.inspector.graded)
        if kind == "audio":
            self._measure_noise(self.inspector.graded)
        if self.selected_overlay() is not None and self.selected_clip() is None:
            # A clip on a video layer: its grade, trim, speed, sound or movement.
            if kind == "footage_color":
                self.bin.refresh()
            if kind in ("stabilize", "trim"):
                self.measure_shake()
            self._overlays_changed()
            return
        multicam.follow(self.project)      # a main-angle box on a layer shows the clip's grade and trim
        if kind in ("trim",):
            self._on_timeline_edited()
            self.tl.refresh()
            self._schedule_audio()
            self.measure_shake()
        elif kind == "footage_color":
            self.bin.refresh()
            self.viewer.refresh()
        elif kind == "audio":
            self._measure_noise(self.inspector.graded)
            self.tl.update()
            self._schedule_audio()
        elif kind == "stabilize":
            self.measure_shake()
            self.viewer.refresh()
        else:
            self.viewer.refresh()
        self._update_title()

    def measure_shake(self):
        """Measure, in the background, the camera's movement for the stabilized clips that need it (once per
        range of a file; trimming or slipping past it measures the new part)."""
        todo = [x for x in stabilize.missing(self.project) if (x[0].id, x[1], x[2]) not in self._measuring]
        if not todo:
            return
        use_proxies = self.project.settings.use_proxies
        for f, t0, t1 in todo:
            key = (f.id, t0, t1)
            self._measuring.add(key)

            def job(progress, cancel, f=f, t0=t0, t1=t1):
                stabilize.analyze(f, t0, t1, use_proxies, lambda p: progress(p, f"Measuring the shake: {f.name}"),
                                  cancel)

            def done(_res, key=key):
                self._measuring.discard(key)
                self.viewer.refresh()
                self.inspector.show_stab_state()

            def failed(*a, key=key, f=f):
                self._measuring.discard(key)
                self._task_error(f"Stabilizing {f.name}")(*a)

            self.tasks.submit(f"Stabilize {f.name}", job, done, failed)

    def _graded_clips(self) -> list:
        """Every clip whose picture can be graded: the main track and the video on the layers (pictures on the
        layers, such as logos and graphics, are left as they are)."""
        return list(self.project.clips) + [o for o in self.project.overlays if o.footage_id in self.project.footage
                                           and not self.project.footage[o.footage_id].is_image]

    def inspector_request(self, req: str):
        clip = self._inspected()
        if req in ("import_lut_log", "import_lut_look"):
            self.import_lut(req.rsplit("_", 1)[1])
            return
        if req == "detect_log":
            if clip is not None:
                self.detect_log(list(dict.fromkeys(c.footage_id for c in self.inspector.targets or [clip])))
            return
        if req == "voice_auto":
            self.clean_voices_with_wind()
            return
        if req == "layer_tab":
            self.show_refine(self.layer_panel)
            return
        if req == "detect_log_all":
            self.detect_log(list(self.project.footage))
            return
        if req == "auto_color_all":
            self._auto_color(self._graded_clips())
            return
        elif clip is None:
            return
        elif req == "auto_color":
            self._auto_color(self.inspector.graded or [clip])
            return
        elif req.startswith("face_zoom:"):
            self._face_zoom(self.inspector.targets or [clip], req.split(":", 1)[1])
            return
        elif req == "reset_framing":
            self.project_checkpoint()
            for c in self.inspector.targets or [clip]:
                c.zoom, c.offset_x, c.offset_y = 1.0, 0.0, 0.0
        elif req == "reset_color":
            self.project_checkpoint()
            for c in self.inspector.graded or [clip]:
                c.grade = Grade()
        elif req == "copy_grade_all":
            self.project_checkpoint()
            for c in self._graded_clips():
                if c is not clip:
                    c.grade = Grade.from_dict(clip.grade.to_dict())
        self.inspector.set_clip(self._inspected())
        self.layer_panel.set_clip(self.selected_overlay())
        self.viewer.refresh()
        self._update_title()

    def _measure_noise(self, clips: list):
        """The background noise level of the clips whose voice is cleaned up, from their analysis (the noise
        reduction works on noise near it; see voice.py)."""
        for c in clips:
            if c.voice_clean > 0 and not c.voice_floor:
                c.voice_floor = voice.noise_floor(self.analyses.get(c.footage_id), c.src_in, c.src_out)

    def clean_voices_with_wind(self):
        """Clean up the voice in every clip (main track and layers) where the smarter analysis heard wind."""
        todo = []
        for c in list(self.project.clips) + list(self.project.overlays):
            f = self.project.footage.get(c.footage_id)
            if f is None or not f.has_audio or f.is_image or c.voice_clean > 0:
                continue
            k = voice.suggest(c, self.analyses.get(c.footage_id))
            if k is not None:
                todo.append((c, k))
        if not todo:
            heard = any(getattr(a, "sound", None) is not None and len(a.sound) for a in self.analyses.values())
            self.lbl_status.setText("No wind heard in the clips" if heard else
                                    "No wind found: it is heard by the smarter analysis (Tools → Smarter analysis)")
            return
        self.project_checkpoint()
        for c, k in todo:
            c.voice_clean = k
        self._measure_noise([c for c, _ in todo])
        self.project.dirty = True
        self.inspector.set_clip(self._inspected())
        self.tl.refresh()
        self._schedule_audio()
        self.lbl_status.setText(f"Voice clean-up on in {len(todo)} clip(s) with wind")

    def _find_motion_faces(self, clips: list):
        """Find (in the background) where the face is in clips zooming on it ("Zoom in on the face"): its centre
        in the clip's picture, from frames of the clip's own moment. None found: the zoom goes to the middle."""
        todo = [c for c in clips if c.motion in motion.FACE_KINDS and c.motion_face is None
                and c.footage_id in self.project.footage and not self.project.footage[c.footage_id].is_image]
        for c in clips:                      # a photo: looked for at once (no decoding)
            f = self.project.footage.get(c.footage_id)
            if c.motion in motion.FACE_KINDS and c.motion_face is None and f is not None and f.is_image:
                todo.append(c)
        if not todo:
            return
        jobs = [(c.id, c.footage_id, c.src_in, c.src_dur, 0 if hasattr(c, "track") else
                 reframe.split_rotation(c.rotation)[0]) for c in todo]
        footage = {fid: self.project.footage[fid] for _, fid, *_ in jobs}
        use_proxies = self.project.settings.use_proxies

        def job(progress, cancel):
            pool = DecoderPool(limit=4)
            try:
                out = {}
                for k, (cid, fid, a, dur, quarters) in enumerate(jobs):
                    if cancel.is_set():
                        raise Cancelled()
                    progress(k / len(jobs), f"Finding the face: {k + 1}/{len(jobs)}")
                    dec = pool.get(footage[fid], use_proxies, 720)
                    frames = [reframe.rotate_quarters(img.copy(), quarters) for i in range(FACE_SAMPLES)
                              if (img := dec.frame_at(a + dur * (i + 0.5) / FACE_SAMPLES)) is not None]
                    face = reframe.find_face(frames) if frames else None
                    out[cid] = [float(face[0]), float(face[1])] if face is not None else []
                return out
            finally:
                pool.close()

        def done(found: dict):
            for c in list(self.project.clips) + list(self.project.overlays):
                if c.id in found and c.motion_face is None:
                    c.motion_face = found[c.id]
            missed = sum(1 for v in found.values() if not v)
            self.project.dirty = True
            self.viewer.refresh()
            self.lbl_status.setText("Zooming on the face" + (f" · no face found in {missed} clip(s): they zoom to "
                                                              "the middle" if missed else ""))

        self.tasks.submit("Find the face", job, done, self._task_error("Find the face"))

    def _face_zoom(self, clips: list[TimelineClip], size: str = "close"):
        """Zoom each clip in on its main face (in the background); applied in one undo step."""
        st = self.project.settings
        tw, th = frame_size(st.aspect, st.resolution)
        jobs = [(TimelineClip(c.footage_id, c.src_in, c.src_out, rotation=c.rotation, id=c.id),
                 self.project.footage[c.footage_id]) for c in clips if c.footage_id in self.project.footage]
        if not jobs:
            return
        use_proxies, mode = st.use_proxies, st.fit_mode
        frac = reframe.FACE_SIZES.get(size, reframe.FACE_SIZES["close"])

        def job(progress, cancel):
            pool = DecoderPool(limit=4)
            try:
                out = {}
                for k, (c, f) in enumerate(jobs):
                    if cancel.is_set():
                        raise Cancelled()
                    progress(k / len(jobs), f"Finding faces: {k + 1}/{len(jobs)}")
                    # 720p keeps small faces in wide shots findable; the clip's quarter turns are applied
                    # first so the face is found (and framed) the way the clip shows it.
                    dec = pool.get(f, use_proxies, 720)
                    quarters = reframe.split_rotation(c.rotation)[0]
                    frames = [reframe.rotate_quarters(img.copy(), quarters) for i in range(FACE_SAMPLES)
                              if (img := dec.frame_at(c.src_in + c.src_dur * (i + 0.5) / FACE_SAMPLES))
                              is not None]
                    face = reframe.find_face(frames)
                    if face is not None:
                        h, w = frames[0].shape[:2]
                        out[c.id] = reframe.face_framing(face, w, h, tw, th, mode, frac)
                return out
            finally:
                pool.close()

        def done(framing: dict):
            targets = [c for c in self.project.clips if c.id in framing]
            missed = len(jobs) - len(targets)
            if targets:
                self.project_checkpoint()
                for c in targets:
                    c.zoom, c.offset_x, c.offset_y = framing[c.id]
                self.project.dirty = True
                self.inspector.set_clip(self._inspected())
                self.viewer.refresh()
                self._update_title()
            self.lbl_status.setText(f"Zoomed in on the face in {len(targets)} clip{'s' * (len(targets) != 1)}"
                                    + (f" · no face found in {missed}" if missed else "") if targets
                                    else "No face found in the selected clip" + ("s" if len(jobs) > 1 else ""))

        self.tasks.submit("Zoom to face", job, done, self._task_error("Zoom to face"))

    def _auto_color(self, clips: list[TimelineClip]):
        """Auto-grade clips in the background (progress + Cancel in the status bar); applied in one undo step."""
        jobs = []
        for c in clips:
            f = self.project.footage.get(c.footage_id)
            if f is not None:
                probe_clip = TimelineClip(c.footage_id, c.src_in, c.src_out, id=c.id)
                jobs.append((probe_clip, f.log_profile, f.tonemap, Grade.from_dict(c.grade.to_dict())))
        if not jobs:
            return
        use_proxies = self.project.settings.use_proxies
        footage = dict(self.project.footage)
        name = "Auto color" if len(jobs) == 1 else f"Auto color {len(jobs)} clips"

        def job(progress, cancel):
            pool = DecoderPool(limit=4)   # own decoders: don't disturb the viewer's
            try:
                grades = {}
                for k, (c, lp, tm, base) in enumerate(jobs):
                    if cancel.is_set():
                        raise Cancelled()
                    progress(k / len(jobs), f"{name}: {k + 1}/{len(jobs)}")
                    dec = pool.get(footage[c.footage_id], use_proxies, 360)
                    frames = []
                    for i in range(5):
                        img = dec.frame_at(c.src_in + c.src_dur * (i + 0.5) / 5)
                        if img is not None:
                            frames.append(img.copy())
                    grades[c.id] = auto_correct(frames, lp, tm, base=base,
                                               full_range=footage[c.footage_id].color_range == "pc")
                progress(1.0, name)
                return grades
            finally:
                pool.close()

        def done(grades: dict):
            targets = [(c, grades[c.id]) for c in [*self.project.clips, *self.project.overlays] if c.id in grades]
            if not targets:
                return
            self.project_checkpoint()
            for c, g in targets:
                c.grade = g
            self.project.dirty = True
            self.inspector.set_clip(self._inspected())
            self.viewer.refresh()
            self._update_title()

        self.tasks.submit(name, job, done, self._task_error(name))

    # ================================================================== captions
    def update_captions(self):
        self.captions = build_captions(self.project)
        self.tl.captions = self.captions
        self.tl.update()
        self.viewer.set_captions(self.captions)
        self.caps_panel.set_captions(self.captions)

    def queue_transcription(self, ids: list[str], force: bool = False, use_cache: Optional[bool] = None):
        """Transcribe footage as one background task: the speech model is loaded once (downloaded on first use),
        then the files are transcribed one after the other. A file that fails doesn't stop the others."""
        s = self.project.settings
        src = self.project.transcript_source        # footage, or a voice-over sound file (Project.voices)
        todo = [src(fid) for fid in dict.fromkeys(ids) if src(fid) is not None and not src(fid).is_image
                and (force or src(fid).transcript is None)]
        if not todo:
            if ids:
                self.lbl_status.setText("Already transcribed. To transcribe again (with another speech model): "
                                        "right-click it in Media → Transcribe (again) for captions")
            return
        model, lang, cleanup = s.whisper_model, s.whisper_language, s.speech_cleanup

        def job(progress, cancel):
            # The speech model is loaded once for the batch, and only when a file isn't transcribed already.
            if any(f.has_audio and (force or cached_transcript(f, model, lang) is None) for f in todo):
                load_model(model, progress, cancel)
            done, errors = [], []
            for k, f in enumerate(todo):
                if cancel.is_set():
                    return done, errors, True
                tag = f" ({k + 1}/{len(todo)})" if len(todo) > 1 else ""

                def step(p, msg="", k=k, tag=tag):
                    progress((k + max(0.0, min(1.0, p))) / len(todo), (msg or "Transcribing") + tag)
                try:
                    words = transcribe(f, model, lang, step, cancel,
                                       use_cache=(not force) if use_cache is None else use_cache, cleanup=cleanup)
                except Cancelled:          # the files finished before the cancel are kept
                    return done, errors, True
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{f.name}: {e}")
                    continue
                done.append((f.id, words))
            return done, errors, False

        def finished(res):
            done, errors, cancelled = res
            if cancelled:
                self.lbl_status.setText(f"Transcription cancelled: {len(done)} of {len(todo)} file(s) done")
            for fid, words in done:
                ff = self.project.transcript_source(fid)
                if ff is not None:
                    ff.transcript = words
                    self.viewer.update_source(ff)
            if done:
                self.project.dirty = True
                self.update_captions()
                self.bin.refresh()
            if errors:
                QMessageBox.warning(self, APP_NAME, "Some files could not be transcribed:\n\n" + "\n".join(errors))

        name = (f"Transcribing {todo[0].name}" if len(todo) == 1 else f"Transcribing {len(todo)} files") \
            + model_note(model)
        waiting = self.tasks.current if self.tasks.busy else None
        self.lbl_status.setText(f"{name}: starts after “{waiting}”" if waiting else f"{name}…")
        self.tasks.submit(name, job, finished, self._task_error("Transcription"))

    def _fill_transcripts(self, ids) -> int:
        """Transcripts of these files made before (in any project, see captions.cached_transcript), without
        transcribing anything. Returns how many were filled in."""
        st = self.project.settings
        n = 0
        for fid in ids:
            f = self.project.footage.get(fid)
            if f is None or f.is_image or not f.has_audio or f.transcript is not None:
                continue
            hit = cached_transcript(f, st.whisper_model, st.whisper_language, any_model=True)
            if hit is not None:
                f.transcript, f.transcript_lang = hit
                n += 1
        return n

    def _ask_transcribe(self, n: int) -> tuple[bool, bool]:
        """Ask whether to transcribe n files now: (yes, don't ask again)."""
        box = QMessageBox(QMessageBox.Question, APP_NAME,
                          f"{n} file(s) in the project have sound but no transcript yet.\n"
                          "Transcribe them now (in the background)? Captions, the full transcript, edits by "
                          "script and cuts on whole sentences use the transcripts.",
                          QMessageBox.Yes | QMessageBox.No, self)
        box.setDefaultButton(QMessageBox.Yes)
        chk = QCheckBox("Don't ask again (Tools → Offer to transcribe after importing)")
        box.setCheckBox(chk)
        return box.exec() == QMessageBox.Yes, chk.isChecked()

    def offer_transcription(self):
        """After an import: offer to transcribe every file of the project that has sound and no transcript."""
        if not self.a_ask_transcribe.isChecked():
            return
        todo = [fid for fid, f in self.project.footage.items()
                if not f.is_image and f.has_audio and f.transcript is None]
        if not todo:
            return
        yes, stop = self._ask_transcribe(len(todo))
        if stop:
            self.a_ask_transcribe.setChecked(False)
        if yes:
            self.queue_transcription(todo)

    def transcript_dialog(self, ids: Optional[list[str]] = None):
        """The transcripts merged in recording order (all footage, or the given files)."""
        dlg = TranscriptDialog(self.project, ids, self, transcribe=self.queue_transcription)
        dlg.exec()

    def show_in_media(self, fid: str):
        """Timeline → Show in Media panel: select the clip's footage there (clearing filters that hide it)."""
        if self.bin.reveal(fid):
            dock = self.bin.parentWidget()
            if dock is not None:
                dock.show()
                dock.raise_()
            self.lbl_status.setText(f"{self.project.footage[fid].name} is selected in the Media panel")

    def transcribe_media(self, ids: list[str]):
        """Media panel → Transcribe (again) for captions: the selected files not yet transcribed with the speech
        model and language set now (Captions panel, Captions → Style…), with them (their transcript replaced);
        the ones already transcribed with them are kept (their manual caption edits too)."""
        s = self.project.settings
        media = [fid for fid in ids if fid in self.project.footage and not self.project.footage[fid].is_image]
        todo = [fid for fid in media
                if not transcribed_with(self.project.footage[fid], s.whisper_model, s.whisper_language)]
        kept = len(media) - len(todo)
        name = resolve_model(s.whisper_model)
        if not todo:
            if media:
                self.lbl_status.setText(("The selected file is" if len(media) == 1 else
                                         f"All {len(media)} selected files are") +
                                        f" already transcribed with {name}: kept. To transcribe again, choose "
                                        "another speech model or language (Captions panel)")
            return
        if kept:
            self.lbl_status.setText(f"Transcribing {len(todo)} file(s) with {name} "
                                    f"({kept} already transcribed with it, kept)")
        # (from the transcript cache when the file was transcribed with this model before: no waiting)
        self.queue_transcription(todo, force=True, use_cache=True)

    def transcribe_again(self, ids: list[str]):
        """Right-click → Transcribe again from scratch: the files transcribed anew with the speech model and
        language set now, the saved transcripts (this project's and the cache's) not used. Undo brings the
        old transcript back."""
        media = [fid for fid in dict.fromkeys(ids) if fid in self.project.footage
                 and not self.project.footage[fid].is_image and self.project.footage[fid].has_audio]
        if not media:
            self.lbl_status.setText("Nothing to transcribe: no sound in the selected footage")
            return
        self.project_checkpoint()
        self.queue_transcription(media, force=True, use_cache=False)

    def _captions_shifted(self, index: int, dt: float):
        """A caption dragged (or nudged) left / right on the timeline: it and its clip's later captions move."""
        if not 0 <= index < len(self.captions):
            return
        self.project_checkpoint()
        moved = shift_captions(self.project, self.captions, index, dt)
        if abs(moved) < 1e-3:
            self.lbl_status.setText("The captions can't move that way: the words before or after them are in the way")
            return
        self.project.dirty = True
        self.update_captions()
        n = len(shift_group(self.captions, index))
        self.lbl_status.setText(f"Moved {n} caption(s) {abs(moved):.2f} s {'later' if moved > 0 else 'earlier'}"
                                + (" (as far as the words around them allow)" if abs(moved - dt) > 0.005 else "")
                                + " · Undo puts them back")

    def generate_captions(self):
        p = self.project
        used = list(dict.fromkeys(c.footage_id for c in p.clips))
        # Voice-over clips on the sound layer (right-click → Caption its speech) are captioned too.
        voices = list(dict.fromkeys(p.voice_for(x.path).id for x in p.voice_sounds()))
        if not used:
            QMessageBox.information(self, APP_NAME, "Put some clips on the timeline first.")
            return
        if all(p.footage[fid].is_image for fid in used) and not voices:
            QMessageBox.information(self, APP_NAME, "The timeline only has images: there is no speech to caption. "
                                                    "A voice-over on the sound layer is captioned when you right-click "
                                                    "it → Caption its speech.")
            return
        used = [fid for fid in used if not p.footage[fid].is_image] + voices
        todo = [fid for fid in used if p.transcript_source(fid).transcript is None]
        force = False
        if not todo:
            r = QMessageBox.question(self, APP_NAME, "All footage on the timeline is already transcribed.\n"
                                                     "Transcribe again (manual caption edits will be lost)?")
            if r != QMessageBox.Yes:
                return
            todo, force = used, True
        if not any(p.transcript_source(fid).has_audio for fid in todo):
            QMessageBox.information(self, APP_NAME, "The footage on the timeline has no audio.")
            return
        p.settings.captions.enabled = True
        self._sync_project_controls()
        self.right_tabs.setCurrentWidget(self.caps_panel)
        self.queue_transcription(todo, force)

    def caption_style_dialog(self):
        dlg = CaptionStyleDialog(self.project, self)
        if dlg.exec() == CaptionStyleDialog.Accepted:
            dlg.apply()
            self.project.dirty = True
            self.caps_panel.set_model(self.project.settings.whisper_model, self.project.settings.speech_cleanup)
            self.update_captions()
            self._update_title()

    def _set_speech_cleanup(self, key: str):
        """Captions panel → Background noise: for the transcriptions from now on (Transcribe again for the files
        transcribed already)."""
        self.project.settings.speech_cleanup = key
        self.project.dirty = True
        self._update_title()
        self.lbl_status.setText("Background noise setting changed: files transcribed already keep their "
                                "transcript (right-click in Media → Transcribe (again) for captions to redo one)")

    def _set_speech_model(self, key: str):
        """The speech model chosen in the Captions panel: for captions and every transcript from now on."""
        from .. import asr_qwen
        self.project.settings.whisper_model = key
        self.project.dirty = True
        self._update_title()
        if key == asr_qwen.MODEL and not asr_qwen.available():
            self.lbl_status.setText("Qwen3-ASR needs sherpa-onnx, which isn't installed: run the installer again "
                                    "(or pip install sherpa-onnx)")
        elif key == asr_qwen.MODEL and not asr_qwen.ready():
            self.lbl_status.setText(f"Qwen3-ASR: the model (about {asr_qwen.DOWNLOAD_GB:g} GB) is downloaded on the "
                                    "first transcription")

    def _caption_edited(self, row: int, text: str):
        if 0 <= row < len(self.captions):
            self.project_checkpoint()
            edit_caption(self.project, self.captions[row], text)
            QTimer.singleShot(0, self.update_captions)

    def _delete_captions(self, rows: list[int]):
        caps = [self.captions[r] for r in rows if 0 <= r < len(self.captions)]
        if not caps:
            return
        if len(caps) == len(self.captions) and len(caps) > 1 and QMessageBox.question(
                self, APP_NAME, f"Delete all {len(caps)} captions?\n\nUndo brings them back; Generate captions "
                                "can transcribe the footage again.") != QMessageBox.Yes:
            return
        self.project_checkpoint()
        delete_captions(self.project, caps)
        self.project.dirty = True
        self.update_captions()
        self.viewer.refresh()
        self._update_title()
        self.lbl_status.setText(f"Deleted {len(caps)} caption(s) — Ctrl+Z to undo")

    def _toggle_captions(self, on: bool):
        self.project.settings.captions.enabled = on
        self.viewer.refresh()

    def save_srt(self):
        if not self.captions:
            QMessageBox.information(self, APP_NAME, "No captions yet — click Generate captions first.")
            return
        start = os.path.splitext(self.project.path)[0] + ".srt" if self.project.path else "captions.srt"
        path, _ = QFileDialog.getSaveFileName(self, "Save subtitles", start, "SubRip (*.srt)")
        if path:
            write_srt(path, self.captions, self.project.settings.captions, project_layout(self.project))
            self.lbl_status.setText(f"Saved {path}")

    # ================================================================== export
    def export_dialog(self):
        if not self.project.clips:
            QMessageBox.information(self, APP_NAME, "The timeline is empty.")
            return
        dlg = ExportDialog(self.project, bool(self.captions), self, self.qs.value("gpu_export", True, type=bool))
        if dlg.exec() != ExportDialog.Accepted:
            return
        es = dlg.settings()
        self.qs.setValue("gpu_export", es.use_gpu)
        if not self.health_check(es):
            return
        proj = self._clone_project()

        def done(path):
            self._exported = True
            self._update_flow()
            if self.tray:
                self.tray.notify("Export finished", os.path.basename(path), lambda: open_path(path))
            r = QMessageBox.information(self, APP_NAME, f"Export finished:\n{path}",
                                        QMessageBox.Open | QMessageBox.Ok)
            if r == QMessageBox.Open:
                open_path(path)

        self.tasks.submit(f"Exporting {os.path.basename(es.output)}",
                          lambda p, c: export(proj, es, p, c), done, self._task_error("Export"))

    def health_check(self, es) -> bool:
        """Check the project before exporting with settings `es`: when there are issues, show them with their
        fixes. Returns whether to export."""
        from .health_dialog import HealthDialog

        def issues():
            return health.check(self.project, es.captions, es.use_proxies)

        if not issues():
            return True
        return HealthDialog(issues, lambda key: self._health_fix(key, es), self).exec() == QDialog.Accepted

    def _health_fix(self, key: str, es) -> str:
        """Apply a health check's fix (see health.check); returns what it did."""
        p = self.project
        if key.startswith("locate:"):
            f = p.footage.get(key.split(":", 1)[1])
            if f is None:
                return ""
            path, _ = QFileDialog.getOpenFileName(self, f"Where is {f.name}?", os.path.dirname(f.path))
            if not path:
                return ""
            f.path = path
            self.viewer.drop_decoders()
            p.dirty = True
            self.refresh_all()
            self._schedule_audio()
            return f"{f.name} found: {path}"
        if key == "drop_sounds":
            self.project_checkpoint()
            p.sounds = [x for x in p.sounds if os.path.exists(x.path)]
            self._sounds_changed()
            self.tl.refresh()
            return "The missing sounds are taken off (Undo brings them back)"
        if key == "drop_texts":
            self.project_checkpoint()
            p.texts = [x for x in p.texts if x.start < p.duration - 0.05]
            p.dirty = True
            self.refresh_all()
            return "The titles past the end are taken off (Undo brings them back)"
        if key == "transcribe":
            self._transcribe_edit()
            return "Transcribing in the background: export once it is done for captions on all the speech"
        if key == "measure":
            self.measure_shake()
            return "Measuring the shake in the background (the export would do it first otherwise)"
        if key == "faces":
            self._find_motion_faces(list(p.clips) + list(p.overlays))
            return "Finding the faces in the background"
        if key == "originals":
            es.use_proxies = False
            return "The export uses the original footage"
        return ""

    def _clone_project(self) -> Project:
        """Frozen copy so edits during a long export don't change what is rendered."""
        p = Project()
        p.settings = copy.deepcopy(self.project.settings)
        p.footage = {fid: copy.deepcopy(f) for fid, f in self.project.footage.items()}
        p.clips = [TimelineClip.from_dict(c.to_dict()) for c in self.project.clips]
        p.sounds = [SoundClip.from_dict(x.to_dict()) for x in self.project.sounds]
        p.texts = [TextClip.from_dict(x.to_dict()) for x in self.project.texts]
        p.overlays = [OverlayClip.from_dict(x.to_dict()) for x in self.project.overlays]
        return p

    # ================================================================== preview audio
    def _schedule_audio(self, delay: int = 700, keep_old: bool = False):
        """Mix the preview sound again (after `delay` ms of no more changes). keep_old: the previous mix plays
        meanwhile (volume, fades: still in time); else it is silenced (a trim or slip moved what is heard)."""
        self.viewer.invalidate_audio(keep_old)
        self.viewer.set_audio_pending(bool(self.project.clips))
        self._audio_timer.start(delay)

    def _render_audio(self):
        if not self.project.clips:
            self.viewer.set_audio(None)
            self.viewer.set_audio_pending(False)
            return
        self.viewer.set_audio_pending(True)
        self._audio_cancel.set()
        self._audio_cancel = threading.Event()
        cancel = self._audio_cancel
        self._audio_version += 1
        self._audio_done = False
        ver = self._audio_version
        proj = self._clone_project()
        use_proxies = self.project.settings.use_proxies
        # never the file the viewer has open: Windows can't replace it, and the new mix would be lost
        loaded = os.path.normcase(os.path.abspath(self.viewer._audio_path or ""))
        path = next(p for k in range(4) if os.path.normcase(
            p := str(cache_dir("tmp") / f"preview_audio_{(ver + k) % 4}.wav")) != loaded)

        def work():
            try:
                render_audio(proj, use_proxies, path, cancel)
                if not cancel.is_set():
                    self.audio_ready.emit(ver, path)
            except Exception as e:  # noqa: BLE001 - preview audio is best effort, but say why there is none
                if not cancel.is_set():
                    self.audio_failed.emit(ver, str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__)

        threading.Thread(target=work, daemon=True).start()
        QTimer.singleShot(AUDIO_SLOW_MS, lambda v=ver: self._audio_slow(v))

    def _audio_slow(self, ver: int):
        """A new preview mix is taking a while: say that playback has the earlier sound until it is ready (a music
        track or sound just added is heard once it is)."""
        if ver == self._audio_version and not self._audio_done:
            self._audio_said = True
            self.lbl_status.setText("Mixing the preview sound… until it is ready, playback has the sound from before "
                                    "the last change (music or sounds just added are heard once it is)")

    def _on_audio_ready(self, ver: int, path: str):
        if ver == self._audio_version:
            self._audio_done = True
            self.viewer.set_audio_pending(False)
            self.viewer.set_audio(path)   # also while playing: sound edits are heard right away
            if self._audio_said:
                self._audio_said = False
                self.lbl_status.setText("The preview sound is ready")

    def _on_audio_failed(self, ver: int, reason: str):
        if ver == self._audio_version:
            self._audio_done, self._audio_said = True, False
            self.viewer.set_audio_pending(False)
            self.lbl_status.setText(f"Playback has no sound: the preview sound couldn't be made ({reason[:200]})")

    def _on_play_state(self, playing: bool):
        if playing:
            self.sfx_panel.stop_preview()

    # ================================================================== updates
    UPDATE_CHECK_EVERY = 15 * 60    # seconds: restarting the app again and again doesn't ask GitHub each time

    def check_updates(self, manual: bool = True):
        """Look for a newer release on GitHub in the background. The check at startup runs every time the app
        starts (unless it started a few minutes ago) and says nothing unless there is an update."""
        if not manual:
            last = self.qs.value("update_last_auto", 0.0, type=float)
            if 0 <= time.time() - last < self.UPDATE_CHECK_EVERY:
                return
            self.qs.setValue("update_last_auto", time.time())
        if manual:
            self.lbl_status.setText("Checking for updates…")

        def work():
            try:
                self.update_checked.emit(updater.latest_release(), "", manual)
            except Exception as e:  # noqa: BLE001 - offline, rate limited...: never in the way
                self.update_checked.emit(None, str(e), manual)

        threading.Thread(target=work, daemon=True).start()

    def _on_update_checked(self, release, error: str, manual: bool):
        if release is None:
            if manual:
                QMessageBox.warning(self, APP_NAME, f"Could not check for updates:\n{error}")
                self.lbl_status.setText("Update check failed")
            return
        if not updater.is_newer(release.version):
            if manual:
                QMessageBox.information(self, APP_NAME, f"{APP_NAME} {version_string()} is the latest version.")
                self.lbl_status.setText("KangEdit is up to date")
            return
        if not manual and self.qs.value("update_skip", "", type=str) == release.version:
            return
        box = QMessageBox(QMessageBox.Information, "Update available",
                          f"<b>{APP_NAME} {release.version}</b> is available (you have {version_string()}).",
                          parent=self)
        kind = updater.install_kind()
        box.setInformativeText({"git": "This copy is a git checkout: Update runs “git pull”.",
                                "source": "Update downloads the new version, replaces the app's files and "
                                          "restarts KangEdit (your projects, settings and caches stay).",
                                "frozen": "Update downloads the new version and restarts KangEdit with it "
                                          "(your projects, settings and caches stay)."}[kind])
        if release.notes:
            box.setDetailedText(release.notes)
        b_update = box.addButton("Update now", QMessageBox.AcceptRole)
        b_page = box.addButton("Release page", QMessageBox.HelpRole)
        b_skip = box.addButton("Skip this version", QMessageBox.DestructiveRole)
        box.addButton("Later", QMessageBox.RejectRole)
        box.setDefaultButton(b_update)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_page:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(release.page))
        elif clicked is b_skip:
            self.qs.setValue("update_skip", release.version)
        elif clicked is b_update:
            self._install_update(release)

    def _install_update(self, release):
        if updater.install_kind() == "git":
            try:
                out = updater.git_pull()
            except updater.UpdateError as e:
                QMessageBox.warning(self, APP_NAME, f"git pull failed:\n{e}")
                return
            QMessageBox.information(self, APP_NAME, f"{out or 'Updated.'}\n\nRestart {APP_NAME} to use the new "
                                                    "version.")
            return
        if not self._confirm_discard():
            return

        def done(prepared):
            try:
                if prepared.kind == "source":
                    updater.install_source(prepared)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, APP_NAME, f"The update could not be installed:\n{e}")
                return
            QMessageBox.information(self, APP_NAME, f"{APP_NAME} {prepared.version} is ready: {APP_NAME} restarts "
                                                    "now to finish the update.")
            self.project.dirty = False
            updater.launch(prepared.command, prepared.env)
            self.close()
            QApplication.instance().quit()

        self.tasks.submit(f"Updating to {release.version}", lambda p, c: updater.prepare(release, p, c), done,
                          self._task_error("Update"))

    # ================================================================== misc
    def show_shortcuts(self):
        QMessageBox.information(self, "Keyboard shortcuts", (
            "Space — play / pause\n←/→ — previous / next frame (Shift: 10 frames)\n"
            "↑/↓ — previous / next clip\nHome / End — start / end\n"
            "S — split at playhead\nDel — delete the selected clips (or sound)\nCtrl+D — duplicate clip\n"
            "Ctrl+A — select all clips, video layers included (in the Media panel: all its media)\nR — other moment for the clip (best one nearby)\n"
            "Double-click footage in Media — play it in the viewer (Source) · I / O — mark in / out there\n"
            "H — choose the hook (point at a card to play it; ticked in the order it plays)\n"
            "F11 — full screen preview (Esc to leave)\n"
            "Ctrl+Shift+− / Ctrl+Shift+= — regenerate the auto edit shorter / longer\n"
            "Ctrl+Shift+R — reset the layout\n"
            "Ctrl+Z / Ctrl+Shift+Z — undo / redo\nShift+Z — fit timeline · Ctrl+= / Ctrl+− — zoom in / out\n"
            "Ctrl+Shift+B — color adjustments in the viewer on / off\n"
            "Ctrl+wheel on timeline — zoom · on the Media thumbnails — thumbnail size\n\n"
            "Timeline: drag a clip to reorder · drag its edges to trim ·\n"
            "Alt/Shift+drag (or Slip mode) to slide the moment inside the clip (its transcript shows:\n"
            "click a sentence to slip to it) ·\n"
            "Clip icons: 🎣 hook · 🏞 scene opener · 👋 opening / closing · 🔍 dynamic zoom · 〰 stabilized ·\n"
            "🎙 voice cleaned up · ♪ updating beside the time: the preview sound is being remade ·\n"
            "Ctrl+click adds a clip to the selection · Shift+click selects a range ·\n"
            "Speed: right-click a clip → Speed (slow / fast motion, 0.1× to 8×) or Speed ramp (Hero moment,\n"
            "Bullet time, Slow → fast…), or the Inspector's Speed group · Sound at this speed: synced\n"
            "(pitch kept or following the speed), normal speed (not synced) or muted\n"
            "SFX layer: drag sounds to move, edges to trim, right-click for volume, fades, fit to timeline,\n"
            "auto ducking\n"
            "Video layers (V2, V3… above the main track): drag footage or images from Media onto a layer\n"
            "(or right-click footage → Add on a video layer) · drag the clip in the viewer to place it,\n"
            "mouse wheel over it to resize · on the timeline drag it to move (up / down: another layer),\n"
            "its edges to trim · double-click it (or the 🧩 Layer tab) for size, crop, rotation, opacity,\n"
            "fades, speed, volume · the 🎨 Color tab grades the selected layer clip like a main clip ·\n"
            "higher layers are in front · right-click an empty layer to add a layer\n"
            "Multicam: ✨ Auto Edit → Multicam syncs camera angles by sound; Tools → Multicam layout\n"
            "changes their layout later · camera angles are linked to their main clip: dragging an angle\n"
            "slips the group, its edges trim it, R gives it another moment · Ctrl+L (🔗 Link) unlinks /\n"
            "links them · layouts include Switching (one angle at a time, cut to the best angle)\n"
            "Lanes: drag a lane's bottom edge to make it taller · Shift+wheel scrolls the lanes up / down"))

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        if self.viewer.is_fullscreen():
            self.viewer.toggle_fullscreen()
        autosave.clear_recovery()
        if self.d_right.isVisible():
            self.qs.setValue("refine_width", self.d_right.width())
        self.viewer.stop()
        self.tasks.shutdown()
        self._stop_refining()
        self._audio_cancel.set()
        self.viewer.shutdown()
        if self.tray:
            self.tray.hide()
        e.accept()


def _same_moment(a: TimelineClip, b: TimelineClip) -> bool:
    return a.footage_id == b.footage_id and a.src_in < b.src_out and b.src_in < a.src_out
