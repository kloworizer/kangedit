"""Main window features, offscreen: recent projects, footage list views, viewer color modes."""

import os

import pytest

from kangedit.models import Project


@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    w._set_recent([])
    yield w
    w.project.dirty = False
    w.close()


def test_recent_projects_most_recent_first_without_duplicates(window, tmp_path):
    paths = []
    for i in range(12):
        p = str(tmp_path / f"p{i}.kgproj")
        Project().save(p)
        paths.append(p)
        window.load_project_file(p)
    window.load_project_file(paths[5])
    recent = window._recent_projects()
    assert len(recent) == window.MAX_RECENT
    assert recent[0] == os.path.abspath(paths[5])
    assert len({os.path.normcase(p) for p in recent}) == len(recent)
    window._fill_recent_menu()
    assert window.recent_menu.isEnabled()
    assert window.recent_menu.actions()[-1].text() == "Clear recent projects"


def test_saving_adds_to_recent(window, tmp_path):
    window.project.path = str(tmp_path / "saved.kgproj")
    assert window.save_project()
    assert window._recent_projects()[0] == os.path.abspath(window.project.path)


def test_footage_list_view_modes_are_remembered(window, qapp):
    window.bin.set_view_mode("thumbnails")
    assert window.bin.view_mode() == "thumbnails"
    from kangedit.ui.media_bin import MediaBin
    assert MediaBin(window.project).view_mode() == "thumbnails"
    window.bin.set_view_mode("details")


@pytest.mark.parametrize("mode, playing, color", [
    ("on", False, True), ("on", True, True),
    ("playback", False, True), ("playback", True, False),
    ("off", False, False), ("off", True, False),
])
def test_viewer_color_modes(window, mode, playing, color):
    window.viewer.set_adjust_mode(mode)
    window.viewer.playing = playing
    try:
        assert window.viewer._look().color is color
    finally:
        window.viewer.playing = False
    assert window.a_adjust[mode].isChecked()


def test_panels_stay_docked(window):
    from PySide6.QtWidgets import QDockWidget
    for dock in (window.d_bin, window.d_right):
        assert dock.features() == QDockWidget.NoDockWidgetFeatures
    menu = window.createPopupMenu()
    if menu is not None:   # the right-click menu can't hide the panels
        assert not {window.d_bin.toggleViewAction(), window.d_right.toggleViewAction()} & set(menu.actions())


def test_refine_tabs_scroll_and_fit_a_small_display(qapp):
    from PySide6.QtWidgets import QScrollArea
    from kangedit.ui.main_window import MainWindow
    window = MainWindow()
    try:
        tabs = window.right_tabs
        assert all(isinstance(tabs.widget(i), QScrollArea) for i in range(tabs.count()))
        window.show_refine(window.sfx_panel)
        assert tabs.currentWidget().widget() is window.sfx_panel
        window.show()
        qapp.processEvents()
        # A 1440x900 display minus the taskbar: the window must fit, with room left to resize the panels.
        hint = window.minimumSizeHint()
        parts = {"media": window.d_bin, "refine": window.d_right, "viewer": window.viewer,
                 "timeline": window.timeline, "media list": window.bin, "inspector": window.inspector.widget(),
                 "text": window.text_panel.widget(), "captions": window.caps_panel, "sounds": window.sfx_panel}
        widths = {k: max(w.minimumSizeHint().width(), w.minimumWidth()) for k, w in parts.items()}
        # Measured against a 9 pt UI font at 100 % (the alphabet is ~170 px wide in Segoe UI or DejaVu Sans).
        # Offscreen Qt on the Windows runners renders text about twice as large, so scale the limit with it.
        from PySide6.QtGui import QFontMetrics
        alphabet = QFontMetrics(qapp.font()).horizontalAdvance("abcdefghijklmnopqrstuvwxyz")
        scale = max(1.0, alphabet / 170)
        font = f"{qapp.font().family()} {qapp.font().pointSizeF():g} pt, alphabet {alphabet} px"
        assert hint.width() <= 1400 * scale and hint.height() <= 820 * scale, f"{hint}: {widths}; {font}"
    finally:
        window.project.dirty = False
        window.close()


def test_system_info_report():
    from kangedit import __version__, sysinfo
    text = sysinfo.report(gpu=False)
    assert __version__ in text
    for key in ("OS:", "Python:", "Qt:", "PySide6:", "numpy:", "Log file:"):
        assert key in text


def test_toolbars_wrap_instead_of_hiding_buttons(qapp):
    """On a small display the toolbar and the timeline's buttons go onto more rows, all visible inside the
    window (nothing in a » menu)."""
    from PySide6.QtCore import QRect
    from kangedit.ui.main_window import MainWindow
    window = MainWindow()
    try:
        window.resize(1340, 740)
        window.show()
        for _ in range(5):
            qapp.processEvents()
        tl = window.timeline
        buttons = [window.btn_import, window.btn_auto, window.btn_export, window.btn_help, window.cb_fit,
                   window.chk_proxy, tl.btn_undo, tl.btn_redo, tl.btn_move, tl.btn_slip, tl.btn_split,
                   tl.btn_delete, tl.btn_fade, tl.btn_hook, tl.btn_alt, tl.zoom, tl.btn_fit]
        area = QRect(0, 0, window.width(), window.height())
        for b in buttons:
            box = QRect(b.mapTo(window, b.rect().topLeft()), b.size())
            assert b.isVisible() and area.contains(box), (b.text() if hasattr(b, "text") else b, box, area)
    finally:
        window.project.dirty = False
        window.close()


def test_guided_tour_walks_every_step(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        w.resize(1400, 900)
        w.show()
        qapp.processEvents()
        w.show_tour()
        tour = w._tour
        titles = [s.title for s in tour.steps]
        assert "For YouTube and socials" in titles
        for k in range(len(tour.steps)):
            tour.show_step(k)
            assert not tour._target_rect().isNull() or tour.steps[k].target is None
        tour.close_tour(True)
        assert w._tour is None
    finally:
        w.project.dirty = False
        w.close()


def test_refine_panel_never_hides_a_tab(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        w.show()
        qapp.processEvents()
        tabs = w.right_tabs
        assert not tabs.usesScrollButtons()
        bar = tabs.tabBar()
        need = sum(bar.tabRect(i).width() for i in range(bar.count()))
        assert w.d_right.minimumWidth() >= need
        from PySide6.QtCore import Qt
        w.resizeDocks([w.d_right], [100], Qt.Horizontal)       # dragged as narrow as it goes
        qapp.processEvents()
        assert w.d_right.width() >= need
        assert all(bar.tabRect(i).right() <= bar.width() for i in range(bar.count()))   # every tab in sight
    finally:
        w.project.dirty = False
        w.close()
