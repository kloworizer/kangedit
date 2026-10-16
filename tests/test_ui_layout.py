"""Auto Edit dialog layout and target in minutes, using (nearly) every footage, Media panel thumbnails that grow
with the panel, and a Refine panel that can be made narrow."""

import numpy as np
from kangedit.analysis import Analysis
from kangedit.autoedit import AutoEditSettings, select
from kangedit.models import Footage, Project


def test_target_length_in_minutes(qapp):
    from kangedit.ui.dialogs import DurationSpinBox, parse_duration
    sp = DurationSpinBox()
    sp.setRange(5, 6 * 3600)
    for v, text in ((45, "45 s"), (180, "3 min"), (750, "12 min 30 s"), (3900, "1 h 5 min")):
        sp.setValue(v)
        assert sp.text() == text and parse_duration(text) == v
    assert parse_duration("12:30") == 750 and parse_duration("1:02:30") == 3750 and parse_duration("200") == 200
    assert parse_duration("20 min") == 1200 and parse_duration("90s") == 90 and parse_duration("abc") is None
    sp.lineEdit().setText("12:30")
    sp.interpretText()
    assert sp.value() == 750


def test_auto_edit_dialog_two_columns_on_a_wide_screen(qapp, monkeypatch):
    from kangedit.ui import dialogs
    p = Project()
    f = Footage("a.mp4", 60.0, 1920, 1080, 30.0, True)
    p.footage = {f.id: f}
    monkeypatch.setattr(dialogs, "TWO_COLUMNS_WIDTH", 0)          # any screen counts as wide
    d = dialogs.AutoEditDialog(p, [], AutoEditSettings())
    d.show()
    qapp.processEvents()
    assert d.two_columns and d.grp_footage.x() > d.sp_target.parentWidget().x() + 100
    d.close()
    monkeypatch.setattr(dialogs, "TWO_COLUMNS_WIDTH", 100000)     # a small screen: one column
    d = dialogs.AutoEditDialog(p, [], AutoEditSettings())
    assert not d.two_columns
    d.close()


def _an(dur, seed):
    n = int(dur * 4)
    t = np.arange(n, dtype=np.float32) / 4
    rng = np.random.default_rng(seed)
    return Analysis(times=t, sharp=rng.random(n, dtype=np.float32) * 5, activity=rng.random(n, dtype=np.float32),
                    motion=np.ones(n, np.float32), jerk=np.zeros(n, np.float32), brightness=np.full(n, 0.5, np.float32),
                    clipped=np.zeros(n, np.float32), colorful=np.full(n, 20.0, np.float32),
                    faces=np.zeros(n, np.float32), audio=np.full(n, -25.0, np.float32), cuts=np.zeros(0, np.float32),
                    embed=rng.random((n, 144), dtype=np.float32), thumbs=np.zeros((0, 4, 4, 3), np.uint8),
                    thumb_times=np.zeros(0, np.float32), wave=np.zeros(10, np.float32))


def test_use_every_footage_uses_nearly_all_of_it():
    files = [Footage(f"c{k:03d}.mp4", 30.0, 1920, 1080, 30.0, True) for k in range(313)]
    footage = {f.id: f for f in files}
    analyses = {f.id: _an(30.0, k) for k, f in enumerate(files)}
    s = AutoEditSettings(video_format="long", target_duration=900.0, clip_min=4.0, clip_max=10.0, min_gap=8.0,
                         order="chronological", use_every_footage=True)
    clips = select(footage, analyses, s, [f.id for f in files])
    used = len({c.footage_id for c in clips})
    assert used >= 0.9 * len(files)                                   # was 130 of 313 with full-length clips
    s.use_every_footage = False
    assert len({c.footage_id for c in select(footage, analyses, s, [f.id for f in files])}) < used


def test_grid_thumbnails_grow_with_the_panel(qapp):
    from kangedit.ui.media_bin import MediaBin
    p = Project()
    f = Footage("a.mp4", 30.0, 1920, 1080, 30.0, True)
    p.footage = {f.id: f}
    b = MediaBin(p)
    an = _an(30.0, 1)
    an.thumbs = np.zeros((30, 72, 128, 3), np.uint8)                  # thumbnails are stored 128 px wide
    b.analyses = {f.id: an}
    b.refresh()
    b.stack.setCurrentWidget(b.grid)
    for width in (400, 900):
        b.resize(width, 600)
        b.show()
        qapp.processEvents()
        b._apply_thumb_size()
        b._refresh_grid_icons()
        icon = b.grid.item(0).icon()
        want = b.grid.iconSize()
        got = icon.availableSizes()[0]
        assert abs(got.width() - want.width()) <= 3                     # filled to the cell, however wide
    assert want.width() > 300
    b.close()


def test_refine_panel_can_be_made_narrow(qapp):
    from kangedit.ui.main_window import REFINE_MIN_WIDTH, MainWindow
    w = MainWindow()
    try:
        assert w.d_right.minimumWidth() == REFINE_MIN_WIDTH <= 300
        assert w._right_width >= 370                                   # but starts wide enough for every tab
    finally:
        w.project.dirty = False
        w.close()
