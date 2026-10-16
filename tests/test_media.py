"""Media panel: bins, filters, Ctrl+A; the message for camera RAW files FFmpeg can't decode."""

import os
import time

import pytest

from kangedit import bins
from kangedit.analysis import undecodable_message
from kangedit.models import Footage, Project


def _footage(tmp_path, name, w=1920, h=1080, kind="video", rotation=0, created=""):
    path = tmp_path / name
    path.write_bytes(b"")
    return Footage(str(path), 10.0, w, h, 30.0, kind == "video", rotation=rotation, creation_time=created, kind=kind)


@pytest.fixture
def project(tmp_path):
    p = Project()
    for f in (_footage(tmp_path, "wide.mp4", created="2026-05-01T10:00:00.000000Z"),
              _footage(tmp_path, "phone.mp4", rotation=90, created="2026-05-02T09:00:00Z"),
              _footage(tmp_path, "square.jpg", 1080, 1080, kind="image"),
              _footage(tmp_path, "cinema.mov", 4096, 1716)):
        p.footage[f.id] = f
    return p


def _by_name(p, name):
    return next(f for f in p.footage.values() if f.name == name)


def test_filters_by_kind_and_shape(project):
    def names(kind="all", ratio="all", b=bins.ALL):
        return sorted(f.name for f in project.footage.values() if bins.visible(f, b, kind, ratio))
    assert names("image") == ["square.jpg"] and "square.jpg" not in names("video")
    assert names(ratio="portrait") == ["phone.mp4"]                 # rotated: shown upright
    assert names(ratio="landscape") == ["cinema.mov", "wide.mp4"]
    assert names(ratio="16:9") == ["wide.mp4"] and names(ratio="9:16") == ["phone.mp4"]
    assert names(ratio="square") == ["square.jpg"] and names(ratio="21:9") == ["cinema.mov"]


def test_bins_by_date_and_bin_editing(project, tmp_path):
    old = _by_name(project, "cinema.mov")
    stamp = time.mktime((2025, 12, 24, 12, 0, 0, 0, 0, -1))
    os.utime(old.path, (stamp, stamp))                              # no creation tag: the file's date
    counts = bins.bins_by_date(project)
    assert _by_name(project, "wide.mp4").bin == "2026-05-01" and _by_name(project, "phone.mp4").bin == "2026-05-02"
    assert old.bin == "2025-12-24" and sum(counts.values()) == 4
    bins.rename_bin(project, "2026-05-01", "Day 1")
    assert _by_name(project, "wide.mp4").bin == "Day 1" and "Day 1" in bins.all_bins(project)
    bins.add_bin(project, "Empty")
    assert "Empty" in bins.all_bins(project)                       # kept though nothing is in it
    bins.delete_bin(project, "Day 1")
    assert _by_name(project, "wide.mp4").bin == "" and "Day 1" not in bins.all_bins(project)
    path = str(tmp_path / "p.kgproj")
    project.save(path)
    q = Project.load(path)
    assert _by_name(q, "phone.mp4").bin == "2026-05-02" and "Empty" in q.settings.bins


@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.project.dirty = False
    w.close()


def _shown(window):
    b = window.bin
    return sorted(b.tree.topLevelItem(i).text(0) for i in range(b.tree.topLevelItemCount()))


def test_media_panel_bins_filters_and_select_all(window, project, qapp, monkeypatch):
    window.set_project(project)
    b = window.bin
    b.cb_kind.setCurrentIndex(b.cb_kind.findData("video"))
    assert _shown(window) == ["cinema.mov", "phone.mp4", "wide.mp4"]
    b.cb_ratio.setCurrentIndex(b.cb_ratio.findData("portrait"))
    assert _shown(window) == ["phone.mp4"]
    b.cb_kind.setCurrentIndex(0)
    b.cb_ratio.setCurrentIndex(0)
    ids = [_by_name(project, "wide.mp4").id, _by_name(project, "cinema.mov").id]
    b.move_to_bin(ids, "Interviews")
    assert b.current_bin() == bins.ALL and "📁 Interviews (2)" in [b.cb_bin.itemText(i) for i in range(b.cb_bin.count())]
    b.show_bin("Interviews")
    assert _shown(window) == ["cinema.mov", "wide.mp4"]
    assert b.tree.topLevelItem(0).text(7) == "Interviews"
    # Ctrl+A with the focus in the Media panel (given directly: a test window may not get the OS focus)
    monkeypatch.setattr("kangedit.ui.main_window.QApplication.focusWidget", staticmethod(lambda: b.tree))
    b.tree.clearSelection()
    window._select_all()                                        # Ctrl+A in the Media panel: what it shows
    assert sorted(b.selected_ids()) == sorted(ids)
    window.undo()                                               # moving into a bin is one undo step
    assert window.project.footage[ids[0]].bin == ""
    b.bins_by_date()
    assert "Sorted 4 item(s)" in window.lbl_status.text()
    b.clear_bins()                                              # right-click → Bins → Clear all bins
    assert bins.all_bins(window.project) == [] and len(_shown(window)) == 4
    assert "Cleared" in window.lbl_status.text()
    window.undo()
    assert bins.all_bins(window.project)


def test_camera_raw_gets_a_clear_message(tmp_path):
    f = _footage(tmp_path, "BURANO XOCN LT Clip 3.mxf")
    f.codec = ""
    msg = undecodable_message(f)
    assert "X-OCN" in msg and "can't decode" in msg and "ProRes" in msg
    assert "Blackmagic RAW" in undecodable_message(_footage(tmp_path, "a.braw"))
    ok = _footage(tmp_path, "b.mp4")
    ok.codec = "h264"
    assert undecodable_message(ok) == "Could not decode frames from b.mp4"
