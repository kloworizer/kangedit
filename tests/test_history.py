"""Version history: versions kept on save, pruning, restore, and the File → Version history dialog."""

import importlib
import time

import pytest

from kangedit import history
from kangedit.models import Project, TimelineClip

from .test_edit_features import two_files


def _project(tmp_path, name="p", clips=1):
    footage, _ = two_files()
    fid = next(iter(footage))
    p = Project()
    p.footage = footage
    p.clips = [TimelineClip(fid, 2.0 * k, 2.0 * k + 1.5) for k in range(clips)]
    p.save(str(tmp_path / (name + Project.FILE_EXT)))
    return p


def _add_clip(p):
    fid = next(iter(p.footage))
    p.clips.append(TimelineClip(fid, 20.0 + len(p.clips), 21.0 + len(p.clips)))


def test_saving_keeps_versions_but_not_identical_ones(tmp_path):
    p = _project(tmp_path)
    assert history.record(p, now=1000.0) is not None
    assert history.record(p, now=1001.0) is None                 # nothing changed
    _add_clip(p)
    e = history.record(p, now=1002.0)
    assert e is not None and e["clips"] == 2 and e["duration"] == pytest.approx(2.5)
    assert [v["clips"] for v in history.versions(p.path)] == [2, 1]
    assert history.record(p, label="Before auto edit", now=1003.0) is None   # same content: label added
    assert history.versions(p.path)[0]["label"] == "Before auto edit"
    _add_clip(p)
    assert history.record(p, min_interval=300, now=1004.0) is None           # an auto-save too soon after
    assert history.record(p, min_interval=300, now=1400.0) is not None
    assert history.record(Project()) is None                     # untitled: no history


def test_pruning_rules():
    day = 86400.0
    now = 100 * day
    entries = [{"time": now - k * 3600} for k in range(60)]       # newest first, an hour apart
    assert len(history.to_prune(entries, now)) == 10              # beyond the newest 50
    old = [{"time": now - 40 * day - k} for k in range(15)]       # all older than 30 days
    dropped = history.to_prune(old, now)
    assert len(dropped) == 5 and dropped[0] is old[10]            # the newest 10 stay anyway


def test_pruning_deletes_files_beyond_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "MAX_VERSIONS", 3)
    p = _project(tmp_path)
    for k in range(5):
        _add_clip(p)
        history.record(p, now=1000.0 + k)
    kept = history.versions(p.path)
    assert len(kept) == 3 and len(list(history.history_dir(p.path).glob("*" + Project.FILE_EXT))) == 3


def test_index_survives_restart_and_projects_dont_mix(tmp_path):
    a, b = _project(tmp_path, "a"), _project(tmp_path, "b", clips=3)
    history.record(a, now=1000.0)
    _add_clip(a)
    history.record(a, now=1001.0)
    history.record(b, now=1002.0)
    fresh = importlib.reload(history)                             # as after a restart
    assert [v["clips"] for v in fresh.versions(a.path)] == [2, 1]
    assert [v["clips"] for v in fresh.versions(b.path)] == [3]
    assert len(fresh.load(a.path, fresh.versions(a.path)[1]).clips) == 1


def test_diff_summary(tmp_path):
    old = _project(tmp_path, clips=2)
    new = Project.load(old.path)
    _add_clip(new)
    text = history.diff_summary(old, new)
    assert "1 clip added" in text and "longer by 1.0 s" in text
    assert "same clips" in history.diff_summary(old, old)
    assert history.ago(1000.0, now=1300.0) == "5 min ago"


@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    w._set_recent([])
    yield w
    w.project.dirty = False
    w.close()


def test_save_and_restore_in_the_window(window, tmp_path, monkeypatch):
    p = _project(tmp_path, "w")
    window.set_project(Project.load(p.path))
    assert window.save_project()
    _add_clip(window.project)
    window.project.dirty = True
    time.sleep(0.01)
    assert window.save_project()
    entries = history.versions(p.path)
    assert [v["clips"] for v in entries] == [2, 1]
    assert window.restore_version(entries[1])
    assert len(window.project.clips) == 1 and window.project.path == p.path and window.project.dirty
    # the restore is itself reversible: the state before it is (already) the newest version
    assert history.versions(p.path)[0]["clips"] == 2 and history.versions(p.path)[0]["label"]
    assert window.save_project() and len(Project.load(p.path).clips) == 1

    window.project.dirty = False
    assert window.open_version_copy(history.versions(p.path)[1])
    assert window.project.path is None and len(window.project.clips) == 2


def test_dialog_lists_versions_and_explains_empty_history(window, tmp_path):
    from kangedit.ui.history_dialog import HistoryDialog
    dlg = HistoryDialog(Project())
    assert dlg.tree.topLevelItemCount() == 0 and not dlg.btn_restore.isEnabled()
    assert "saved" in dlg.head.text()

    p = _project(tmp_path, "d")
    history.record(p, now=time.time() - 600)
    _add_clip(p)
    history.record(p, label="Before auto edit")
    dlg = HistoryDialog(p)
    assert dlg.tree.topLevelItemCount() == 2 and dlg.btn_restore.isEnabled()
    assert dlg.tree.topLevelItem(0).text(4) == "Before auto edit"
    assert dlg.tree.topLevelItem(1).text(1) == "10 min ago"
    dlg.tree.setCurrentItem(dlg.tree.topLevelItem(1))
    assert "1 clip added" in dlg.diff.text()
    dlg.btn_restore.click()
    assert dlg.choice[0] == HistoryDialog.RESTORE and dlg.choice[1]["clips"] == 1
