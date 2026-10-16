"""Recording dates (the oldest one a file carries), scenes as time groups, scenic openers and transitions
between scenes, and clearing all bins."""

import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pytest

from kangedit import bins, scenes
from kangedit.analysis import Analysis
from kangedit.autoedit import AutoEditSettings, apply_edit_style
from kangedit.bookends import story_order
from kangedit.models import Footage, Project, TimelineClip

RATE = scenes.SAMPLE_RATE


def _file(tmp_path, name, tag="", mtime=None, dur=120.0):
    path = tmp_path / name
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return Footage(str(path), dur, 1920, 1080, 30.0, True, creation_time=tag)


def _iso(y, mo, d, h=12, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _ts(y, mo, d, h=12, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()


def test_recorded_time_is_the_oldest_believable_date(tmp_path):
    copied = _file(tmp_path, "copied.mp4", tag=_iso(2026, 3, 2), mtime=_ts(2026, 3, 1))   # tag later than the file
    assert scenes.recorded_time(copied) == pytest.approx(_ts(2026, 3, 1))
    tagged = _file(tmp_path, "tagged.mp4", tag=_iso(2026, 3, 5), mtime=_ts(2026, 4, 1))
    assert scenes.recorded_time(tagged) == pytest.approx(_ts(2026, 3, 5))
    unset = _file(tmp_path, "unset.mp4", tag=_iso(1970, 1, 1), mtime=_ts(2026, 3, 9))  # camera clock never set
    assert scenes.recorded_time(unset) == pytest.approx(_ts(2026, 3, 9))
    gone = Footage(str(tmp_path / "missing.mp4"), 5.0, 1920, 1080, 30.0, True)
    assert scenes.recorded_time(gone) is None
    # story order (where the opening and closing are looked for) and bins by date use the same date
    assert story_order({f.id: f for f in (tagged, copied, unset)}, [tagged.id, unset.id, copied.id]) == \
        [copied.id, tagged.id, unset.id]
    assert bins.recorded_date(copied) == datetime.fromtimestamp(_ts(2026, 3, 1)).date().isoformat()


def test_clear_all_bins(tmp_path):
    p = Project()
    a, b = _file(tmp_path, "a.mp4"), _file(tmp_path, "b.mp4")
    p.footage = {a.id: a, b.id: b}
    bins.move_to_bin(p, [a.id], "Day 1")
    bins.add_bin(p, "Empty")
    assert bins.clear_bins(p) == 1
    assert bins.all_bins(p) == [] and a.bin == "" and set(p.footage) == {a.id, b.id}


def test_time_groups(tmp_path):
    day1 = [_file(tmp_path, f"d1_{k}.mp4", tag=_iso(2026, 5, 1, 9, 10 * k), dur=300) for k in range(3)]
    day2 = [_file(tmp_path, f"d2_{k}.mp4", tag=_iso(2026, 5, 2, 15, 10 * k), dur=300) for k in range(2)]
    for f in day1 + day2:
        os.utime(f.path, (time.time(), time.time()))
    footage = {f.id: f for f in day1 + day2}
    order = story_order(footage, list(footage))
    g = scenes.time_groups(footage, order)
    assert [g[f.id] for f in day1] == [0, 0, 0] and [g[f.id] for f in day2] == [1, 1]


def _analysis(dur, scenic, seed):
    """Talking to the camera (a big face, loud speech, shaky hand-held), except during `scenic` (a, b): a quiet,
    sharp, colourful, gently panning view with nobody in it."""
    n = int(dur * RATE)
    t = np.arange(n, dtype=np.float32) / RATE
    view = (t >= scenic[0]) & (t < scenic[1])
    rng = np.random.default_rng(seed)
    return Analysis(
        times=t, sharp=np.where(view, 8.0, 4.0).astype(np.float32), activity=np.full(n, 5.0, np.float32),
        motion=np.where(view, 4.0, 1.0).astype(np.float32), jerk=np.where(view, 0.1, 3.0).astype(np.float32),
        brightness=np.full(n, 0.5, np.float32), clipped=np.zeros(n, np.float32),
        colorful=np.where(view, 40.0, 10.0).astype(np.float32), faces=np.where(view, 0.0, 0.08).astype(np.float32),
        audio=np.where(view, -45.0, -15.0).astype(np.float32), cuts=np.zeros(0, np.float32),
        embed=rng.random((n, 144), dtype=np.float32), thumbs=np.zeros((0, 4, 4, 3), np.uint8),
        thumb_times=np.zeros(0, np.float32), wave=np.zeros(10, np.float32))


def test_each_scene_opens_with_a_scenic_shot_and_transitions_mark_scenes(tmp_path):
    a1 = _file(tmp_path, "a1.mp4", tag=_iso(2026, 5, 1, 9), dur=60)
    a2 = _file(tmp_path, "a2.mp4", tag=_iso(2026, 5, 1, 9, 5), dur=60)
    b1 = _file(tmp_path, "b1.mp4", tag=_iso(2026, 5, 2, 15), dur=60)      # the next day: a new scene
    b2 = _file(tmp_path, "b2.mp4", tag=_iso(2026, 5, 2, 15, 3), dur=60)
    footage = {f.id: f for f in (a1, a2, b1, b2)}
    analyses = {a1.id: _analysis(60, (40, 55), 1), a2.id: _analysis(60, (5, 15), 2),
                b1.id: _analysis(60, (0, 0), 3), b2.id: _analysis(60, (20, 32), 4)}
    order = story_order(footage, list(footage))
    groups = scenes.time_groups(footage, order)
    clips = [TimelineClip(a1.id, 2.0, 8.0, label="opening"), TimelineClip(a2.id, 30.0, 36.0),
             TimelineClip(b1.id, 10.0, 16.0), TimelineClip(b2.id, 40.0, 46.0)]
    n = scenes.add_scene_openers(clips, groups, analyses, footage, 6.0, gap=2.0)
    assert n == 1                                                   # the first scene opens the timeline itself
    op = clips[2]
    assert op.label == "scene" and op.footage_id == b2.id and 20.0 <= op.src_in and op.src_out <= 32.0
    assert [c.footage_id for c in clips] == [a1.id, a2.id, b2.id, b1.id, b2.id]
    s = AutoEditSettings(transitions="scenes", transition_style="dissolve", jl_cuts=False)
    apply_edit_style(clips, footage, analyses, s, groups=groups)
    # out of the opening greeting and into the new scene, not between files of one scene
    assert [c.transition for c in clips[1:]] == ["dissolve", "dissolve", "", ""]
    # nothing scenic in a scene: no opener (better none than a talking head)
    plain = {fid: _analysis(60, (0, 0), 9) for fid in footage}
    clips = [TimelineClip(a1.id, 2.0, 8.0), TimelineClip(b1.id, 10.0, 16.0)]
    assert scenes.add_scene_openers(clips, groups, plain, footage, 6.0) == 0 and len(clips) == 2


def test_long_form_auto_edit_opens_scenes(qapp, tmp_path, monkeypatch):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        a1 = _file(tmp_path, "a1.mp4", tag=_iso(2026, 5, 1, 9), dur=60)
        b1 = _file(tmp_path, "b1.mp4", tag=_iso(2026, 5, 2, 15), dur=60)
        p = Project()
        p.footage = {a1.id: a1, b1.id: b1}
        w.set_project(p)
        an = {a1.id: _analysis(60, (40, 55), 1), b1.id: _analysis(60, (5, 18), 2)}
        monkeypatch.setattr("kangedit.ui.main_window.load_or_analyze", lambda f, **kw: an[f.id])
        monkeypatch.setattr(w, "queue_analysis", lambda ids, force=False: None)
        monkeypatch.setattr(w.tasks, "submit",
                            lambda name, job, done, err=None: done(job(lambda *a: None, threading.Event())))
        s = AutoEditSettings(video_format="long", target_duration=40.0, clip_min=4.0, clip_max=8.0, min_gap=2.0,
                             order="chronological", transitions="scenes", transition_style="dissolve",
                             bookends=False)
        w._run_auto_edit(s, [a1.id, b1.id], None)
        clips = w.project.clips
        firsts_b = next(k for k, c in enumerate(clips) if c.footage_id == b1.id)
        assert clips[firsts_b].label == "scene" and 5.0 <= clips[firsts_b].src_in < clips[firsts_b].src_out <= 18.0
        assert clips[firsts_b].transition == "dissolve"
        assert all(c.transition == "" for k, c in enumerate(clips[1:], 1) if k != firsts_b)
        assert "scenic shot" in w.lbl_status.text()
    finally:
        w.project.dirty = False
        w.close()


def test_broll_joins_a_scene_without_splitting_it(tmp_path):
    a = _file(tmp_path, "a.mp4", tag=_iso(2026, 5, 1, 1, 0), dur=300)
    b = _file(tmp_path, "b.mp4", tag=_iso(2026, 5, 1, 1, 20), dur=300)
    r = _file(tmp_path, "r.mp4", tag=_iso(2026, 5, 1, 1, 10), dur=60)      # a view 2 km away, in between
    other_day = _file(tmp_path, "o.mp4", tag=_iso(2026, 5, 9, 1, 0), dur=60)
    for f in (a, b, r, other_day):
        os.utime(f.path, (time.time(), time.time()))
    a.gps, b.gps, r.gps, other_day.gps = [-8.65, 115.13], [-8.65, 115.13], [-8.65, 115.15], []
    body = {a.id: a, b.id: b}
    groups = scenes.join_scenes(scenes.time_groups(body, [a.id, b.id]), body, {r.id: r, other_day.id: other_day})
    assert groups == {a.id: 0, b.id: 0, r.id: 0}                  # one scene; the far-off day is left out
