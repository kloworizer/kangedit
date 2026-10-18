"""YouTube chapters from the edit's scenes (or parts of it), as description lines, and their title cards."""

import os
import time

from kangedit import chapters
from kangedit.models import Footage, Project, TimelineClip


def _file(tmp_path, name, day, dur=120.0, words=None):
    path = tmp_path / name
    path.write_bytes(b"x")
    t = time.mktime((2026, 5, day, 10, 0, 0, 0, 0, -1))
    os.utime(path, (t, t))
    f = Footage(str(path), dur, 1920, 1080, 30.0, True, creation_time=f"2026-05-{day:02d}T10:00:00Z")
    f.transcript = words or []
    return f


def test_stamps_and_parsing():
    assert chapters.stamp(0) == "0:00" and chapters.stamp(83.9) == "1:23" and chapters.stamp(3725) == "1:02:05"
    got = chapters.parse("0:00 Intro\n1:23 The beach\nnot a chapter\n1:02:05 Night")
    assert [(c.start, c.title) for c in got] == [(0, "Intro"), (83, "The beach"), (3725, "Night")]


def test_a_chapter_for_each_scene(tmp_path):
    d1 = _file(tmp_path, "d1.mp4", 1, words=[{"s": 5.0, "e": 5.5, "w": "Good"}, {"s": 5.6, "e": 6.0, "w": "morning!"}])
    d2 = _file(tmp_path, "d2.mp4", 2, words=[{"s": 31.0, "e": 31.4, "w": "The"}, {"s": 31.5, "e": 32.0, "w": "beach."}])
    d3 = _file(tmp_path, "d3.mp4", 3)
    p = Project()
    p.footage = {f.id: f for f in (d1, d2, d3)}
    p.clips = [TimelineClip(d2.id, 60.0, 63.0, label="hook"),                     # the hook: part of the first
               TimelineClip(d1.id, 4.0, 20.0), TimelineClip(d1.id, 40.0, 50.0),
               TimelineClip(d2.id, 30.0, 45.0), TimelineClip(d2.id, 70.0, 75.0),
               TimelineClip(d3.id, 10.0, 22.0), TimelineClip(d3.id, 50.0, 53.0)]
    got, how = chapters.make(p)
    assert how == "scenes" and [c.start for c in got] == [0.0, 29.0, 49.0]
    assert got[0].title == "Intro" and got[1].title.endswith("The beach.")
    assert got[2].title.startswith(time.strftime("%a", time.localtime(time.mktime((2026, 5, 3, 10, 0, 0, 0, 0, -1)))))
    assert chapters.text(got).splitlines()[1].startswith("0:29 ")


def test_too_few_scenes_split_into_parts(tmp_path):
    f = _file(tmp_path, "one.mp4", 1, dur=600.0)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, k * 20.0, k * 20.0 + 15.0) for k in range(12)]      # 3 minutes, one scene
    got, how = chapters.make(p)
    assert how == "parts" and len(got) >= 3 and got[0].start == 0.0
    assert all(b.start - a.start >= chapters.MIN_CHAPTER for a, b in zip(got, got[1:]))
    assert got[0].title == "Part 1"
    p.clips = p.clips[:1]                                                           # 15 s: no chapters
    assert chapters.make(p) == ([], "")
    p.clips = [TimelineClip(f.id, 0.0, 300.0), TimelineClip(f.id, 300.0, 600.0)]   # 10 minutes, 2 clips
    assert chapters.make(p) == ([], "")                                             # 2 parts: YouTube ignores


def test_title_cards(qapp, tmp_path):
    from kangedit.ui.chapters_dialog import ChaptersDialog
    f = _file(tmp_path, "one.mp4", 1, dur=600.0)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, k * 20.0, k * 20.0 + 15.0) for k in range(12)]
    d = ChaptersDialog(p)
    d.text.setPlainText("0:00 Start\n0:45 Middle bit\n2:00 The end")
    d.add_cards()
    assert [(x.start, x.text) for x in p.texts] == [(45.0, "Middle bit"), (120.0, "The end")]
    d.add_cards()                                     # again: replaced, not doubled
    assert len(p.texts) == 2
    p.undo()
    assert len(p.texts) == 2                          # the second add's undo brings back the first's cards
    d = ChaptersDialog(p)
    d.text.setPlainText("")                           # every line taken out: the cards go
    d.add_cards()
    assert not p.texts and d.added == 0 and d.changed  # (the main window refreshes the timeline)
