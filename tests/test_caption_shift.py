"""Moving captions in time (right words, wrong timing): a caption and its clip's later ones, dragged or
nudged on the timeline; and transcribing footage again from scratch."""

import pytest

from kangedit.captions import build_captions, shift_captions, shift_group
from kangedit.models import Footage, Project, TimelineClip


def _project():
    f = Footage("talk.mp4", 60.0, 1920, 1080, 30.0, True)
    f.transcript = [{"s": 0.5, "e": 0.9, "w": "Before."},                            # not in the clip
                    {"s": 2.0, "e": 2.4, "w": "Satu."}, {"s": 4.0, "e": 4.4, "w": "Dua."},
                    {"s": 6.0, "e": 6.4, "w": "Tiga."},
                    {"s": 12.0, "e": 12.4, "w": "After."}]                           # after the clip
    g = Footage("other.mp4", 60.0, 1920, 1080, 30.0, True)
    g.transcript = [{"s": 1.0, "e": 1.5, "w": "Lain."}]
    p = Project()
    p.footage = {f.id: f, g.id: g}
    p.clips = [TimelineClip(f.id, 1.0, 8.0), TimelineClip(g.id, 0.0, 3.0)]
    return p, f, g


def test_a_caption_and_the_clips_later_ones_move():
    p, f, g = _project()
    caps = build_captions(p)
    texts = [c.text for c in caps]
    k = texts.index("Dua.")
    assert [caps[i].text for i in shift_group(caps, k)] == ["Dua.", "Tiga."]     # not "Satu.", not the next clip
    moved = shift_captions(p, caps, k, 0.5)
    assert moved == pytest.approx(0.5)
    caps2 = {c.text: c.start for c in build_captions(p)}
    old = {c.text: c.start for c in caps}
    assert caps2["Dua."] == pytest.approx(old["Dua."] + 0.5) and caps2["Tiga."] == pytest.approx(old["Tiga."] + 0.5)
    assert caps2["Satu."] == old["Satu."] and caps2["Lain."] == old["Lain."]
    assert f.transcript[0]["s"] == 0.5 and f.transcript[-1]["s"] == 12.0         # outside the clip: untouched


def test_the_move_stops_at_the_words_around():
    p, f, _g = _project()
    caps = build_captions(p)
    k = [c.text for c in caps].index("Dua.")
    moved = shift_captions(p, caps, k, -5.0)                                    # far left: up to "Satu."
    assert moved == pytest.approx(-(4.0 - 2.4))
    assert f.transcript[2]["s"] == pytest.approx(2.4)
    words = [w["s"] for w in f.transcript]
    assert words == sorted(words)                                               # the transcript keeps its order


def test_drag_and_nudge_in_the_window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p, f, _g = _project()
        w.set_project(p)
        w.refresh_all()
        k = [c.text for c in w.captions].index("Dua.")
        start = w.captions[k].start
        assert w.tl.captions == w.captions
        w.tl.captions_shifted.emit(k, 0.1)                                       # the menu's "0.1 s later"
        assert w.captions[k].start == pytest.approx(start + 0.1) and "Moved 2 caption(s) 0.10 s later" in \
            w.lbl_status.text()
        w.undo()
        assert w.captions[k].start == pytest.approx(start)
        # the drag: press on the caption, move, release
        tl = w.tl
        tl._drag = {"kind": "cap_shift", "cap": k, "x0": 100.0, "active": True, "dt": 0.3,
                    "group": shift_group(w.captions, k)}
        tl.update()
        tl.repaint()                                                              # drawn where the drag puts them
        tl.captions_shifted.emit(k, 0.3)
        assert w.captions[k].start == pytest.approx(start + 0.3)
    finally:
        w.project.dirty = False
        w.close()


def test_transcribe_again_from_scratch(qapp, monkeypatch):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p, f, g = _project()
        w.set_project(p)
        queued = []
        monkeypatch.setattr(w, "queue_transcription",
                            lambda ids, force=False, use_cache=None: queued.append((list(ids), force, use_cache)))
        w.bin_action("transcribe_force", [f.id])                                  # Media panel
        w.clip_action("transcribe_force", p.clips[1].id)                          # timeline clip
        assert queued == [([f.id], True, False), ([g.id], True, False)]
    finally:
        w.project.dirty = False
        w.close()
