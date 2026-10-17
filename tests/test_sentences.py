"""Long-form cuts on whole sentences, no repeated moments, the merged transcript, the offer to transcribe after
an import, and Show in Media panel."""

import numpy as np

from kangedit import polish
from kangedit.autoedit import AutoEditSettings
from kangedit.captions import merged_transcript
from kangedit.models import Footage, Project, TimelineClip


def _words(text_times):
    return [{"w": w, "s": s, "e": e} for w, s, e in text_times]


def _talk(name="talk.mp4", dur=60.0):
    f = Footage(name, dur, 1920, 1080, 30.0, True)
    # "Halo semua." 1.0-2.0 | "Hari ini kita jalan ke pantai." 2.5-5.5 | "Seru sekali!" 6.0-7.0 | (pause)
    # "Ayo berangkat" 9.0-10.0
    f.transcript = _words([("Halo", 1.0, 1.4), ("semua.", 1.5, 2.0), ("Hari", 2.5, 2.8), ("ini", 2.9, 3.1),
                           ("kita", 3.2, 3.5), ("jalan", 3.6, 4.0), ("ke", 4.1, 4.3), ("pantai.", 4.4, 5.5),
                           ("Seru", 6.0, 6.4), ("sekali!", 6.5, 7.0), ("Ayo", 9.0, 9.4), ("berangkat", 9.5, 10.0)])
    return f


def test_sentences_end_on_punctuation_or_a_pause():
    f = _talk()
    assert polish.sentences(f.transcript) == [(1.0, 2.0), (2.5, 5.5), (6.0, 7.0), (9.0, 10.0)]


def test_clips_grow_or_shrink_to_whole_sentences():
    f = _talk()
    fp = {f.id: f}
    c = TimelineClip(f.id, 3.0, 6.3)                   # starts mid-sentence, ends inside "Seru"
    assert polish.snap_to_sentences([c], fp, max_len=20.0) == 1
    assert c.src_in < 2.5 and c.src_out > 7.0           # both sentences taken in whole
    c = TimelineClip(f.id, 3.0, 6.3)                   # too long to take both in: the one sentence it cuts into
    polish.snap_to_sentences([c], fp, max_len=3.6)      # at the start is taken whole, the one at the end left out
    assert (c.src_in, c.src_out) == (2.44, 5.62)
    c = TimelineClip(f.id, 3.0, 6.3)
    polish.snap_to_sentences([c], fp, max_len=1.0)      # no whole sentence fits: unchanged
    assert (c.src_in, c.src_out) == (3.0, 6.3)
    c = TimelineClip(f.id, 1.2, 6.3)
    polish.snap_to_sentences([c], fp, max_len=4.4)
    assert 2.3 < c.src_in < 2.5 and 5.5 < c.src_out < 5.8   # "Hari ini … pantai." alone
    # never onto a moment another clip shows
    a, b = TimelineClip(f.id, 3.0, 6.3), TimelineClip(f.id, 0.2, 1.9)
    polish.snap_to_sentences([a, b], fp, max_len=20.0, gap=0.5)
    assert a.src_in >= b.src_out + 0.5 - 1e-6


def test_long_form_finish_uses_sentences_short_form_does_not():
    f = _talk()
    fp = {f.id: f}
    for fmt, whole in (("long", True), ("short", False)):
        s = AutoEditSettings(video_format=fmt, clip_min=2.0, clip_max=6.0, min_gap=1.0, pacing="steady",
                             cut_on_pauses=False, shot_variety=False, even_loudness=False, beat_sync=False)
        clips, _ = polish.finish([TimelineClip(f.id, 3.0, 6.3)], fp, {}, s)
        assert (clips[0].src_in < 2.5) == whole


def test_repeats_are_dropped_jump_cuts_kept():
    f = Footage("a.mp4", 100.0, 1920, 1080, 30.0, True)
    clips = [TimelineClip(f.id, 10.0, 14.0), TimelineClip(f.id, 14.5, 18.0),      # jump cuts: kept
             TimelineClip(f.id, 18.3, 19.0),
             TimelineClip(f.id, 50.0, 54.0), TimelineClip(f.id, 20.0, 24.0),      # 20 s is 2 s after 18: repeat
             TimelineClip(f.id, 52.0, 56.0)]                                       # overlaps 50-54
    kept, n = polish.drop_repeats(clips, 12.0)
    assert n == 2 and [c.src_in for c in kept] == [10.0, 14.5, 18.3, 50.0]
    p = Project()
    p.footage, p.clips = {f.id: f}, kept
    assert not p.conflicts(12.0)


def test_merged_transcript_in_recording_order(tmp_path):
    import os
    a, b = _talk(str(tmp_path / "b_later.mp4")), _talk(str(tmp_path / "a_first.mp4"))
    for f, t in ((a, 1_700_000_000), (b, 1_600_000_000)):       # in the past: macOS also counts the creation date
        open(f.path, "wb").close()
        os.utime(f.path, (t, t))
    b.transcript = _words([("Pagi.", 0.5, 1.0)])
    none = Footage(str(tmp_path / "c.mp4"), 5.0, 1920, 1080, 30.0, True)
    text = merged_transcript([a, b, none])
    assert text.index("a_first.mp4") < text.index("b_later.mp4") and "c.mp4" not in text
    assert "Hari ini kita jalan ke pantai." in text.splitlines()
    plain = merged_transcript([a, b], headers=False)
    assert plain.splitlines()[0] == "Pagi." and "──" not in plain


def test_transcript_dialog_copies_everything(qapp):
    from PySide6.QtWidgets import QApplication

    from kangedit.ui.dialogs import TranscriptDialog
    p = Project()
    f, g = _talk(), Footage("quiet.mp4", 5.0, 1920, 1080, 30.0, True)
    p.footage = {f.id: f, g.id: g}
    asked = []
    d = TranscriptDialog(p, transcribe=asked.append)
    assert "Halo semua." in d.text.toPlainText() and "1 of 2" in d.lbl.text()
    d.copy_all()
    assert QApplication.clipboard().text() == d.text.toPlainText()
    d._transcribe_rest()
    assert asked == [[g.id]]
    d.close()


def test_import_offers_to_transcribe_and_show_in_media(qapp, monkeypatch, no_blocking_message_boxes):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p = Project()
        f, g, done = _talk("a.mp4"), Footage("b.mp4", 9.0, 1920, 1080, 30.0, True), _talk("c.mp4")
        f.transcript = None
        p.footage = {x.id: x for x in (f, g, done)}
        w.set_project(p)
        queued = []
        monkeypatch.setattr(w, "queue_transcription", lambda ids, force=False: queued.append(ids))
        monkeypatch.setattr(w, "_ask_transcribe", lambda n: (True, False))
        w.a_ask_transcribe.setChecked(True)
        w.offer_transcription()
        assert queued == [[f.id, g.id]]
        monkeypatch.setattr(w, "_ask_transcribe", lambda n: (False, True))      # "don't ask again"
        w.offer_transcription()
        assert not w.a_ask_transcribe.isChecked() and len(queued) == 1
        w.offer_transcription()                                                   # not asked any more
        assert len(queued) == 1
        w.a_ask_transcribe.setChecked(True)
        # Show in Media panel: selects the clip's footage, clearing a bin filter that hides it
        p.clips = [TimelineClip(g.id, 0.0, 3.0)]
        f.bin = "Day 1"
        w.refresh_all(captions=False)
        w.bin.show_bin("Day 1")
        assert w.bin.tree.topLevelItemCount() == 1
        w._timeline_action("show_media", p.clips[0].id)
        assert w.bin.selected_ids() == [g.id] and w.bin.tree.topLevelItemCount() == 3
    finally:
        w.project.dirty = False
        w.close()


def test_beat_snap_never_cuts_off_the_last_words():
    f = _talk()
    c = TimelineClip(f.id, 0.8, 5.62)                  # ends just after "pantai." (5.5)
    nxt = TimelineClip(f.id, 30.0, 33.0)
    # a beat 0.4 s early would cut "pantai." off
    assert polish.snap_to_beats([c, nxt], {f.id: f}, np.array([0.0, 4.42, 9.0])) == 0
    assert c.src_out == 5.62
