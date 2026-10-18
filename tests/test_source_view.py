"""The viewer's Source mode: a file from the Media panel played on its own, marked by the playhead or by its
words, and the marked part put on the timeline."""

import numpy as np
import pytest

from kangedit.ffmpeg import ffmpeg_bin, probe, run
from kangedit.models import Footage, Project, TimelineClip


@pytest.fixture
def window(qapp, tmp_path):
    from kangedit.ui.main_window import MainWindow
    if not ffmpeg_bin():
        pytest.skip("needs FFmpeg")
    src = str(tmp_path / "talk.mp4")
    run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=12",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=12", "-shortest", "-pix_fmt", "yuv420p", "-y", src])
    i = probe(src)
    talk = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    talk.transcript = [{"s": 1.0, "e": 1.4, "w": "Halo"}, {"s": 1.5, "e": 2.0, "w": "semua."},
                       {"s": 5.0, "e": 5.3, "w": "Lihat"}, {"s": 5.4, "e": 5.8, "w": "ombaknya"},
                       {"s": 5.9, "e": 6.5, "w": "besar!"}]
    other = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    w = MainWindow()
    p = Project()
    p.footage = {talk.id: talk, other.id: other}
    p.clips = [TimelineClip(other.id, 0.0, 3.0), TimelineClip(other.id, 6.0, 9.0)]
    w.set_project(p)
    yield w, talk
    w.project.dirty = False
    w.close()


def test_double_click_plays_the_file_on_its_own(window):
    w, talk = window
    v = w.viewer
    moved = []
    v.time_changed.connect(moved.append)
    w.bin_action("play", [talk.id])
    assert v.mode == "source" and v.source_fid == talk.id and not v.source_panel.isHidden()
    assert v.project is not w.project and v.project.duration == pytest.approx(talk.duration, abs=0.05)
    assert v.btn_source.isChecked() and talk.name in v.btn_source.toolTip()
    v.seek(4.0)
    assert moved == [] and v.t == pytest.approx(4.0)            # the timeline's playhead stays
    assert v.timeline_time() == 0.0
    assert v.audio_valid and v._audio_path == talk.path          # the file's own sound
    v.set_audio("/x/mix.wav")                                    # a new timeline mix meanwhile: kept for later
    assert v._audio_path == talk.path
    w._timeline_seek(2.0)                                        # the timeline clicked: back to the edit
    assert v.mode == "timeline" and v.project is w.project and v.t == pytest.approx(2.0)
    assert v._audio_path == "/x/mix.wav" and v.source_panel.isHidden()


def test_marks_by_playhead_and_by_words(window):
    w, talk = window
    v = w.viewer
    v.open_source(talk.id)
    assert v.marked_range() == pytest.approx((0.0, talk.duration), abs=0.05)   # nothing marked: all
    v.seek(2.0)
    w._source_mark("in")
    v.seek(4.0)
    w._source_mark("out")
    a, b = v.marked_range()
    assert a == pytest.approx(2.0) and 4.0 < b < 4.1
    assert "In 00:02:00" in v.source_panel.lbl.text()
    words = v.source_panel.words
    assert words.toPlainText() == "Halo semua. Lihat ombaknya besar!" and words.isVisibleTo(v)
    # select "Lihat ombaknya besar!": the marks follow the words (a little room around them)
    from PySide6.QtGui import QTextCursor
    cur = words.textCursor()
    start = words.toPlainText().index("Lihat")
    cur.setPosition(start + 2)
    cur.setPosition(len(words.toPlainText()) - 3, QTextCursor.KeepAnchor)
    words.setTextCursor(cur)
    a, b = v.marked_range()
    assert a == pytest.approx(4.95) and b == pytest.approx(6.6)
    assert v.t == pytest.approx(4.95)
    w._source_mark("clear")
    assert v.marked_range()[0] == 0.0
    w.viewer.show_timeline()
    w._source_mark("in")                                         # not in Source mode: nothing marked
    assert "double-click" in w.lbl_status.text()


def test_marked_part_goes_on_the_timeline(window):
    w, talk = window
    v = w.viewer
    v.seek(3.2)                                                  # the timeline's playhead near the cut at 3 s
    v.open_source(talk.id)
    v.set_marks(5.0, 6.5)
    v.source_panel.use.emit("insert")
    clips = w.project.clips
    assert len(clips) == 3 and clips[1].footage_id == talk.id
    assert (clips[1].src_in, clips[1].src_out) == pytest.approx((4.95, 6.6))
    assert v.mode == "source"                                    # still on the file, for the next part
    assert v.timeline_time() == pytest.approx(3.0 + clips[1].duration)   # the next one goes after it
    v.set_marks(1.0, 2.0)
    v.source_panel.use.emit("insert")
    assert [c.src_in for c in w.project.clips[1:3]] == pytest.approx([4.95, 0.95])
    v.source_panel.use.emit("append")
    assert w.project.clips[-1].footage_id == talk.id and len(w.project.clips) == 5
    w.undo()
    assert len(w.project.clips) == 4


def test_source_frames_are_the_files(window):
    w, talk = window
    v = w.viewer
    v.open_source(talk.id)
    got = {}
    v.engine._on_still = lambda r: got.setdefault("img", r.image)
    v.seek(7.0)
    import time
    from PySide6.QtWidgets import QApplication
    end = time.time() + 5
    while "img" not in got and time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)
    assert got.get("img") is not None and np.asarray(got["img"]).std() > 5


def test_timeline_jumps_leave_source_mode(window):
    w, talk = window
    w.bin_action("play", [talk.id])
    w.viewer.seek(4.5)
    w.caps_panel.seek.emit(2.0)                      # a caption clicked: a time on the timeline
    assert w.viewer.mode == "timeline" and w.viewer.t == pytest.approx(2.0)
    w.bin_action("play", [talk.id])
    w.edit_text.seek.emit(1.0)                       # a word in the 🗒 Transcript tab
    assert w.viewer.mode == "timeline" and w.viewer.t == pytest.approx(1.0)


def test_back_to_a_shorter_edit_keeps_the_playhead_on_it(window):
    w, talk = window
    w.bin_action("play", [talk.id])
    w.viewer.set_marks(5.0, 6.5)
    w.use_source_part(talk.id, *w.viewer.marked_range(), "append")
    assert w.viewer.timeline_time() > 6.0
    w.undo()
    w.viewer.show_timeline()
    assert w.viewer.t < w.project.duration


def test_removing_the_file_shown_goes_back_to_the_edit(window):
    w, talk = window
    w.bin_action("play", [talk.id])
    w.bin_action("remove", [talk.id])
    v = w.viewer
    assert v.mode == "timeline" and not v.source_fid and not v.btn_source.isEnabled()
    assert v.btn_timeline.isChecked() and not v.btn_source.isChecked()
