"""Thumbnails: the edit's best stills, framed on the face, with outlined text, saved as a JPG."""

import numpy as np
import pytest

from kangedit import thumbnail as th
from kangedit.ffmpeg import ffmpeg_bin, probe, run
from kangedit.models import Footage, Project, TimelineClip

from .test_edit_features import fake_analysis, two_files


def test_candidates_are_the_lively_moments_of_the_edit():
    footage, analyses = two_files()                    # a lively moment at 8 s in one file, 20 s in the other
    fa, fb = list(footage)
    p = Project()
    p.footage = footage
    p.clips = [TimelineClip(fa, 2.0, 12.0), TimelineClip(fb, 15.0, 25.0), TimelineClip(fa, 20.0, 28.0)]
    got = th.candidates(p, analyses)
    assert len(got) == 3 and {got[0].footage_id, got[1].footage_id} == {fa, fb}      # other files first
    by_clip = {s.clip_id: s for s in got}
    assert abs(by_clip[p.clips[0].id].t - 8.0) < 1.0 and abs(by_clip[p.clips[1].id].t - 20.0) < 1.0
    assert all(p.clips[p.index_of(s.clip_id)].src_in <= s.t <= p.clips[p.index_of(s.clip_id)].src_out for s in got)


def test_framed_on_the_face_and_livelier():
    img = np.zeros((360, 640, 3), np.uint8)
    img[100:140, 500:540] = 255                        # "a face" on the right
    tall = th.framed(img, 180, 320, face=(520 / 640, 120 / 360))
    assert tall.shape == (320, 180, 3)
    ys, xs = np.nonzero(tall[..., 0] > 128)
    assert abs(xs.mean() - 90) < 25                    # the face is in the middle of the vertical picture
    centred = th.framed(img, 180, 320)
    assert (centred[..., 0] > 128).sum() == 0          # framed on the middle, the face is cut off
    grey = np.full((10, 10, 3), (120, 100, 90), np.uint8)
    assert th.pop(grey).std(axis=2).mean() > grey.std(axis=2).mean()


def test_text_is_drawn_and_saved(qapp, tmp_path):
    from PySide6.QtGui import QImage
    from kangedit.ui.thumbnail_dialog import compose
    rgb = np.full((720, 1280, 3), 60, np.uint8)
    plain = compose(rgb, "")
    img = compose(rgb, "BEST DAY\nEVER", "#FFD400", "bottom", 14.0)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(720, img.bytesPerLine() // 4, 4)[:, :1280, :3]
    yellow = (a[..., 2] > 200) & (a[..., 1] > 180) & (a[..., 0] < 80)      # BGRA: yellow letters
    ys = np.nonzero(yellow)[0]
    assert yellow.sum() > 5000 and ys.min() > 360                           # at the bottom
    assert (a[..., :3] < 20).all(axis=2).sum() > 2000                       # with a dark outline
    assert plain.size() == img.size()
    out = str(tmp_path / "t.jpg")
    assert img.save(out, None, 92) and QImage(out).width() == 1280


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_dialog_makes_a_thumbnail_from_a_real_clip(qapp, tmp_path):
    from PySide6.QtGui import QImage
    from kangedit.ui.thumbnail_dialog import ThumbnailDialog
    src = str(tmp_path / "v.mp4")
    run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4",
         "-pix_fmt", "yuv420p", "-y", src])
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.5, 3.5)]
    d = ThumbnailDialog(p, {f.id: fake_analysis(4.0, 2.0, 1)}, th.Still(f.id, 1.0, 0.0, p.clips[0].id))
    assert d.list.count() == 2 and d.image is not None                       # the playhead's frame, picked
    d.text.setPlainText("WOW")
    d.cb_size.setCurrentIndex(d.cb_size.findData("9:16"))
    out = str(tmp_path / "thumb.jpg")
    assert d.save(out)
    img = QImage(out)
    assert (img.width(), img.height()) == (1080, 1920)
