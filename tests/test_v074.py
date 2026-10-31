"""0.7.4: text opacity, video inside text, the export's overwrite warning, quick zoom effects, face filters, black
frames and the non-magnetic timeline."""

import os
import subprocess

import numpy as np
import pytest

from kangedit import facefx, gaps, images, motion, titles
from kangedit.export import ExportSettings, build_export, existing_outputs, export, face_fx_clips
from kangedit.ffmpeg import FFmpegError, ffmpeg_bin, probe, run
from kangedit.models import Footage, Project, TextClip, TimelineClip


def _video(tmp_path, name="src.mp4", src="testsrc2", size="640x360", dur=3, sound=False):
    path = str(tmp_path / name)
    args = [ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i",
            f"{src}{':' if '=' in src else '='}size={size}:rate=30:duration={dur}"]
    if sound:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}", "-shortest"]
    try:
        run(args + ["-pix_fmt", "yuv420p", "-y", path])
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"FFmpeg can't make the test clip: {e}")
    i = probe(path)
    return Footage(path, i.duration, i.width, i.height, i.fps, i.has_audio)


def _frames(path, w=320, h=180):
    raw = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def _export(p, out, res="720p"):
    try:
        export(p, ExportSettings(out, p.settings.aspect, res, 30.0, use_gpu=False))
    except FFmpegError as e:
        pytest.skip(f"export needs a newer FFmpeg: {e}")


# ----------------------------------------------------------------------------- text opacity
def test_text_opacity_reaches_the_preview_state_and_the_export(tmp_path):
    tc = TextClip("Hello", 0.0, 3.0, opacity=40.0, anim_in="none", anim_out="none")
    assert titles.text_state(tc, 1.0).alpha == pytest.approx(0.4)
    tc.anim_in = "fade"
    assert titles.text_state(tc, tc.anim_dur / 2).alpha == pytest.approx(0.2)    # fades to its opacity
    tc.anim_in = "none"
    path = str(tmp_path / "t.ass")
    titles.write_titles_ass(path, [tc], 1920, 1080, 30.0)
    held = [line for line in open(path, encoding="utf-8") if line.startswith("Dialogue")]
    assert held and all("\\alpha&H99&" in line for line in held)                # 60 % transparent


def test_old_projects_keep_solid_colour_text():
    tc = TextClip.from_dict({"text": "Hi", "start": 0.0})
    assert tc.opacity == 100.0 and tc.fill == "color" and not titles.is_video_text(tc)


# ----------------------------------------------------------------------------- video inside text
def test_video_text_writes_a_mask_and_a_cover(tmp_path):
    tc = TextClip("WOW", 0.0, 2.0, fill="video", bg_color="#102030", bg_opacity=80.0, anim_in="none",
                  anim_out="none")
    m, c = str(tmp_path / "m.ass"), str(tmp_path / "c.ass")
    titles.write_titles_ass(m, [tc], 1280, 720, 30.0, mode="mask")
    titles.write_titles_ass(c, [tc], 1280, 720, 30.0, mode="cover")
    mask = open(m, encoding="utf-8").read()
    cover = open(c, encoding="utf-8").read()
    assert "m 0 0 l 1280 0 1280 720 0 720" in mask and "\\1c&HFFFFFF&\\alpha&H33&" in mask   # 80 % cover
    assert "WOW" in mask and "Dialogue: 1," in mask                     # the letters cut out, on top
    assert "\\1c&H302010&" in cover and "WOW" not in cover              # BGR order, the colour only


def test_video_text_export_graph(tmp_path):
    f = Footage(str(tmp_path / "x.mp4"), 10.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 3.0)]
    p.texts = [TextClip("BIG", 0.5, 2.0, fill="video"), TextClip("plain", 0.0, 1.0)]
    build_export(p, ExportSettings(str(tmp_path / "o.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    graph = (tmp_path / "graph.txt").read_text(encoding="utf-8")
    assert "subtitles=vtext_mask.ass" in graph and "alphamerge" in graph and "subtitles=titles.ass" in graph
    assert graph.index("vtext") < graph.index("titles.ass")              # plain titles on top of it
    assert "BIG" not in (tmp_path / "titles.ass").read_text(encoding="utf-8")


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_video_text_shows_the_video_only_in_the_letters(tmp_path):
    f = _video(tmp_path, src="color=c=0xff2020", dur=2)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 2.0)]
    p.texts = [TextClip("HOLA", 0.0, 2.0, size_pct=40.0, fill="video", anim_in="none", anim_out="none")]
    out = str(tmp_path / "out.mp4")
    _export(p, out)
    fr = _frames(out)[30].astype(int)
    red = (fr[..., 0] > 150) & (fr[..., 1] < 100)
    dark = fr.max(axis=2) < 40
    assert dark[:15].all() and dark[-15:].all()                  # around the letters: the black background
    assert 0.03 < red.mean() < 0.6                               # in the letters: the (red) video
    assert dark.mean() > 0.3


def test_video_text_preview(qapp):
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from kangedit.ui.text_render import draw_text
    img = QImage(320, 180, QImage.Format_RGB32)
    img.fill(QColor("#ff0000"))
    p = QPainter(img)
    tc = TextClip("HOLA", 0.0, 2.0, size_pct=40.0, fill="video", anim_in="none", anim_out="none")
    draw_text(p, QRectF(0, 0, 320, 180), tc, titles.text_state(tc, 1.0))
    p.end()
    assert QColor(img.pixel(2, 2)).red() < 10                    # covered
    reds = sum(QColor(img.pixel(x, 90)).red() > 200 for x in range(320))
    assert 10 < reds < 300                                       # the letters still show the picture


# ----------------------------------------------------------------------------- overwrite warning
def test_existing_outputs_lists_the_files_to_be_replaced(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "a_9x16.mp4"
    a.write_bytes(b"x")
    batch = [ExportSettings(str(a), "16:9 (Landscape)", "1080p", 30.0),
             ExportSettings(str(b), "9:16 (Vertical)", "1080p", 30.0)]
    assert existing_outputs(batch) == [os.path.abspath(a)]
    b.write_bytes(b"x")
    assert existing_outputs(batch) == [os.path.abspath(a), os.path.abspath(b)]
    assert existing_outputs([ExportSettings(str(tmp_path / "new.mp4"), "16:9 (Landscape)", "1080p", 30.0)]) == []


def test_export_dialog_asks_before_replacing(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from kangedit.ui.dialogs import ExportDialog
    p = Project()
    dlg = ExportDialog(p, False)
    for cb in dlg.chk_batch.values():
        cb.setChecked(False)
    target = tmp_path / "exists.mp4"
    dlg.ed_out.setText(str(target))
    asked = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: asked.append(self.text()) or 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)
    dlg._accept()
    assert not asked and dlg.result() == ExportDialog.Accepted     # nothing there: no question
    target.write_bytes(b"x")
    dlg.setResult(0)
    assert dlg.existing_files() == [os.path.abspath(target)]
    dlg._accept()
    assert asked and "exists.mp4" in asked[0]
    assert dlg.result() != ExportDialog.Accepted                   # "Choose another name": stays open


# ----------------------------------------------------------------------------- quick zoom effects
def test_fast_zoom_effects():
    assert set(motion.FAST_KINDS) <= set(motion.MOTIONS)
    a = motion.FAST_AMOUNT
    # punch in: zoomed in within a quarter second, and stays
    assert motion.dynamic("punch_in", 0.1, "both", 0.0, dur=4.0)[2] == pytest.approx(1.0)
    assert motion.dynamic("punch_in", 0.1, "both", 0.25 / 4, dur=4.0)[2] == pytest.approx(1 + a)
    assert motion.dynamic("punch_in", 0.1, "both", 1.0, dur=4.0)[2] == pytest.approx(1 + a)
    # punch out: the other way
    assert motion.dynamic("punch_out", 0.5, "both", 0.0, dur=4.0)[2] == pytest.approx(1.5)
    assert motion.dynamic("punch_out", 0.5, "both", 0.5, dur=4.0)[2] == pytest.approx(1.0)
    # a pulse goes in and comes back
    zs = [motion.dynamic("pulse", 0.4, "both", t / 4.0, dur=4.0)[2] for t in (0.0, 0.2, 1.0)]
    assert zs[0] == pytest.approx(1.0) and zs[1] > 1.3 and zs[2] == pytest.approx(1.0)
    # a crash zoom only at the end
    assert motion.dynamic("crash", 0.4, "both", 0.5, dur=4.0)[2] == pytest.approx(1.0)
    assert motion.dynamic("crash", 0.4, "both", 1.0, dur=4.0)[2] == pytest.approx(1.4)
    # heartbeat repeats every second
    hb = [motion.dynamic("heartbeat", 0.3, "both", t / 4.0, dur=4.0)[2] for t in (0.08, 1.08, 0.6)]
    assert hb[0] == pytest.approx(hb[1]) and hb[0] > 1.1 and hb[2] == pytest.approx(1.0)
    # shake moves the window around but keeps it inside the picture
    pts = [motion.dynamic("shake", 0.3, "both", t / 2.0, dur=2.0) for t in np.linspace(0, 2, 40)]
    xs = [x for x, _, _ in pts]
    assert max(xs) - min(xs) > 0.02
    for x, y, z in pts:
        half = 0.5 / z
        assert half - 1e-9 <= x <= 1 - half + 1e-9 and half - 1e-9 <= y <= 1 - half + 1e-9


def test_fast_effect_exports_by_command(tmp_path):
    f = Footage(str(tmp_path / "x.mp4"), 10.0, 1920, 1080, 30.0, False)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 2.0, motion="punch_in", motion_amount=0.3)]
    build_export(p, ExportSettings(str(tmp_path / "o.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    cmds = (tmp_path / "motion_0.txt").read_text().splitlines()
    widths = [int(c.split("scale@c0s w ")[1].split(",")[0]) for c in cmds]
    assert widths[0] < widths[10] == widths[-1]                  # zoomed in by a third of a second, then held


def test_inspector_fast_effect_sets_a_visible_amount(qapp):
    from kangedit.ui.inspector import Inspector
    f = Footage("x.mp4", 10.0, 1920, 1080, 30.0, False)
    p = Project()
    p.footage = {f.id: f}
    c = TimelineClip(f.id, 0.0, 2.0)
    p.clips = [c]
    ins = Inspector(p)
    ins.set_clip(c)
    ins.cb_motion.setCurrentIndex(ins.cb_motion.findData("pulse"))
    assert c.motion == "pulse" and c.motion_amount == pytest.approx(motion.FAST_AMOUNT)
    assert not ins.cb_ease.isEnabled()
    ins.cb_face.setCurrentIndex(ins.cb_face.findData("big_head"))
    assert c.face_fx == "big_head"


# ----------------------------------------------------------------------------- face filters
def _dot_image(w=320, h=240):
    img = np.full((h, w, 3), 30, np.uint8)
    import cv2
    cv2.circle(img, (w // 2, h // 2), 10, (255, 255, 255), -1)
    return img


FACE = facefx.Face(0.5, 0.5 + 0.12 * 0.25, 0.25, 0.25, ((0.45, 0.47), (0.55, 0.47)), (0.5, 0.58))


@pytest.mark.parametrize("kind", [k for k in facefx.FACE_FX if k])
def test_every_face_filter_changes_only_around_the_face(kind):
    img = (np.random.default_rng(1).random((240, 320, 3)) * 255).astype(np.uint8)
    out = facefx.apply(img, kind, FACE, 0.8)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert not np.array_equal(out, img)
    assert np.array_equal(out[:, :40], img[:, :40]) and np.array_equal(out[:, -40:], img[:, -40:])
    assert np.array_equal(facefx.apply(img, kind, None, 0.8), img)          # no face: nothing changes


def test_big_head_grows_and_tiny_head_shrinks():
    img = _dot_image()
    face = facefx.Face(0.5, 0.5 + 0.12 * 0.3, 0.3, 0.3)          # the head's middle on the dot
    big = (facefx.apply(img, "big_head", face, 1.0)[..., 0] > 128).sum()
    tiny = (facefx.apply(img, "tiny_head", face, 1.0)[..., 0] > 128).sum()
    base = (img[..., 0] > 128).sum()
    assert big > 1.8 * base and tiny < 0.7 * base
    # 16-bit pictures (the export's) work the same
    assert facefx.apply(img.astype(np.uint16) * 257, "big_head", face, 1.0).dtype == np.uint16


def test_tracker_follows_and_lets_go(monkeypatch):
    seq = iter([FACE, None, None, None, None, None, None, None, None, None, None, None])
    monkeypatch.setattr(facefx, "find", lambda img: next(seq))
    tr = facefx.Tracker(fps=10.0, every=1)
    img = np.zeros((10, 10, 3), np.uint8)
    face, k = tr.step(img)
    assert face == FACE and k == 1.0
    ks = [tr.step(img)[1] for _ in range(10)]
    assert 0 < ks[0] < 1.0 and ks == sorted(ks, reverse=True) and ks[-1] == 0.0    # fades, then dropped


def test_preview_applies_the_face_filter(monkeypatch):
    calls = []
    monkeypatch.setattr(facefx, "find", lambda img: calls.append(1) or FACE)
    img = (np.random.default_rng(2).random((240, 320, 3)) * 255).astype(np.uint8)
    a = facefx.apply_frame(img, "big_eyes", 0.8, "fid-x", 1.234)
    b = facefx.apply_frame(img, "big_eyes", 0.8, "fid-x", 1.234)
    assert not np.array_equal(a, img) and np.array_equal(a, b) and len(calls) == 1   # found once, cached
    assert facefx.apply_frame(img, "", 0.8, "fid-x", 1.0) is img


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_face_filter_export_matches_the_preview_warp(tmp_path, monkeypatch):
    monkeypatch.setattr(facefx, "find", lambda img: FACE)
    f = _video(tmp_path, dur=2, sound=True)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.5, 1.5, face_fx="big_head", face_amount=0.9), TimelineClip(f.id, 1.5, 2.0)]
    assert face_fx_clips(p) == [0]
    out = str(tmp_path / "out.mp4")
    _export(p, out, "720p")
    src, got = _frames(f.path, 640, 360), _frames(out, 640, 360)
    assert len(got) == 45
    want = facefx.apply(src[30], "big_head", FACE, 0.9)       # clip frame 15 = source 1.0 s
    plain = src[30]
    face = (slice(100, 260), slice(200, 440))                 # around the face: where the warp is

    def diff(a, b):
        return np.abs(a[face].astype(float) - b[face].astype(float)).mean()
    assert diff(got[15], want) < 8.0 < diff(got[15], plain)
    assert np.abs(got[40].astype(float) - src[55].astype(float)).mean() < 8.0    # the next clip as it is
    pcm = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", out, "-t", "0.9", "-f", "f32le", "-ac", "1", "-"],
                         capture_output=True).stdout
    assert np.sqrt(np.mean(np.frombuffer(pcm, np.float32) ** 2)) > 0.05            # its sound kept


# ----------------------------------------------------------------------------- black frames, the magnet
def _gap_project():
    f = Footage("a.mp4", 60.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 2.0, id="A"), TimelineClip(f.id, 10.0, 13.0, id="B"),
               TimelineClip(f.id, 20.0, 21.0, id="C")]
    return p


def _layout(p):
    return [("_" if gaps.is_black(p, c) else c.id, round(c.duration, 3)) for c in p.clips]


def test_insert_black_frame_and_seconds():
    p = _gap_project()
    c = gaps.insert_black(p, 2.1, 0.0)
    assert _layout(p) == [("A", 2.0), ("_", round(1 / 30, 3)), ("B", 3.0), ("C", 1.0)]
    assert c.duration == pytest.approx(1 / 30, abs=1e-4)
    f = p.footage[c.footage_id]
    assert f.generated == "black" and f.is_image and os.path.exists(f.path)
    assert images.load(f.path)[0][0][..., :3].max() == 0
    gaps.insert_black(p, 0.0, 1.0)
    assert _layout(p)[0] == ("_", 1.0)
    assert len([x for x in p.footage.values() if x.generated]) == 1      # one black picture for all of them
    assert not p.conflicts()                                             # black repeats nothing


def test_moving_with_the_magnet_off_leaves_a_gap():
    p = _gap_project()
    gaps.place(p, "A", 3.0)                  # A over B: in at the nearest cut, its place left black
    assert _layout(p) == [("_", 2.0), ("A", 2.0), ("B", 3.0), ("C", 1.0)]
    p = _gap_project()
    gaps.place(p, "C", 9.0)                  # past the end: black fills the space before it
    assert _layout(p) == [("A", 2.0), ("B", 3.0), ("_", 4.0), ("C", 1.0)]
    gaps.place(p, "A", 6.0)                  # into the gap: takes part of it, the rest stays
    assert _layout(p) == [("_", 2.0), ("B", 3.0), ("_", 1.0), ("A", 2.0), ("_", 1.0), ("C", 1.0)]
    assert p.duration == pytest.approx(10.0)
    gaps.place(p, "C", 0.0)                  # into the black at the start; the end's black goes
    assert _layout(p) == [("C", 1.0), ("_", 1.0), ("B", 3.0), ("_", 1.0), ("A", 2.0)]


def test_lift_and_tidy():
    p = _gap_project()
    assert gaps.lift(p, ["B"]) == 1
    assert _layout(p) == [("A", 2.0), ("_", 3.0), ("C", 1.0)]
    gaps.lift(p, ["C"])                      # black at the very end is dropped
    assert _layout(p) == [("A", 2.0)]


def test_black_clips_export_and_survive_save(tmp_path):
    p = _gap_project()
    gaps.insert_black(p, 2.0, 1.0)
    path = str(tmp_path / "p.kgproj")
    p.save(path)
    q = Project.load(path)
    bf = next(f for f in q.footage.values() if f.generated)
    bf.path = str(tmp_path / "gone.png")
    q.save(path)
    q = Project.load(path)                   # opened on another computer: its own black picture
    assert os.path.exists(next(f for f in q.footage.values() if f.generated).path)
    args = build_export(q, ExportSettings(str(tmp_path / "o.mp4"), q.settings.aspect, "1080p", 30.0,
                                          use_gpu=False), str(tmp_path))
    assert "-loop" in args                   # the black picture, held as long as the clip


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_black_gap_exports_black(tmp_path):
    f = _video(tmp_path, src="color=c=0xffffff", dur=2)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 1.0), TimelineClip(f.id, 1.0, 2.0)]
    gaps.insert_black(p, 1.0, 1.0)
    out = str(tmp_path / "o.mp4")
    _export(p, out, "720p")
    fr = _frames(out)
    assert len(fr) == 90
    assert fr[15].mean() > 200 and fr[45].mean() < 20 and fr[75].mean() > 200


@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.project.dirty = False
    w.close()


def test_window_magnet_delete_and_black(window):
    p = _gap_project()
    window.set_project(p)
    window.timeline.set_magnetic(False)
    assert not window.tl.magnetic and not window.a_magnet.isChecked()
    assert window.qs.value("magnetic", True, type=bool) is False
    window.tl.selected = "B"
    window.clip_action("delete", "B")
    assert _layout(window.project) == [("A", 2.0), ("_", 3.0), ("C", 1.0)]
    window.undo()
    assert _layout(window.project) == [("A", 2.0), ("B", 3.0), ("C", 1.0)]
    window.timeline.set_magnetic(True)
    window.clip_action("delete", "B")
    assert _layout(window.project) == [("A", 2.0), ("C", 1.0)]
    window._timeline_action("insert_black", "0.5@2.0")
    assert _layout(window.project) == [("A", 2.0), ("_", 0.5), ("C", 1.0)]
    from PySide6.QtCore import Qt
    assert not window.bin.tree.findItems("black", Qt.MatchContains | Qt.MatchRecursive)   # not shown as media
    window.qs.setValue("magnetic", True)


def test_timeline_free_drag_places_the_clip(window):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    p = _gap_project()
    window.set_project(p)
    window.timeline.set_magnetic(False)
    tl = window.tl
    tl.resize(1200, 400)
    tl.scroll_x = 0.0
    tl.set_zoom(50.0)
    r = tl._clip_rects()[2][1]                     # C, 5..6 s
    start = r.center()

    def ev(kind, pt, buttons):
        return QMouseEvent(kind, pt, QPointF(tl.mapToGlobal(pt.toPoint())), Qt.LeftButton, buttons, Qt.NoModifier)

    tl.mousePressEvent(ev(QEvent.MouseButtonPress, start, Qt.LeftButton))
    end = QPointF(start.x() + 3.0 * tl.pps, start.y())          # 3 s later: past the end of the edit
    for k in range(1, 11):
        tl.mouseMoveEvent(ev(QEvent.MouseMove, QPointF(start.x() + (end.x() - start.x()) * k / 10, start.y()),
                             Qt.LeftButton))
    assert tl._drag["free"] == pytest.approx(8.0, abs=0.05)
    tl.mouseReleaseEvent(ev(QEvent.MouseButtonRelease, end, Qt.NoButton))
    lay = _layout(window.project)
    assert [x for x, _ in lay] == ["A", "B", "_", "C"] and lay[2][1] == pytest.approx(3.0, abs=0.05)
    window.undo()
    assert _layout(window.project) == [("A", 2.0), ("B", 3.0), ("C", 1.0)]
    window.timeline.set_magnetic(True)
    window.qs.setValue("magnetic", True)


def test_face_filter_reaches_the_viewer_and_finds_no_face_in_noise(window):
    f = Footage("a.mp4", 60.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 2.0, face_fx="big_eyes", face_amount=0.5)]
    window.set_project(p)
    shot = window.viewer._shots()[0]
    assert (shot.face_fx, shot.face_amount) == ("big_eyes", 0.5)
    noise = (np.random.default_rng(3).random((360, 640, 3)) * 255).astype(np.uint8)
    assert facefx.find(noise) is None
