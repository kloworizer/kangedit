"""Dynamic zoom (a slow zoom / pan inside the picture) and stabilization: the window each frame shows, the same
in the preview and the export, and a shaky camera held steady."""

import subprocess

import av
import cv2
import numpy as np
import pytest

from kangedit import motion, reframe, stabilize
from kangedit.export import ExportSettings, build_export, export
from kangedit.ffmpeg import FFmpegError, ffmpeg_bin, probe, run
from kangedit.models import Footage, Project, TimelineClip


def test_dynamic_zoom_presets():
    assert motion.dynamic("", 0.2, "both", 0.5) == (0.5, 0.5, 1.0)
    x, y, z = motion.dynamic("in", 0.2, "linear", 1.0)
    assert (x, y) == (0.5, 0.5) and abs(z - 1.2) < 1e-9
    assert motion.dynamic("out", 0.2, "linear", 0.0)[2] == pytest.approx(1.2)
    assert motion.dynamic("out", 0.2, "linear", 1.0)[2] == pytest.approx(1.0)
    # a pan keeps its zoom and moves from one edge to the other (as far as the zoom leaves room)
    a, b = motion.dynamic("left", 0.2, "linear", 0.0), motion.dynamic("left", 0.2, "linear", 1.0)
    assert a[2] == b[2] == pytest.approx(1.2) and a[0] > 0.5 > b[0]
    assert b[0] == pytest.approx(0.5 / 1.2)                 # the window's left edge on the picture's
    assert motion.dynamic("up", 0.2, "linear", 0.0)[1] > motion.dynamic("up", 0.2, "linear", 1.0)[1]
    # zoom out and pan left: the view ends wide and moves left on the way
    seq = [motion.dynamic("out_left", 0.2, "linear", u) for u in (0.0, 0.5, 1.0)]
    assert seq[0][0] > seq[1][0] > seq[2][0] - 1e-9 and seq[-1][2] == pytest.approx(1.0)
    # easing starts and ends gently, linear doesn't
    assert motion.ease(0.1, "both") < 0.1 < motion.ease(0.9, "both") and motion.ease(0.1, "linear") == 0.1


def test_preview_window_is_the_zoomed_part_of_the_picture():
    img = (np.random.default_rng(0).random((360, 640, 3)) * 255).astype(np.uint8)
    # zoom 2 on the centre: the middle half of the picture, enlarged
    got = reframe.render(img, 320, 180, "fill", window=(0.5, 0.5, 2.0, 0.0))
    want = reframe.render(img[90:270, 160:480], 320, 180, "fill")
    assert np.abs(got.astype(int) - want.astype(int)).mean() < 1.0
    # no movement: exactly the still framing
    assert np.array_equal(reframe.render(img, 320, 180, "fill", window=(0.5, 0.5, 1.0, 0.0)),
                          reframe.render(img, 320, 180, "fill"))


def _clip_file(tmp_path, size="640x360", dur=3):
    src = str(tmp_path / "src.mp4")
    try:
        run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={dur}",
             "-pix_fmt", "yuv420p", "-y", src])
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"FFmpeg can't make the test clip: {e}")
    i = probe(src)
    return Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)


def _frames(path, w=320, h=180):
    raw = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def test_export_drives_the_window_by_command(tmp_path):
    f = Footage(str(tmp_path / "x.mp4"), 10.0, 1920, 1080, 30.0, False)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 2.0, motion="in", motion_amount=0.2), TimelineClip(f.id, 3.0, 4.0)]
    args = build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0,
                                          use_gpu=False), str(tmp_path))
    graph = open(tmp_path / args[args.index("-/filter_complex") + 1]).read()
    assert "sendcmd=f=motion_0.txt" in graph and "scale@c0s" in graph and "crop@c0c" in graph
    assert "motion_1.txt" not in graph                       # a clip without movement: the plain framing
    cmds = open(tmp_path / "motion_0.txt").read().splitlines()
    assert len(cmds) == 60 and "rotate@" not in cmds[0]
    first, last = (int(c.split("scale@c0s w ")[1].split(",")[0]) for c in (cmds[0], cmds[-1]))
    assert first < last                                      # zooms in over the clip


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
@pytest.mark.parametrize("kind", ["in_right", "up"])
def test_exported_dynamic_zoom_matches_the_preview(tmp_path, kind):
    f = _clip_file(tmp_path)
    p = Project()
    p.footage = {f.id: f}
    c = TimelineClip(f.id, 0.0, 2.0, motion=kind, motion_amount=0.5, motion_ease="linear")
    p.clips = [c]
    out = str(tmp_path / "out.mp4")
    try:
        export(p, ExportSettings(out, p.settings.aspect, "720p", 30.0, use_gpu=False))
    except FFmpegError as e:
        pytest.skip(f"export needs a newer FFmpeg: {e}")
    s, o = _frames(f.path, 640, 360), _frames(out)
    m = motion.of_clip(c, f, 16 / 9)
    pl = reframe.place(640, 360, 320, 180, "fill")
    for n in (15, 30, 59):
        win = m.window(n / 30, c.src_at(n / 30), pl.scaled_w / pl.crop_w, pl.scaled_h / pl.crop_h)
        moved = reframe.render(s[n], 320, 180, "fill", window=win).astype(float)
        still = reframe.render(s[n], 320, 180, "fill").astype(float)
        assert np.abs(o[n] - moved).mean() < 8.0 < np.abs(o[n] - still).mean()


def _shaky(path, n=90, seed=1):
    """A slow pan over a texture with a shaking, turning camera."""
    rng = np.random.default_rng(seed)
    big = cv2.resize((rng.random((90, 160, 3)) * 255).astype(np.uint8), (1280, 720), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), 2)
    jx, jy, ja = rng.normal(0, 6, n), rng.normal(0, 6, n), rng.normal(0, 0.8, n)
    pan = np.linspace(0, 120, n)
    out = av.open(path, "w")
    st = out.add_stream("libx264", rate=30)
    st.width, st.height, st.pix_fmt = 640, 360, "yuv420p"
    st.options = {"crf": "12"}
    for k in range(n):
        cx, cy = 480 + pan[k] + jx[k], 360 + jy[k]
        m = cv2.getRotationMatrix2D((cx, cy), ja[k], 1.0)
        m[:, 2] += [320 - cx, 180 - cy]
        for pk in st.encode(av.VideoFrame.from_ndarray(cv2.warpAffine(big, m, (640, 360)), format="rgb24")):
            out.mux(pk)
    for pk in st.encode():
        out.mux(pk)
    out.close()


def _jitter(seq) -> float:
    """How much the picture shakes: frame-to-frame movement less its slow part (pixels)."""
    g = [cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2GRAY) for x in seq]
    mv = np.array([stabilize.frame_motion(a, b)[:2] for a, b in zip(g, g[1:])])
    slow = np.array([np.convolve(mv[:, j], np.ones(9) / 9, mode="same") for j in (0, 1)]).T
    return float(np.sqrt(((mv - slow)[4:-4] ** 2).mean()))


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_stabilize_holds_a_shaky_camera_steady(tmp_path, monkeypatch):
    monkeypatch.setattr(stabilize, "cache_dir", lambda sub: tmp_path)
    monkeypatch.setattr(stabilize, "_tracks", {})
    monkeypatch.setattr(stabilize, "_corrections", {})
    src = str(tmp_path / "shaky.mp4")
    _shaky(src)
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    c = TimelineClip(f.id, 0.5, 2.5, stabilize=0.6)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [c]
    assert stabilize.missing(p) == [(f, 0.5 - stabilize.HANDLE, 2.5 + stabilize.HANDLE)]
    assert motion.of_clip(c, f) is None                     # not measured yet: shown as it is
    stabilize.analyze(f, 0.0, f.duration, use_proxy=False)
    assert stabilize.missing(p) == [] and stabilize.load(f).covers(0.0, f.duration)
    m = motion.of_clip(c, f, 16 / 9)
    assert m is not None and 1.02 < m.stab.zoom < 1.5
    s = _frames(src, 640, 360)
    pl = reframe.place(640, 360, 320, 180, "fill")
    orig, steady = [], []
    for k in range(60):
        off = k / 30
        n = int(round(c.src_at(off) * 30))
        win = m.window(off, c.src_at(off), pl.scaled_w / pl.crop_w, pl.scaled_h / pl.crop_h)
        orig.append(reframe.render(s[n], 320, 180, "fill"))
        steady.append(reframe.render(s[n], 320, 180, "fill", window=win))
    assert _jitter(steady) < 0.25 * _jitter(orig)
    out = str(tmp_path / "out.mp4")
    try:
        export(p, ExportSettings(out, p.settings.aspect, "720p", 30.0, use_gpu=False))
    except FFmpegError as e:
        pytest.skip(f"export needs a newer FFmpeg: {e}")
    e = _frames(out)
    assert len(e) == 60 and _jitter(list(e)) < 0.25 * _jitter(orig)
    assert np.mean([np.abs(e[k].astype(float) - steady[k]).mean() for k in range(5, 55)]) < 12.0
    full = _frames(out, 1280, 720)                       # turned, its corners are picture (no black wedges)
    corners = [fr[a:a + 40, b:b + 40] for fr in full[5:55] for a in (0, 680) for b in (0, 1240)]
    assert max(int((c.max(axis=2) < 16).sum()) for c in corners) < 20


def test_zoom_on_the_face_goes_to_the_face():
    x, y, z = motion.dynamic("face", 0.5, "linear", 1.0, target=(0.6, 0.2))
    assert z == pytest.approx(1.5) and x == pytest.approx(0.6) and y == pytest.approx(1 / 3)   # (kept inside)
    assert motion.dynamic("face", 0.5, "linear", 0.0, target=(0.6, 0.2)) == (0.5, 0.5, 1.0)
    assert motion.dynamic("face_out", 0.5, "linear", 0.0, target=(0.6, 0.5))[:2] == pytest.approx((0.6, 0.5))
    f = Footage("x.mp4", 10.0, 1920, 1080, 30.0, False)
    c = TimelineClip(f.id, 0.0, 4.0, motion="face", motion_amount=0.5, motion_face=[0.6, 0.5])
    m = motion.of_clip(c, f)
    # framed 16:9 into 16:9: the face's place in the picture is where it is in the source
    p = reframe.place(1920, 1080, 1280, 720, "fill")
    assert m.at_placement(4.0, 4.0, p)[0] == pytest.approx(0.6, abs=1e-3)
    # framed into 9:16 (a crop of the middle): the face right of the middle sits at the crop's right
    p = reframe.place(1920, 1080, 720, 1280, "fill")
    cx = m.at_placement(4.0, 4.0, p)[0]
    want = (0.6 * p.scaled_w - p.crop_x) / p.crop_w
    assert want > 0.6 and cx == pytest.approx(min(want, 1 - 0.5 / 1.5), abs=1e-3)
    # no face found: to the middle
    c.motion_face = []
    assert motion.of_clip(c, f).at_placement(4.0, 4.0, p)[:2] == pytest.approx((0.5, 0.5))


def test_layer_clips_move_inside_their_picture():
    from kangedit.models import OverlayClip
    f = Footage("x.mp4", 10.0, 1920, 1080, 30.0, False)
    ov = OverlayClip(f.id, 2.0, 0.0, 4.0, scale=0.5, motion="in", motion_amount=0.3, rotation=90.0)
    m = motion.of_clip(ov, f)
    assert m is not None and m.quarters == 0                     # a layer's rotation turns the layer, not its picture
    rgba = np.zeros((90, 160, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[40:50, 75:85, :3] = 255                                 # a white square in the middle
    still = reframe.overlay_layer(rgba, 320, 180, 0.5, 0.0)
    zoomed = reframe.overlay_layer(rgba, 320, 180, 0.5, 0.0, window=(0.5, 0.5, 2.0, 0.0))
    assert still.shape == zoomed.shape
    assert (zoomed[..., 0] > 128).sum() > 3 * (still[..., 0] > 128).sum()   # twice as big: four times the area


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_exported_layer_zoom_matches_the_preview(tmp_path):
    from kangedit.models import OverlayClip
    f = _clip_file(tmp_path)
    bg = str(tmp_path / "bg.mp4")
    run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=c=0x202020:size=640x360:rate=30:duration=3",
         "-pix_fmt", "yuv420p", "-y", bg])
    i = probe(bg)
    fb = Footage(bg, i.duration, i.width, i.height, i.fps, i.has_audio)
    p = Project()
    p.footage = {f.id: f, fb.id: fb}
    p.clips = [TimelineClip(fb.id, 0.0, 2.0)]
    ov = OverlayClip(f.id, 0.0, 0.0, 2.0, scale=0.6, motion="in_right", motion_amount=0.5, motion_ease="linear")
    p.overlays = [ov]
    out = str(tmp_path / "out.mp4")
    try:
        export(p, ExportSettings(out, p.settings.aspect, "720p", 30.0, use_gpu=False))
    except FFmpegError as e:
        pytest.skip(f"export needs a newer FFmpeg: {e}")
    s, o = _frames(f.path, 640, 360), _frames(out)
    m = motion.of_clip(ov, f)
    pl = reframe.overlay_place(640, 360, 320, 180, 0.6)
    for n in (20, 50):
        rgba = np.dstack([s[n], np.full(s[n].shape[:2], 255, np.uint8)])
        layer = reframe.overlay_layer(rgba, 320, 180, 0.6, 0.0, window=m.at_placement(n / 30, ov.src_at(n / 30), pl))
        plain = reframe.overlay_layer(rgba, 320, 180, 0.6, 0.0)
        h, w = layer.shape[:2]
        y0, x0 = (180 - h) // 2, (320 - w) // 2
        got = o[n][y0 + 2:y0 + h - 2, x0 + 2:x0 + w - 2].astype(float)
        assert np.abs(got - layer[2:-2, 2:-2, :3]).mean() < 10.0 < np.abs(got - plain[2:-2, 2:-2, :3]).mean()


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
@pytest.mark.parametrize("kw", [{"motion": "in"}, {"stabilize": 0.6}])
def test_moving_clip_keeps_its_sound_next_to_another(tmp_path, kw):
    """A dynamic zoom / stabilized clip later in its file, next to another clip: exported with its sound (its
    sound was cut from the wrong place, past the end of its input: silent, and the export failed)."""
    src = str(tmp_path / "talk.mp4")
    run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=12",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-shortest", "-pix_fmt", "yuv420p", "-y", src])
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 6.0, 8.0, **kw), TimelineClip(f.id, 9.0, 10.0)]
    out = str(tmp_path / "out.mp4")
    try:
        export(p, ExportSettings(out, p.settings.aspect, "720p", 30.0, use_gpu=False))
    except FFmpegError as e:
        pytest.skip(f"export needs a newer FFmpeg: {e}")
    pcm = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", out, "-t", "1.8", "-f", "f32le", "-ac", "1", "-"],
                         capture_output=True).stdout
    sound = np.frombuffer(pcm, np.float32)
    assert sound.size > 40000 and np.sqrt(np.mean(sound ** 2)) > 0.05           # the moving clip is heard


def test_shake_is_measured_on_every_frame_of_a_clip_later_in_its_file(tmp_path, monkeypatch):
    monkeypatch.setattr(stabilize, "cache_dir", lambda sub: tmp_path)
    monkeypatch.setattr(stabilize, "_tracks", {})
    monkeypatch.setattr(stabilize, "_corrections", {})
    src = str(tmp_path / "shaky.mp4")
    _shaky(src, n=300)                                   # 10 s, with B-frames (PyAV: no FFmpeg needed)
    f = Footage(src, 10.0, 640, 360, 30.0, False)
    tr = stabilize.analyze(f, 5.0, 8.0, use_proxy=False)
    got = int(np.sum((tr.t >= 5.0) & (tr.t <= 8.0)))
    assert got >= 85                                     # 3 s at 30 fps: every frame, not every other one
