"""Chroma key (green / blue screen on a video layer) and 360° footage (a view of the sphere): the preview's
numpy versions against FFmpeg's chromakey / v360 filters, the export graph, and the controls."""

import subprocess

import cv2
import numpy as np
import pytest

from kangedit import keying, spherical
from kangedit.export import ExportSettings, build_export
from kangedit.ffmpeg import ffmpeg_bin
from kangedit.models import Footage, OverlayClip, Project, TimelineClip

needs_ffmpeg = pytest.mark.skipif(ffmpeg_bin() is None, reason="FFmpeg not found")


def _ffmpeg(img: np.ndarray, vf: str, w: int, h: int, pix_in="rgb24", pix_out="rgb24", ch=3) -> np.ndarray:
    hh, ww = img.shape[:2]
    out = subprocess.run([ffmpeg_bin(), "-v", "error", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{ww}x{hh}",
                          "-i", "-", "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix_out, "-"],
                         input=img.tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(h, w, ch)


def _screen_shot(color=(0, 177, 64)) -> np.ndarray:
    """A green screen with a skin-coloured person-ish block in the middle (RGB)."""
    img = np.zeros((90, 160, 3), np.uint8)
    img[:] = color
    img[20:70, 50:110] = (205, 150, 120)
    return img


# ----------------------------------------------------------------------------- chroma key
def test_guess_finds_the_screen_colour():
    assert keying.guess(_screen_shot()) is not None
    g = keying.parse(keying.guess(_screen_shot()))
    assert g[1] > 150 and g[0] < 40                     # green
    b = keying.parse(keying.guess(_screen_shot((10, 60, 190))))
    assert b[2] > 150
    assert keying.guess(np.full((90, 160, 3), (205, 150, 120), np.uint8)) is None    # no screen


def test_key_makes_the_screen_see_through_and_keeps_the_subject():
    rgba = np.dstack([_screen_shot(), np.full((90, 160), 255, np.uint8)])
    out = keying.apply(rgba, keying.PRESETS["green"], keying.SIMILARITY, keying.BLEND, True)
    assert out[5, 5, 3] < 10                    # the screen
    assert out[45, 80, 3] > 245                 # the subject
    assert tuple(out[45, 80, :3]) == (205, 150, 120)


@needs_ffmpeg
def test_preview_key_matches_ffmpeg():
    rng = np.random.default_rng(3)
    img = _screen_shot()
    img = np.clip(img.astype(int) + rng.integers(-25, 25, img.shape), 0, 255).astype(np.uint8)
    flt = keying.ffmpeg_filter(keying.PRESETS["green"], 0.12, 0.08, False).rstrip(",")
    want = _ffmpeg(img, flt, 160, 90, pix_out="rgba", ch=4)[..., 3].astype(int)
    got = keying.apply(np.dstack([img, np.full((90, 160), 255, np.uint8)]), keying.PRESETS["green"], 0.12, 0.08,
                       False)[..., 3].astype(int)
    assert np.abs(want - got).mean() < 6


# ----------------------------------------------------------------------------- 360°
def test_detect_projection():
    assert spherical.detect("a.mp4", "equirectangular", 3840, 1920) == "equirect"
    assert spherical.detect("VID_1.insv", "", 5760, 2880) == "dfisheye"
    assert spherical.detect("CAM_1.OSV", "", 3840, 1920) == "dfisheye"
    assert spherical.detect("a.mp4", "", 3840, 1920) == ""            # 2:1 alone is not enough
    assert spherical.detect("a.insv", "", 1920, 1080) == ""


@needs_ffmpeg
@pytest.mark.parametrize("projection", ["equirect", "dfisheye"])
@pytest.mark.parametrize("view", [spherical.View(30, -20, 10, 100), spherical.View(-120, 40, 0, 80),
                                  spherical.View(15, 0, 0, 290, True)])
def test_preview_view_matches_ffmpeg_v360(projection, view):
    rng = np.random.default_rng(1)
    src = np.kron((rng.random((32, 64, 3)) * 255).astype(np.uint8), np.ones((8, 8, 1), np.uint8))
    w, h = (160, 160) if view.tiny else (192, 108)
    want = _ffmpeg(src, spherical.ffmpeg_filter(projection, view, w, h).replace("interp=cubic", "interp=near"), w, h)
    m = spherical.source_maps(projection, view, w, h, src.shape[1], src.shape[0])
    got = cv2.remap(src, m[0], m[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    assert (np.abs(want.astype(int) - got).max(-1) > 40).mean() < 0.005


def test_reproject_sizes_and_caches():
    src = np.zeros((100, 200, 4), np.uint8)
    a = spherical.reproject(src, "equirect", spherical.View(), 64, 36)
    assert a.shape == (36, 64, 4)
    assert spherical.reproject(src, "equirect", spherical.View(), 64, 36).shape == (36, 64, 4)


def _project(tmp_path, projection="equirect"):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    f = Footage(str(video), 30.0, 3840, 1920, 30.0, True, projection=projection)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 4.0, view_yaw=45.0, view_fov=90.0)]
    p.overlays = [OverlayClip(f.id, 0.0, 0.0, 2.0, scale=0.4, view_tiny=True, view_fov=290.0,
                              key_color="#00b140")]
    return p, f


def _graph(tmp_path, p) -> str:
    build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    return (tmp_path / "graph.txt").read_text(encoding="utf-8")


def test_export_shows_each_clips_view_and_keys_the_layer(tmp_path):
    p, _ = _project(tmp_path)
    graph = _graph(tmp_path, p)
    assert "v360=input=e:output=flat:yaw=45.000" in graph and ":w=1920:h=1080" in graph      # the main clip
    assert "v360=input=e:output=sg:yaw=0.000:pitch=-90.000" in graph                          # a tiny planet
    assert ":w=1080:h=1080" in graph
    assert "chromakey=color=0x00b140" in graph.lower() and "despill=type=green" in graph
    p.footage[p.clips[0].footage_id].projection = "dfisheye"
    assert "input=dfisheye:ih_fov=195.00" in _graph(tmp_path, p)


def test_plain_footage_has_no_view(tmp_path):
    p, _ = _project(tmp_path, projection="")
    assert "v360" not in _graph(tmp_path, p)


def test_old_projects_load_without_360_fields():
    f = Footage.from_dict({"path": "a.mp4", "duration": 1.0, "width": 10, "height": 10, "fps": 30.0,
                           "has_audio": False})
    assert f.projection == "" and not spherical.is_360(f)
    c = TimelineClip.from_dict({"footage_id": f.id, "src_in": 0.0, "src_out": 1.0})
    assert c.view_fov == spherical.DEFAULT_FOV and not c.view_tiny
    o = OverlayClip.from_dict({"footage_id": f.id, "start": 0.0})
    assert o.key_color == "" and o.view_yaw == 0.0


def test_inspector_shows_the_view_controls_for_360_clips(qapp, tmp_path):
    from kangedit.ui.inspector import Inspector
    p, f = _project(tmp_path)
    ins = Inspector(p)
    seen = []
    ins.changed.connect(seen.append)
    ins.set_clip(p.clips[0])
    assert not ins.grp_view.isHidden()
    assert ins.vw_yaw.value() == 45.0
    ins.vw_pitch.set_value(-30.0, emit=True)
    assert p.clips[0].view_pitch == -30.0 and seen[-1] == "view"
    ins.chk_tiny.setChecked(True)
    assert p.clips[0].view_tiny and p.clips[0].view_fov == spherical.TINY_FOV
    f.projection = ""
    ins.set_clip(p.clips[0])
    assert ins.grp_view.isHidden()


@needs_ffmpeg
def test_preview_frame_shows_the_same_view_as_the_export():
    from kangedit.color import Grade
    from kangedit.preview import Look, PreviewEngine, Shot
    x = np.linspace(0, 1, 256)
    src = np.zeros((128, 256, 3), np.uint8)       # smooth colours, so bilinear vs cubic hardly matters
    src[..., 0] = (np.sin(x * 2 * np.pi) * 0.5 + 0.5)[None, :] * 255
    src[..., 1] = np.linspace(0, 255, 128)[:, None]
    src[..., 2] = 128
    f = Footage("a.mp4", 1.0, 256, 128, 25.0, False, projection="equirect")
    view = spherical.View(60, 20, 0, 90)
    shot = Shot(0, 0.0, 1.0, 0.0, f.id, Grade(), "none", "filmic", 1.0, 0.0, 0.0, f, view=view)
    e = PreviewEngine(lambda r: None)
    try:
        got = e._finish(src.copy(), shot, Look(False, "fill", 160, 90, 90, color=False), False)
    finally:
        e.close()
    want = _ffmpeg(src, spherical.ffmpeg_filter("equirect", view, 160, 90), 160, 90)
    assert np.abs(cv2.cvtColor(got, cv2.COLOR_BGRA2RGB).astype(int) - want).mean() < 4


# ----------------------------------------------------------------------------- UI
@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.project.dirty = False
    w.close()


def test_auto_key_reads_the_screen_and_the_panel_follows(window, tmp_path):
    from kangedit.analysis import Analysis
    p, f = _project(tmp_path, projection="")
    c = p.overlays[0]
    c.key_color = ""
    window.set_project(p)
    z = np.zeros(4)
    window.analyses[f.id] = Analysis(*[z] * 10, np.zeros(0), np.zeros((4, 144)),
                                     np.stack([_screen_shot((10, 60, 190))] * 4), np.arange(4.0), np.zeros(10))
    window.tl.select_overlay(c.id)
    lp = window.layer_panel
    assert lp.clip is c and not lp.chk_key.isChecked()
    lp.chk_key.setChecked(True)                      # finds the blue screen
    assert keying.parse(c.key_color)[2] > 150 and lp.chk_key.isChecked()
    lp.sl_key_sim.set_value(20.0, emit=True)
    assert c.key_similarity == pytest.approx(0.2)
    lp.chk_key.setChecked(False)
    assert c.key_color == ""
    window.project.undo()
    assert window.project.overlays[0].key_color          # switching it off is one undo step


def test_media_menu_marks_footage_as_360(window, tmp_path):
    p, f = _project(tmp_path, projection="")
    window.set_project(p)
    window.bin_action("projection:equirect", [f.id])
    assert window.project.footage[f.id].projection == "equirect"
    window.bin_action("projection:", [f.id])
    assert window.project.footage[f.id].projection == ""
