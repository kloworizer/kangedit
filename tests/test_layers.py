"""Images, video layers (overlays), rotation, auto ducking, lane heights, the media bin's used marks and updates."""

import io
import json
import os
import tarfile
import threading
import zipfile

import cv2
import numpy as np
import pytest

from kangedit import images, reframe, sfx, updater
from kangedit.export import ExportSettings, build_export
from kangedit.models import Footage, OverlayClip, Project, SoundClip, TimelineClip


def _png(path, w=40, h=20, alpha=True):
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 2] = 255                     # red (BGRA)
    img[..., 3] = 128 if alpha else 255
    cv2.imwrite(str(path), img)
    return str(path)


def _project(tmp_path, overlays=(), sounds=()):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    f = Footage(str(video), 30.0, 1920, 1080, 30.0, True)
    logo = Footage(_png(tmp_path / "logo.png"), images.STILL_LENGTH, 40, 20, 30.0, False, codec="png", kind="image")
    p = Project()
    p.footage = {f.id: f, logo.id: logo}
    p.clips = [TimelineClip(f.id, 0.0, 4.0), TimelineClip(logo.id, 0.0, 3.0, rotation=90)]
    p.overlays = [OverlayClip(**{"footage_id": f.id if k == "video" else logo.id, **kw}) for k, kw in overlays]
    p.sounds = list(sounds)
    return p, f, logo


def _graph(tmp_path, p) -> str:
    build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    return (tmp_path / "graph.txt").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- images
def test_image_probe_and_decoder(tmp_path):
    path = _png(tmp_path / "x.png", 64, 32)
    info = images.probe(path)
    assert (info.width, info.height, info.frames, info.has_alpha) == (64, 32, 1, True)
    dec = images.ImageDecoder(path, max_height=16)
    frame, ok = dec.raw_at(1234.5)
    assert ok and dec.rgba_of(frame).shape == (16, 32, 4)
    rgb = dec.rgb_of(frame)
    assert rgb.shape == (16, 32, 3) and abs(int(rgb[0, 0, 0]) - 128) <= 1   # half-transparent red over black
    assert images.thumbnail(path, 20).shape[0] == 20
    assert images.is_image("A.JPG") and not images.is_image("a.mp4")


def test_animated_frames_loop():
    dec = images.ImageDecoder.__new__(images.ImageDecoder)
    dec.frames, dec.times, dec.loop = [None] * 4, [0.0, 0.1, 0.2, 0.3], 0.4
    assert [dec.index_at(t) for t in (0.0, 0.15, 0.39, 0.55, 4.05)] == [0, 1, 3, 1, 0]


def test_image_clips_export_as_looped_pictures(tmp_path):
    p, _, _ = _project(tmp_path)
    graph = _graph(tmp_path, p)
    assert "premultiply=inplace=1" in graph       # transparent pixels show black
    assert "transpose=1" in graph                 # the image clip turned a quarter right


# ----------------------------------------------------------------------------- rotation
@pytest.mark.parametrize("deg, want", [(0, (0, 0.0)), (90, (1, 0.0)), (-90, (3, 0.0)), (180, (2, 0.0)),
                                       (100, (1, 10.0)), (-30, (0, -30.0)), (46, (1, -44.0))])
def test_split_rotation(deg, want):
    assert reframe.split_rotation(deg) == want


def test_quarter_turn_swaps_the_picture():
    img = np.zeros((10, 20, 3), np.uint8)
    img[0, -1] = 255                        # top right
    out = reframe.rotate_quarters(img, 1)   # clockwise: top right goes to bottom right
    assert out.shape == (20, 10, 3) and out[-1, -1, 0] == 255
    assert reframe.quarter_dims(1920, 1080, 1) == (1080, 1920)
    assert reframe.ffmpeg_quarters(3) == "transpose=2" and reframe.ffmpeg_tilt(0) == ""


def test_export_tilts_rotated_clips(tmp_path):
    p, _, _ = _project(tmp_path)
    p.clips[0].rotation = 12.0
    assert "rotate=0.209440:ow=iw:oh=ih:c=black" in _graph(tmp_path, p)


# ----------------------------------------------------------------------------- overlays
def test_overlay_geometry_and_composite():
    assert reframe.overlay_size(1920, 1080, 1280, 720, 0.5) == (640, 360)
    layer = np.zeros((10, 10, 4), np.uint8)
    layer[..., 0] = 200
    layer[..., 3] = 255
    canvas = np.zeros((20, 20, 4), np.uint8)
    reframe.composite(canvas, layer, 0, 0, 0.5)      # centred on the corner: only a quarter lands
    assert canvas[0, 0, 0] == 100 and canvas[10, 10, 0] == 0
    rot = reframe.overlay_layer(np.full((10, 20, 4), 255, np.uint8), 100, 100, 1.0, 45)
    assert rot.shape[0] > 50 and rot[0, 0, 3] == 0   # rotated on a transparent canvas


def test_overlay_alpha_and_fades():
    c = OverlayClip("f", 2.0, 0.0, 4.0, opacity=0.8, fade_in=1.0, fade_out=1.0)
    assert c.alpha_at(1.9) == 0.0 and c.alpha_at(6.0) == 0.0
    assert c.alpha_at(2.5) == pytest.approx(0.4) and c.alpha_at(4.0) == pytest.approx(0.8)


def test_overlays_save_load_and_undo(tmp_path):
    p, f, logo = _project(tmp_path, overlays=[("logo", {"start": 1.0, "track": 1, "rotation": 30})])
    p.settings.overlay_tracks = 2
    path = str(tmp_path / "p.kgproj")
    p.save(path)
    q = Project.load(path)
    assert [(c.track, c.rotation) for c in q.overlays] == [(1, 30)]
    assert q.footage[logo.id].is_image and q.clips[1].rotation == 90
    assert q.usage() == {f.id: 1, logo.id: 2}
    q.checkpoint()
    q.overlays.clear()
    q.settings.overlay_tracks = 1
    assert q.undo() and len(q.overlays) == 1 and q.settings.overlay_tracks == 2


def test_export_lays_overlays_and_mixes_their_sound(tmp_path):
    p, _, _ = _project(tmp_path, overlays=[("video", {"start": 1.0, "src_in": 2.0, "src_out": 4.0, "scale": 0.4,
                                                      "rotation": -15, "fade_in": 0.5, "opacity": 0.5}),
                                           ("logo", {"start": 0.0, "track": 1})])
    graph = _graph(tmp_path, p)
    assert graph.index("[ov0]overlay") < graph.index("[ov1]overlay")      # bottom layer first
    assert "setpts=PTS+1.0000/TB[ov0]" in graph and "eof_action=pass" in graph
    assert "fade=t=in:st=0:d=0.5000:alpha=1" in graph and "colorchannelmixer=aa=0.5000" in graph
    assert "c=none" in graph                                              # rotated on transparency
    assert "[ova0]" in graph and "[aprog]" in graph                       # the overlay video's sound


# ----------------------------------------------------------------------------- auto ducking
def test_ducked_music_uses_a_sidechain(tmp_path):
    music = tmp_path / "m.wav"
    music.write_bytes(b"")
    p, _, _ = _project(tmp_path, sounds=[SoundClip(str(music), 0.0, 0.0, 6.0, 0.5, "music", duck=True)])
    p.settings.duck_db = 12.0
    graph = _graph(tmp_path, p)
    assert "sidechaincompress" in graph and "mix=0.7488" in graph        # floor: -12 dB
    p.sounds[0].duck = False
    assert "sidechaincompress" not in _graph(tmp_path, p)


def test_fit_to_timeline_keeps_ducking():
    x = SoundClip("m.wav", 0.0, 0.0, 10.0, duck=True)
    assert all(c.duck for c in sfx.fit_to_timeline(x, 10.0, 25.0))


# ----------------------------------------------------------------------------- updates
def test_versions():
    assert updater.parse_version("v1.2.10") == (1, 2, 10) and updater.parse_version("2") == (2, 0, 0)
    assert updater.is_newer("0.10.0", "0.9.9") and not updater.is_newer("v0.2.1", "0.2.1")


def test_release_assets_per_platform():
    r = updater.release_from_json({"tag_name": "v1.0.0", "html_url": "page", "assets": [
        {"name": f"KangEdit-1.0.0-{t}", "browser_download_url": t} for t in
        ("linux-x64.tar.gz", "macos-arm64.zip", "windows-x64.zip")] + [
        {"name": "SHA256SUMS.txt", "browser_download_url": "sums"}]})
    assert r.version == "1.0.0" and r.page == "page"
    assert updater.platform_asset(r, "windows-x64").url == "windows-x64.zip"
    assert updater.platform_asset(r, "linux-arm64") is None


def test_extract_refuses_paths_outside(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(updater.UpdateError):
        updater._extract(str(bad), str(tmp_path / "out"))
    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as t:
        data = b"hello"
        info = tarfile.TarInfo("KangEdit/readme.txt")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    updater._extract(str(good), str(tmp_path / "out"))
    assert (tmp_path / "out" / "KangEdit" / "readme.txt").read_text() == "hello"


def test_release_json_is_parsed_from_the_api_shape():
    data = json.loads('{"tag_name": "v9.9.9", "name": "KangEdit 9.9.9", "body": "notes", "assets": []}')
    r = updater.release_from_json(data)
    assert (r.version, r.title, r.notes) == ("9.9.9", "KangEdit 9.9.9", "notes") and r.source_zip.endswith(".zip")


# ----------------------------------------------------------------------------- UI
@pytest.fixture
def window(qapp, tmp_path):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.project.dirty = False
    w.close()


def test_images_import_and_go_on_layers(window, tmp_path):
    p, f, logo = _project(tmp_path)
    window.set_project(p)
    window.import_files([_png(tmp_path / "pic.png")])
    pic = next(x for x in window.project.footage.values() if x.name == "pic.png")
    assert pic.is_image and pic.id not in window.analyses
    window.add_overlays([pic.id, pic.id], 0.5)
    tracks = sorted(c.track for c in window.project.overlays)
    assert tracks == [0, 1] and window.tl.layers == 2          # the second one goes on a free layer
    oid = window.project.overlays[-1].id
    window._timeline_action("ov_place:tr", oid)
    window._timeline_action("ov_rotate:90", oid)
    c = window._overlay(oid)
    assert (c.x, c.y, c.rotation) == (0.78, 0.22, 90.0)
    window.tl.select_overlay(oid)
    assert window.layer_panel.clip is c and window.right_tabs.currentWidget() is window.layer_panel
    window._delete_selected()
    assert window._overlay(oid) is None
    window._footage_dropped([pic.id], 3.0, -1)                 # on the main track: in at the nearest cut
    assert [x.footage_id for x in window.project.clips] == [f.id, pic.id, logo.id]
    assert window.project.clips[1].duration == images.DEFAULT_LENGTH


def test_media_bin_marks_used_footage(window, tmp_path):
    p, f, logo = _project(tmp_path)
    p.clips = p.clips[:1]
    window.set_project(p)
    items = {window.bin.tree.topLevelItem(i).text(0): window.bin.tree.topLevelItem(i)
             for i in range(window.bin.tree.topLevelItemCount())}
    assert items["a.mp4"].text(1) == "✓" and items["logo.png"].text(1) == "—"
    window.add_overlays([logo.id], 0.0)
    items = {window.bin.tree.topLevelItem(i).text(0): window.bin.tree.topLevelItem(i)
             for i in range(window.bin.tree.topLevelItemCount())}
    assert items["logo.png"].text(1) == "✓"


def test_lane_heights_and_fit(window, tmp_path):
    from kangedit.ui.timeline import HEIGHT_RANGES
    tl = window.tl
    tl.set_heights({"video": 9999, "layer": 1, "sound": 30})
    assert tl.heights == {"video": HEIGHT_RANGES["video"][1], "layer": HEIGHT_RANGES["layer"][0], "sound": 30}
    assert tl.track_h == tl.heights["video"] and tl.sfx_y > tl.track_y + tl.track_h
    p, _, logo = _project(tmp_path, overlays=[("logo", {"start": 20.0, "src_out": 10.0})])
    window.set_project(p)
    tl.resize(1040, tl.height())
    tl.fit()
    assert tl.content_end() == 30.0 and tl.pps == pytest.approx(1000 / 30.0)   # fits the overlay too
    pps = tl.pps
    tl.zoom_by(2.0)
    assert tl.pps == pytest.approx(pps * 2)


def test_rotate_and_duck_actions(window, tmp_path):
    music = tmp_path / "m.wav"
    music.write_bytes(b"")
    p, f, _ = _project(tmp_path, sounds=[SoundClip(str(music), 0.0, 0.0, 6.0)])
    window.set_project(p)
    cid = p.clips[0].id
    window.clip_action("rotate:-90", cid)
    window.clip_action("rotate:-90", cid)
    window.clip_action("rotate:-90", cid)
    assert p.clips[0].rotation == 90.0
    window._sound_action("sound_duck:1", p.sounds[0].id)
    assert p.sounds[0].duck
    assert os.path.basename(p.sounds[0].path) == "m.wav"


# ----------------------------------------------------------------------------- zoom to face
def test_face_detector_model_loads():
    from kangedit import faces
    assert os.path.exists(faces.MODEL) and faces._yunet() is not False   # works on OpenCV 4.8+ and 5
    assert faces.detect(np.zeros((90, 160, 3), np.uint8)) == []


def test_main_face_is_the_one_seen_most(monkeypatch):
    from kangedit import faces
    person = (0.3, 0.4, 0.1, 0.2, 0.9)
    passer = (0.8, 0.5, 0.3, 0.5, 0.95)          # bigger, but in one frame only
    seen = iter([[person], [passer, (0.31, 0.41, 0.1, 0.22, 0.8)], [(0.29, 0.39, 0.1, 0.18, 0.9)], []])
    monkeypatch.setattr(faces, "detect", lambda img: next(seen))
    cx, cy, h = faces.main_face([None] * 4)
    assert (cx, cy, h) == pytest.approx((0.3, 0.4, 0.2))
    monkeypatch.setattr(faces, "detect", lambda img: [])
    assert reframe.find_face([np.zeros((4, 4, 3), np.uint8)]) is None


def test_looping_sound_renders_as_one_crossfaded_clip(tmp_path):
    music = tmp_path / "m.wav"
    music.write_bytes(b"")
    [y] = sfx.fit_to_timeline(SoundClip(str(music), 0.0, 0.0, 5.0), 5.0, 14.0)
    p, _, _ = _project(tmp_path, sounds=[y])
    graph = _graph(tmp_path, p)
    assert y.repeats() == 4 and "asplit=4" in graph and graph.count("acrossfade=d=2.0000:c1=qsin") == 3


def test_fit_leaves_nothing_to_scroll_and_reset_layout(window, tmp_path, qapp):
    from kangedit.ui.timeline import DEFAULT_HEIGHTS
    p, _, _ = _project(tmp_path)
    p.clips[0].src_out = 3600.0     # an hour: below the old zoom limit
    os.remove(p.footage[p.clips[0].footage_id].path)   # nothing to analyze (a failed analysis opens a dialog)
    window.set_project(p)
    window.resize(1400, 900)
    window.show()
    qapp.processEvents()
    window.tl.fit()
    qapp.processEvents()
    assert window.timeline.scroll.maximum() == 0 and not window.timeline.scroll.isVisible()
    window.tl.zoom_by(4.0)
    qapp.processEvents()
    assert window.timeline.scroll.isVisible()
    window.tl.set_heights({"video": 200})
    window.right_tabs.setCurrentIndex(2)
    window.reset_layout()
    for _ in range(3):
        qapp.processEvents()
    assert window.tl.heights == DEFAULT_HEIGHTS and window.right_tabs.currentIndex() == 0
    assert window.timeline.scroll.maximum() == 0


def test_ctrl_a_then_conversion_lut_changes_every_selected_clip(window, qapp, lut_library, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog
    from kangedit.color import lut_profile
    from .conftest import write_cube
    p = Project()
    fs = [Footage(str(tmp_path / f"{n}.mp4"), 30.0, 1920, 1080, 30.0, True) for n in "abc"]
    p.footage = {f.id: f for f in fs}
    p.clips = [TimelineClip(fs[0].id, 0, 3), TimelineClip(fs[1].id, 0, 3), TimelineClip(fs[2].id, 0, 3),
               TimelineClip(fs[0].id, 5, 8)]
    window.set_project(p)
    window.show()
    tl = window.tl
    QTest.mouseClick(tl, Qt.LeftButton, Qt.NoModifier, tl._clip_rects()[1][1].center().toPoint())
    tl.setFocus()
    QTest.keyClick(tl, Qt.Key_A, Qt.ControlModifier)
    qapp.processEvents()
    assert len(tl.selected_ids()) == 4 and len(window.inspector.targets) == 4

    # picked from the list
    first = write_cube(lut_library / "first.cube")
    window.inspector.reload_luts()
    cb = window.inspector.cb_log
    cb.setCurrentIndex(cb.findData(lut_profile(first)))
    assert all(f.log_profile == lut_profile(first) for f in p.footage.values())

    # imported from the list's "Import LUT (.cube)…" entry, as a conversion and as a look
    src = write_cube(tmp_path / "conv.cube", 3)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([src], ""))
    window.inspector_request("import_lut_log")
    conv = [f.log_profile for f in p.footage.values()]
    assert len(set(conv)) == 1 and conv[0] != lut_profile(first) and conv[0].endswith("conv.cube")
    window.inspector_request("import_lut_look")
    looks = {c.grade.look for c in p.clips}
    assert len(looks) == 1 and looks.pop().endswith(".cube")


# ----------------------------------------------------------------------------- refining clips on every layer
def _layered(window, tmp_path):
    p, f, logo = _project(tmp_path, overlays=[("video", {"start": 0.5, "src_in": 2.0, "src_out": 5.0, "track": 0}),
                                              ("video", {"start": 1.0, "src_in": 6.0, "src_out": 8.0, "track": 1}),
                                              ("logo", {"start": 0.0, "src_out": 4.0, "track": 2})])
    window.set_project(p)
    return p, f, logo


def test_inspector_refines_a_layer_clip(window, tmp_path):
    p, f, _ = _layered(window, tmp_path)
    o = p.overlays[1]                                   # on V3
    window.tl.select_overlay(o.id)
    ins = window.inspector
    assert ins.clip is o and "layer V3" in ins.lbl_title.text()
    assert ins.b_layer.isVisibleTo(ins) and not ins.grp_edit.isVisibleTo(ins) and not ins.grp_reframe.isVisibleTo(ins)
    ins.params["exposure"].set_value(0.7, emit=True)
    ins.params["saturation"].set_value(120, emit=True)
    assert (o.grade.exposure, o.grade.saturation) == (0.7, 120.0)
    assert p.clips[0].grade.exposure == 0.0 and p.overlays[0].grade.exposure == 0.0   # only that clip
    ins.sp_out.setValue(7.5)                             # trim its source
    assert o.src_out == 7.5
    ins.vol.set_value(40, emit=True)
    assert o.volume == pytest.approx(0.4)
    ins.cb_log.setCurrentIndex(ins.cb_log.findData("vlog"))   # the footage's log profile
    assert f.log_profile == "vlog"
    window.undo()
    assert f.log_profile != "vlog"
    for _ in range(3):                                   # volume, trim, saturation: one step each
        window.undo()
    o = window._overlay(o.id)
    assert o.src_out == 8.0 and o.volume == 1.0 and o.grade.saturation == 100.0 and o.grade.exposure == 0.7
    window.undo()
    assert window._overlay(o.id).grade.exposure == 0.0
    # back on the main track the Inspector shows the main clip again, with its transition and reframe
    window.tl.selected = p.clips[1].id
    window.tl.select_overlay("")
    window._on_select(p.clips[1].id)
    assert window.inspector.clip is window.project.clips[1] and window.inspector.grp_reframe.isVisibleTo(ins)


def test_layer_grade_reaches_the_preview_and_the_export(window, tmp_path):
    p, f, _ = _layered(window, tmp_path)
    o = p.overlays[0]
    window.tl.select_overlay(o.id)
    window.inspector.params["exposure"].set_value(1.0, emit=True)
    shots = window.viewer._overlay_shots()
    assert next(s for s in shots if s.clip.id == o.id).grade.exposure == 1.0
    g = _graph(tmp_path, window.project)
    assert "lut_ov0.cube" in g or "lut_ov1.cube" in g


def test_copy_and_reset_grade_across_layers(window, tmp_path):
    p, f, logo = _layered(window, tmp_path)
    a, b, pic = p.overlays
    a.grade.contrast = 25.0
    window._timeline_action("ov_copy_grade", a.id)
    assert b.grade.contrast == 25.0 and p.clips[0].grade.contrast == 25.0
    assert pic.grade.contrast == 0.0                   # a picture on a layer (logo) is left alone
    window._timeline_action("ov_reset_color", b.id)
    assert b.grade.contrast == 0.0 and a.grade.contrast == 25.0
    window.undo()
    assert window._overlay(b.id).grade.contrast == 25.0
    # from a main-track clip, "Copy to all clips" reaches the layers too
    c = window.project.clips[0]
    c.grade.temperature = -12.0
    window.tl.selected = c.id
    window._on_select(c.id)
    window.inspector_request("copy_grade_all")
    assert all(o.grade.temperature == -12.0 for o in window.project.overlays[:2])
    assert window.project.overlays[2].grade.temperature == 0.0


def test_auto_color_covers_every_layer(window, tmp_path, monkeypatch):
    import kangedit.ui.main_window as mw
    from kangedit.color import Grade

    class FakeDec:
        def frame_at(self, t):
            return np.full((36, 64, 3), 90, np.uint8)

    class FakePool:
        def __init__(self, *a, **k):
            pass

        def get(self, *a, **k):
            return FakeDec()

        def close(self):
            pass

    monkeypatch.setattr(mw, "DecoderPool", FakePool)
    monkeypatch.setattr(mw, "auto_correct", lambda frames, lp, tm, base=None: Grade(exposure=0.4))
    monkeypatch.setattr(window.tasks, "submit",
                        lambda name, job, done, err=None: done(job(lambda *a: None, threading.Event())))
    p, f, logo = _layered(window, tmp_path)
    window._timeline_action("ov_auto_color", p.overlays[1].id)
    assert p.overlays[1].grade.exposure == 0.4 and p.overlays[0].grade.exposure == 0.0
    window.inspector_request("auto_color_all")
    assert all(c.grade.exposure == 0.4 for c in p.clips) and p.overlays[0].grade.exposure == 0.4
    assert p.overlays[2].grade.exposure == 0.0         # the logo
    window.undo()
    assert window.project.overlays[0].grade.exposure == 0.0


def test_layer_tab_and_menu_lead_to_the_color_tools(window, tmp_path):
    p, f, _ = _layered(window, tmp_path)
    o = p.overlays[0]
    window.tl.select_overlay(o.id)
    assert window.right_tabs.currentWidget() is window.layer_panel
    window.layer_panel.b_color.click()
    assert window.right_tabs.currentWidget() is window.inspector and window.inspector.clip is o
    window.inspector.b_layer.click()
    assert window.right_tabs.currentWidget() is window.layer_panel
