"""Gaming extras: the facecam layout (the webcam corner of a game recording big beside the game) and labels at
the edit's multi-kills."""

import numpy as np
import pytest

from kangedit import facecam, games, multicam, reframe
from kangedit.config import frame_size
from kangedit.models import Footage, Project, TextClip, TimelineClip
from kangedit.sync import SyncResult

from .test_games import KILL, TRIPLE, gameplay

VERTICAL, LANDSCAPE = "9:16 (Vertical)", "16:9 (Landscape)"


def _game_project(aspect=LANDSCAPE, clips=((10.0, 30.0), (45.0, 60.0))):
    f = Footage("game.mp4", 130.0, 1920, 1080, 60.0, True)
    p = Project()
    p.settings.aspect = aspect
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, a, b) for a, b in clips]
    return p, f


# ----------------------------------------------------------------------------- labelling the plays
def test_labels_at_the_triple_kill_and_relabelling_replaces_them():
    p, f = _game_project()
    an = {f.id: gameplay()}
    p.texts = [TextClip("My title", 1.0)]
    assert games.apply_labels(p, an) == 1
    lab = [x for x in p.texts if x.preset == games.LABEL_PRESET]
    assert len(lab) == 1 and lab[0].text == "TRIPLE KILL"
    # at the third kill, on the timeline: the second clip starts at 20 s and shows source 45 s onwards
    assert lab[0].start == pytest.approx(20.0 + TRIPLE[-1] - 45.0, abs=0.3)
    assert lab[0].duration == pytest.approx(games.LABEL_SECONDS) and lab[0].bold
    # again (with single plays and other words): the old labels are replaced, other titles stay
    assert games.apply_labels(p, an, {3: "THREE!"}, singles=True) == 2
    texts = sorted((x.text, round(x.start)) for x in p.texts if x.preset == games.LABEL_PRESET)
    assert texts == [("KILL", round(KILL - 10.0)), ("THREE!", round(20.0 + TRIPLE[-1] - 45.0))]
    assert any(x.text == "My title" for x in p.texts)


def test_labels_count_only_the_bursts_inside_the_clip_and_undo():
    p, f = _game_project(clips=((TRIPLE[0] + 1.0, 70.0),))        # the first kill of the three is cut off
    p.checkpoint()
    games.apply_labels(p, {f.id: gameplay()})
    assert [x.text for x in p.texts] == ["DOUBLE KILL"]
    p.undo()
    assert p.texts == []
    assert games.label_text(5) == "MULTI KILL ×5" and games.label_text(1) == "KILL"
    assert games.clusters([1.0, 3.0, 4.5, 20.0, 30.0, 33.0]) == [[1.0, 3.0, 4.5], [20.0], [30.0, 33.0]]


# ----------------------------------------------------------------------------- finding the webcam
def test_a_steady_corner_face_makes_the_cam_box():
    cam = (0.1, 0.85, 0.05, 0.09, 0.9)
    frames = [[cam], [cam, (0.5, 0.45, 0.2, 0.3, 0.95)], [cam], [(0.8, 0.2, 0.05, 0.08, 0.7)], [cam]]
    face = facecam.stable_face(frames)                 # not the big middle face, not the one-frame game face
    assert face is not None and face[:2] == pytest.approx(cam[:2])
    x, y, w, h = facecam.cam_rect(face, 1920, 1080)
    assert x == 0.0 and y + h == pytest.approx(1.0)                # against the left and bottom edges
    assert w * 1920 / (h * 1080) == pytest.approx(16 / 9, rel=0.02)
    assert x < cam[0] < x + w and y < cam[1] < y + h and h == pytest.approx(cam[3] * facecam.HEAD)
    assert facecam.corner_of((x, y, w, h)) == "bl"
    # a face only in the middle (a talking-head video), or seen once, is no facecam
    assert facecam.stable_face([[(0.5, 0.4, 0.2, 0.3, 0.9)]] * 4) is None
    assert facecam.stable_face([[cam], [], [], [], []]) is None
    assert facecam.find([]) is None


@pytest.mark.parametrize("corner", list(facecam.CORNERS))
def test_a_corner_picked_by_hand(corner):
    rect = facecam.corner_rect(corner, 1920, 1080)
    assert facecam.corner_of(rect) == corner and rect[2] == facecam.CORNER_W
    assert 0.0 <= rect[0] and rect[0] + rect[2] <= 1.0 and 0.0 <= rect[1] and rect[1] + rect[3] <= 1.0
    p, f = _game_project()
    tw, th = frame_size(p.settings.aspect, p.settings.resolution)
    oc = facecam.lay_out(p.clips[0], f, rect, tw, th, 0.0)
    assert ("l" in corner) == (oc.x < 0.5) and ("t" in corner) == (oc.y < 0.5)


# ----------------------------------------------------------------------------- the layout
def _shown(sw, sh, p: reframe.Placement):
    """The part of the picture a placement shows: (x0, y0, x1, y1), parts of the picture."""
    return (p.crop_x / p.scaled_w, p.crop_y / p.scaled_h, (p.crop_x + p.crop_w) / p.scaled_w,
            (p.crop_y + p.crop_h) / p.scaled_h)


def test_stacked_layout_for_vertical():
    p, f = _game_project(VERTICAL)
    tw, th = frame_size(p.settings.aspect, p.settings.resolution)
    cam = facecam.cam_rect((0.1, 0.85, 0.05, 0.09), 1920, 1080)
    clip = p.clips[0]
    oc = facecam.lay_out(clip, f, cam, tw, th, 0.0)
    # the cam: a full-width band on the top third, showing the webcam box of the same recording
    assert (oc.x, oc.y, oc.box_w, oc.box_h) == pytest.approx((0.5, facecam.BAND / 2, 1.0, facecam.BAND))
    assert oc.footage_id == f.id and (oc.src_in, oc.src_out) == (clip.src_in, clip.src_out)
    assert oc.link == clip.id and oc.role == facecam.FACECAM and oc.volume == 0.0 and oc.zoom > 1.5
    x0, y0, x1, y1 = _shown(1920, 1080, reframe.overlay_place(1920, 1080, tw, th, oc.scale, oc.box_w, oc.box_h,
                                                              oc.pan_x, oc.pan_y, oc.zoom))
    assert cam[0] - 0.01 <= x0 and x1 <= cam[0] + cam[2] + 0.01 and cam[1] - 0.01 <= y0 and y1 <= cam[1] + cam[3] + 0.01
    assert (x0 + x1) / 2 == pytest.approx(cam[0] + cam[2] / 2, abs=0.01)
    # the game: the rest of the frame below, the game area beside the cam (not showing it)
    assert (clip.box_x, clip.box_y, clip.box_w, clip.box_h) == pytest.approx((0.5, 2 / 3, 1.0, 2 / 3))
    bw, bh = reframe.box_rect(tw, th, (clip.box_x, clip.box_y, clip.box_w, clip.box_h))[2:]
    gx0, gy0, gx1, gy1 = _shown(1920, 1080, reframe.place(1920, 1080, bw, bh, "fill", clip.zoom, clip.offset_x,
                                                          clip.offset_y))
    assert gx0 >= cam[0] + cam[2] - 0.01 and (gy0, gy1) == pytest.approx((0.0, 1.0), abs=0.01)


def test_picture_in_picture_for_landscape_covers_the_cam_corner():
    p, f = _game_project(LANDSCAPE)
    tw, th = frame_size(p.settings.aspect, p.settings.resolution)
    cam = facecam.cam_rect((0.92, 0.12, 0.04, 0.08), 1920, 1080)     # top right
    clip = p.clips[0]
    clip.box_w, clip.zoom = 0.5, 2.0                                 # an earlier framing is cleared
    oc = facecam.lay_out(clip, f, cam, tw, th, 0.0)
    assert clip.box_w == 0.0 and clip.zoom == 1.0
    # flush in the top right corner, bigger than the recording's own cam, which it covers
    assert oc.x + oc.box_w / 2 == pytest.approx(1.0) and oc.y - oc.box_h / 2 == pytest.approx(0.0)
    assert oc.box_w >= cam[2] * facecam.PIP_GROW - 1e-6 and oc.box_w <= facecam.PIP_MAX + 1e-6
    assert 1.0 - oc.box_w <= cam[0] + 1e-6 and oc.box_h >= cam[1] + cam[3] - 1e-6
    x0, y0, x1, y1 = _shown(1920, 1080, reframe.overlay_place(1920, 1080, tw, th, oc.scale, oc.box_w, oc.box_h,
                                                              oc.pan_x, oc.pan_y, oc.zoom))
    assert (x0, y0, x1, y1) == pytest.approx((cam[0], cam[1], cam[0] + cam[2], cam[1] + cam[3]), abs=0.01)


def test_the_cam_layer_shows_the_corner_of_the_picture():
    """The crop zoom reaches the layer's picture (as the preview and the export draw it)."""
    img = np.zeros((1080, 1920, 4), np.uint8)
    img[..., 3] = 255
    img[756:, :540, :3] = (0, 200, 0)                              # the webcam corner: green
    cam = facecam.corner_rect("bl", 1920, 1080, 16 / 9)
    p, f = _game_project(VERTICAL)
    oc = facecam.lay_out(p.clips[0], f, cam, 1080, 1920, 0.0)
    layer = reframe.overlay_layer(img, 1080, 1920, oc.scale, 0.0, oc.box_w, oc.box_h, oc.pan_x, oc.pan_y,
                                  zoom=oc.zoom)
    assert layer.shape[:2] == (640, 1080) and (layer[..., 1] > 150).mean() > 0.97


def test_the_facecam_follows_trims_splits_and_deletes():
    p, f = _game_project(VERTICAL, clips=((10.0, 30.0), (45.0, 60.0)))
    tw, th = frame_size(p.settings.aspect, p.settings.resolution)
    cam = facecam.corner_rect("bl", 1920, 1080)
    p.checkpoint()
    assert facecam.apply(p, [p.clips[1]], cam, tw, th) == 1
    oc = p.overlays[0]
    assert oc.start == pytest.approx(20.0) and oc.link == p.clips[1].id
    # trims and slips of either clip: the cam keeps the main clip's range and place
    p.clips[0].src_out -= 5.0
    p.clips[1].src_in += 2.0
    assert multicam.follow(p)
    oc = p.overlays[0]
    assert (oc.start, oc.src_in, oc.src_out) == pytest.approx((15.0, 47.0, 60.0))
    # a split: the new piece gets the cam too (the first clip, never laid out, doesn't)
    second = TimelineClip.from_dict(p.clips[1].to_dict())
    second.id = TimelineClip(f.id, 0, 1).id
    p.clips[1].src_out, second.src_in = 52.0, 52.0
    p.clips.append(second)
    multicam.follow(p)
    cams = sorted(p.overlays, key=lambda o: o.start)
    assert [o.link for o in cams] == [p.clips[1].id, second.id] and cams[1].start == pytest.approx(20.0)
    assert (cams[1].box_w, cams[1].zoom) == (cams[0].box_w, cams[0].zoom)
    # deleted by hand: stays deleted; its main clip deleted: it goes too
    p.overlays = [o for o in p.overlays if o.link != second.id]
    multicam.follow(p)
    assert [o.link for o in p.overlays] == [p.clips[1].id]
    p.clips.pop(1)
    multicam.follow(p)
    assert p.overlays == []
    p.undo()                                       # back to before the layout
    assert p.overlays == [] and p.clips[1].box_w == 0.0 and p.settings.facecam == {}


def test_a_new_edit_of_the_recording_keeps_the_layout():
    p, f = _game_project(VERTICAL)
    tw, th = frame_size(p.settings.aspect, p.settings.resolution)
    facecam.apply(p, list(p.clips), facecam.corner_rect("tl", 1920, 1080), tw, th)
    framing = (p.clips[0].box_y, p.clips[0].zoom, p.clips[0].offset_x)
    p.clips = [TimelineClip(f.id, 70.0, 75.0), TimelineClip(f.id, 80.0, 84.0)]     # a new auto edit
    multicam.follow(p)
    assert sorted(o.link for o in p.overlays) == sorted(c.id for c in p.clips)
    assert all((c.box_y, c.zoom, c.offset_x) == framing for c in p.clips)
    assert [o.start for o in sorted(p.overlays, key=lambda o: o.start)] == pytest.approx([0.0, 5.0])


def test_a_separate_webcam_file_is_a_multicam_angle_in_the_band():
    p, game = _game_project(VERTICAL)
    cam = Footage("cam.mp4", 130.0, 1280, 720, 30.0, True, face=[0.5, 0.4, 0.3])
    p.footage[cam.id] = cam
    tw, th = frame_size(p.settings.aspect, p.settings.resolution)
    angles, syncs = {game.id: 1, cam.id: 2}, {cam.id: (game.id, SyncResult(2.0, 0.0, 0.9, 120.0))}
    new = multicam.build(p, p.clips, angles, syncs, "stack", "main", layout_main=False)
    for oc in new:
        facecam.cam_overlay_place(oc, cam, tw, th, "bl")
    for c in p.clips:
        facecam.game_box(c, game, tw, th)
    p.overlays = new
    p.settings.multicam = {"angles": angles, "template": "stack", "audio": "main",
                           "syncs": {cam.id: [game.id, 2.0, 0.0, 0.9, 120.0]}, "clips": [c.id for c in p.clips]}
    p.clips[0].src_in += 1.0
    multicam.follow(p)
    o = next(o for o in p.overlays if o.link == p.clips[0].id)
    assert o.src_in == pytest.approx(p.clips[0].src_in - 2.0)          # in sync with the game
    assert (o.y, o.box_w, o.box_h) == pytest.approx((facecam.BAND / 2, 1.0, facecam.BAND))
    assert p.clips[0].box_y == pytest.approx(2 / 3)


def test_the_face_detector_finds_a_drawn_webcam_corner_when_it_can():
    """The real detector on a drawn face in the bottom-left corner of a noisy "game" (skipped when the model
    doesn't take the drawing for a face: the geometry above is what is relied on)."""
    import cv2
    from kangedit import faces
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(4):
        img = cv2.GaussianBlur((rng.random((720, 1280, 3)) * 255).astype(np.uint8), (31, 31), 0)
        cv2.rectangle(img, (0, 500), (380, 720), (90, 110, 130), -1)
        cx, cy = 190, 600
        cv2.ellipse(img, (cx, cy), (45, 60), 0, 0, 360, (200, 160, 130), -1)
        cv2.ellipse(img, (cx, cy - 55), (50, 25), 0, 180, 360, (40, 30, 20), -1)
        for ex in (cx - 18, cx + 18):
            cv2.ellipse(img, (ex, cy - 10), (9, 5), 0, 0, 360, (250, 250, 250), -1)
            cv2.circle(img, (ex, cy - 10), 4, (30, 20, 10), -1)
            cv2.line(img, (ex - 10, cy - 22), (ex + 10, cy - 24), (40, 30, 20), 3)
        cv2.line(img, (cx, cy - 5), (cx - 4, cy + 15), (160, 120, 100), 2)
        cv2.ellipse(img, (cx, cy + 30), (16, 6), 0, 0, 180, (150, 60, 60), 3)
        frames.append(img)
    if not faces.detect(frames[0]):
        pytest.skip("the face detector doesn't see the drawing as a face")
    rect = facecam.find(frames)
    assert rect is not None and facecam.corner_of(rect) == "bl"
    assert rect[0] < 190 / 1280 < rect[0] + rect[2] and rect[1] < 600 / 720 < rect[1] + rect[3]


# ----------------------------------------------------------------------------- in the app
@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    w._set_recent([])
    yield w
    w.project.dirty = False
    w.close()


def test_the_tools_lay_out_the_facecam_and_label_the_plays(window, monkeypatch):
    import threading
    from PySide6.QtWidgets import QDialog, QInputDialog
    from kangedit.ui import main_window as mw
    from kangedit.ui.games_dialog import PlayLabelsDialog
    p, f = _game_project(VERTICAL)
    window.set_project(p)

    class Dec:
        def frame_at(self, t):
            return np.zeros((72, 128, 3), np.uint8)       # no face: the corner is asked for

    class Pool:
        def __init__(self, limit=4):
            pass

        def get(self, *a):
            return Dec()

        def close(self):
            pass

    monkeypatch.setattr(mw, "DecoderPool", Pool)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("Top left", True))
    monkeypatch.setattr(window.tasks, "submit",
                        lambda name, job, done, err=None: done(job(lambda *a: None, threading.Event())))
    window.facecam_layout()
    assert [o.link for o in p.overlays] == [c.id for c in p.clips] and all(o.role == "facecam" for o in p.overlays)
    assert all(c.box_w == 1.0 for c in p.clips) and "Facecam layout" in window.lbl_status.text()
    window.undo()
    assert window.project.overlays == []
    window.analyses[f.id] = gameplay()
    monkeypatch.setattr(PlayLabelsDialog, "exec", lambda self: QDialog.Accepted)
    window.label_plays()
    assert [x.text for x in window.project.texts] == ["TRIPLE KILL"]


def test_the_export_crops_the_cam_corner(tmp_path):
    import re
    from kangedit.export import ExportSettings, build_export
    p, f = _game_project(VERTICAL, clips=((10.0, 14.0),))
    video = tmp_path / "game.mp4"
    video.write_bytes(b"")
    f.path = str(video)
    tw, th = frame_size(p.settings.aspect, p.settings.resolution)
    facecam.apply(p, p.clips, facecam.corner_rect("bl", 1920, 1080), tw, th)
    oc = p.overlays[0]
    build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    graph = (tmp_path / "graph.txt").read_text(encoding="utf-8")
    pl = reframe.overlay_place(1920, 1080, tw, th, 1.0, oc.box_w, oc.box_h, oc.pan_x, oc.pan_y, oc.zoom)
    assert f"scale={pl.scaled_w}:{pl.scaled_h}:flags=lanczos,crop={pl.crop_w}:{pl.crop_h}:{pl.crop_x}:{pl.crop_y}" in graph
    assert pl.scaled_w > 3 * tw and pl.crop_y > 0                       # magnified, the bottom-left corner
    assert re.search(r"\[ov0\]overlay=x='540\.00-w/2':y='320\.00-h/2'", graph)
