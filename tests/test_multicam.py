"""Multicam: syncing angles by sound, laying them out, and keeping them in step with the edit."""

import numpy as np
import pytest

from kangedit import multicam, reframe
from kangedit.models import Footage, OverlayClip, Project, TimelineClip
from kangedit.sync import RATE, SyncResult, sync_audio

rng = np.random.default_rng(7)


def _speech(seconds: float) -> np.ndarray:
    """Noise in syllable-like bursts with pauses: enough like speech for onsets and fine matching."""
    n = int(seconds * RATE)
    x = np.convolve(rng.standard_normal(n), np.hanning(9) / 4, "same").astype(np.float32)
    env = np.zeros(n, np.float32)
    t = 0
    while t < n:
        k = int(rng.uniform(0.08, 0.35) * RATE)
        env[t:t + k] = rng.uniform(0.2, 1.0) * np.hanning(min(k, n - t))
        t += k + int(rng.exponential(0.15) * RATE)
    return x * env * 0.3


def _angle(a: np.ndarray, offset: float, seconds: float, drift: float = 0.0) -> np.ndarray:
    """What a second camera starting `offset` s into recording a would hear (another mic, a little noise)."""
    ta = offset + np.arange(int(seconds * RATE)) / RATE * (1 + drift)
    b = np.interp(ta * RATE, np.arange(len(a)), a, left=0, right=0).astype(np.float32)
    return np.convolve(b, [0.6, 0.3, 0.1], "same") * 2.0 + rng.standard_normal(len(b)).astype(np.float32) * 0.01


A = _speech(240)


@pytest.mark.parametrize("offset, drift", [(13.3721, 0.0), (61.25, 40e-6), (-9.5, -30e-6)])
def test_sync_is_sub_millisecond_and_finds_drift(offset, drift):
    b = _angle(A, offset, 150, drift)
    r = sync_audio(A, b)
    assert r.reliable and r.confidence > 0.8
    for tb in (5.0, 75.0, 145.0):
        assert r.a_time(tb) == pytest.approx(offset + tb * (1 + drift), abs=0.001)    # within 1 ms
        assert r.b_time(r.a_time(tb)) == pytest.approx(tb, abs=1e-6)
    assert r.drift == pytest.approx(drift, abs=5e-6)


def test_unrelated_sound_does_not_sync():
    assert not sync_audio(A, _speech(100)).reliable


# ----------------------------------------------------------------------------- layout and timing
def _project(aspect="16:9 (Landscape)"):
    main = Footage("main.mp4", 120.0, 1920, 1080, 30.0, True)
    cam2 = Footage("cam2.mp4", 60.0, 1080, 1920, 30.0, True, face=[0.3, 0.35, 0.2])
    p = Project()
    p.settings.aspect = aspect
    p.footage = {main.id: main, cam2.id: cam2}
    p.clips = [TimelineClip(main.id, 30.0, 33.0), TimelineClip(main.id, 50.0, 52.0),
               TimelineClip(main.id, 5.0, 8.0)]                        # angle 2 starts at 20 s: not here
    angles = {main.id: 1, cam2.id: 2}
    syncs = {cam2.id: (main.id, SyncResult(20.0, 0.0, 1.0, 60.0))}    # cam2's 0 s = main's 20 s
    return p, main, cam2, angles, syncs


def _store(p, angles, syncs, template):
    p.overlays = multicam.build(p, p.clips, angles, syncs, template)
    p.settings.multicam = {"angles": angles, "template": template, "audio": "main",
                           "syncs": {f: [ref, r.offset, r.drift, r.confidence, r.overlap] for f, (ref, r) in syncs.items()},
                           "clips": [c.id for c in p.clips]}


def test_angles_cut_at_the_same_moments():
    p, main, cam2, angles, syncs = _project()
    _store(p, angles, syncs, "pip")
    starts = p.clip_starts()
    ov = sorted(p.overlays, key=lambda o: o.start)
    assert [o.angle for o in ov] == [2, 2]                     # the third clip is before cam2 started
    for o, c, t0 in zip(ov, p.clips, starts):
        assert o.link == c.id and o.start == pytest.approx(t0) and o.src_in == pytest.approx(c.src_in - 20.0)
        assert o.duration == pytest.approx(c.duration) and o.volume == 0.0 and o.box_w == 0.0
    assert all(c.volume == 1.0 for c in p.clips)


def test_stacked_layout_two_angles_use_the_main_track_and_one_layer(tmp_path):
    p, main, cam2, angles, syncs = _project("9:16 (Vertical)")
    _store(p, angles, syncs, "stack")
    # 2 angles: the main track (in the top box) and one layer (the bottom box), not a copy of the main angle
    assert {o.angle for o in p.overlays} == {2} and {o.track for o in p.overlays} == {0}
    assert all((c.box_x, c.box_y, c.box_w, c.box_h) == (0.5, 0.25, 1.0, 0.5) for c in p.clips)
    o = p.overlays[0]
    assert (o.y, o.box_h) == (0.75, 0.5) and o.pan_y != 0.0       # cropped toward the face
    # the export shows the main clip in its box, black around it
    for f in p.footage.values():
        f.path = str(tmp_path / f.name)
        open(f.path, "wb").close()
    from kangedit.export import ExportSettings, build_export
    build_export(p, ExportSettings(str(tmp_path / "o.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    graph = (tmp_path / "graph.txt").read_text()
    assert "pad=1080:1920:0:0:black" in graph and "scale=1706:960" in graph   # 16:9 filling a 1080x960 box


def test_main_box_in_the_preview():
    from kangedit.preview import Look, Shot, PreviewEngine
    from kangedit.color import Grade
    img = np.full((90, 160, 3), 200, np.uint8)
    f = Footage("x.mp4", 1.0, 160, 90, 30.0, False)
    shot = Shot(0, 0.0, 1.0, 0.0, f.id, Grade(), "none", "filmic", 1.0, 0.0, 0.0, f, box=(0.5, 0.25, 1.0, 0.5))
    out = PreviewEngine._finish(PreviewEngine.__new__(PreviewEngine), img, shot,
                                Look(False, "fill", 108, 192, 192, False), False)
    assert out.shape == (192, 108, 4) and out[40, 50, 0] == 200 and out[150, 50, 0] == 0


def test_audio_from_another_angle():
    p, main, cam2, angles, syncs = _project()
    p.overlays = multicam.build(p, p.clips, angles, syncs, "pip", "angle:2")
    assert all(c.volume == 0.0 for c in p.clips) and all(o.volume == 1.0 for o in p.overlays)


def test_angles_follow_trims_splits_and_deletes():
    p, main, cam2, angles, syncs = _project()
    _store(p, angles, syncs, "pip")
    first = next(o for o in p.overlays if o.link == p.clips[0].id)
    first.x, first.scale = 0.2, 0.5                              # placed by hand
    p.clips[0].src_in += 1.0                                     # trimmed
    assert multicam.follow(p)
    o = next(o for o in p.overlays if o.link == p.clips[0].id)
    assert o.src_in == pytest.approx(p.clips[0].src_in - 20.0) and (o.x, o.scale) == (0.2, 0.5)
    assert p.overlays[[x.id for x in p.overlays].index(o.id)] is o and o.id == first.id
    # a deleted angle clip stays deleted
    p.overlays = [x for x in p.overlays if x.link != p.clips[1].id]
    multicam.follow(p)
    assert not any(x.link == p.clips[1].id for x in p.overlays)
    # a new main clip (e.g. after a split) gets its angle, placed like the others
    p.clips.insert(1, TimelineClip(main.id, 40.0, 41.0))
    multicam.follow(p)
    new = next(x for x in p.overlays if x.link == p.clips[1].id)
    assert new.src_in == pytest.approx(20.0) and (new.x, new.scale) == (0.2, 0.5)
    # a removed main clip takes its angle with it
    gone = p.clips.pop(0).id
    multicam.follow(p)
    assert not any(x.link == gone for x in p.overlays)


def test_relayout_boxes_and_unboxes_the_main_track():
    p, main, cam2, angles, syncs = _project()
    _store(p, angles, syncs, "pip")
    assert multicam.relayout(p, "side") == 2
    assert all((c.box_x, c.box_w) == (0.25, 0.5) for c in p.clips) and p.settings.multicam["template"] == "side"
    assert p.settings.overlay_tracks == 1 and not any(o.angle == 1 for o in p.overlays)
    multicam.relayout(p, "talking")                               # landscape: main full frame again
    assert all(c.box_w == 0 for c in p.clips)
    box = next(o for o in p.overlays if o.angle == 2)
    w, h = reframe.overlay_size(1080, 1920, 1920, 1080, box.scale, box.box_w, box.box_h)
    assert abs(w - h) <= 2 and box.x + w / 1920 / 2 == pytest.approx(1 - multicam.MARGIN, abs=0.01)


def test_coverage_blockers_leave_only_shared_moments():
    p, main, cam2, angles, syncs = _project()
    blocks = multicam.coverage_blockers(p.footage, angles, syncs)
    assert [(b.src_in, b.src_out) for b in blocks] == [(0.0, 20.0), (80.0, 120.0)]
    assert multicam.coverage_blockers(p.footage, angles, {}) == []       # nothing synced: nothing blocked


def test_box_crop_geometry():
    pl = reframe.overlay_place(1920, 1080, 1080, 1920, 1.0, 1.0, 0.5)    # a 16:9 picture in half a 9:16 frame
    assert (pl.crop_w, pl.crop_h) == (1080, 960) and pl.scaled_h == 960
    x, _ = reframe.pan_for_point(0.2, 0.5, 1920, 1080, 1080, 1920, 1.0, 0.5)
    assert x < 0


def test_multicam_settings_saved(tmp_path):
    p, main, cam2, angles, syncs = _project()
    _store(p, angles, syncs, "stack")
    path = str(tmp_path / "m.kgproj")
    p.save(path)
    q = Project.load(path)
    assert q.settings.multicam["template"] == "stack" and q.footage[cam2.id].face == [0.3, 0.35, 0.2]
    assert {(o.angle, o.link) for o in q.overlays} == {(o.angle, o.link) for o in p.overlays}
    assert multicam.stored_syncs(q)[cam2.id][1].offset == 20.0
    assert not multicam.follow(q)                                     # already in step


def test_dialog_multicam_choices(qapp):
    from kangedit.autoedit import AutoEditSettings
    from kangedit.ui.dialogs import AutoEditDialog
    p, main, cam2, angles, syncs = _project()
    d = AutoEditDialog(p, [], AutoEditSettings())
    d.chk_multi.setChecked(True)
    d.angle_rows[cam2.id].setCurrentIndex(1)
    mc = d.multicam()
    assert mc["angles"] == {main.id: 1, cam2.id: 2} and d.footage_ids() == [main.id]
    assert d.cb_audio.findData("angle:2") >= 0 and not d.lbl_angles.text()
    d.angle_rows[cam2.id].setCurrentIndex(2)                      # "Angle 3" with no angle 2
    assert d.lbl_angles.text() and isinstance(OverlayClip("x", 0).angle, int)


def test_old_main_angle_layer_copies_are_dropped():
    p, main, cam2, angles, syncs = _project("9:16 (Vertical)")
    _store(p, angles, syncs, "stack")
    p.overlays.append(OverlayClip(main.id, 0.0, 30.0, 33.0, 0, angle=1, link=p.clips[0].id))  # an earlier version's
    assert multicam.follow(p) and not any(o.angle == 1 for o in p.overlays)


def test_new_main_clip_joins_the_boxed_layout():
    p, main, cam2, angles, syncs = _project("9:16 (Vertical)")
    _store(p, angles, syncs, "stack")
    p.clips.append(TimelineClip(main.id, 60.0, 62.0))
    multicam.follow(p)
    assert (p.clips[-1].box_y, p.clips[-1].box_h) == (0.25, 0.5)


# ----------------------------------------------------------------------------- linked editing in the window
@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1500, 950)
    yield w
    w.project.dirty = False
    w.close()


def _drag(view, a, b):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent

    def ev(kind, p, buttons):
        return QMouseEvent(kind, QPointF(p), QPointF(view.mapToGlobal(p)), Qt.LeftButton, buttons, Qt.NoModifier)
    view.mousePressEvent(ev(QEvent.MouseButtonPress, a, Qt.LeftButton))
    for k in range(1, 6):
        view.mouseMoveEvent(ev(QEvent.MouseMove, a + (b - a) * (k / 5), Qt.LeftButton))
    view.mouseReleaseEvent(ev(QEvent.MouseButtonRelease, b, Qt.NoButton))


def test_angle_clips_are_handles_on_their_group(window, qapp):
    from PySide6.QtCore import QPointF
    p, main, cam2, angles, syncs = _project()
    p.clips = p.clips[:2]
    _store(p, angles, syncs, "pip")
    window.set_project(p)
    window.show()
    qapp.processEvents()
    tl = window.tl
    tl.pps = 50.0
    tl.refresh()
    o = next(x for x in p.overlays if x.link == p.clips[0].id)
    r = next(rr for c, rr in tl._overlay_rects() if c.id == o.id)
    # body drag 1 s to the left slips the main clip 1 s later, and the angle follows
    _drag(tl, r.center(), r.center() - QPointF(50, 0))
    c = p.clips[0]
    assert c.src_in == pytest.approx(31.0) and c.duration == pytest.approx(3.0)
    o = next(x for x in p.overlays if x.link == c.id)
    assert o.src_in == pytest.approx(c.src_in - 20.0) and tl.selected_overlay == o.id
    # its left edge trims the whole group
    r = next(rr for cc, rr in tl._overlay_rects() if cc.id == o.id)
    _drag(tl, QPointF(r.left() + 2, r.center().y()), QPointF(r.left() + 27, r.center().y()))
    assert c.src_in == pytest.approx(31.5) and c.duration == pytest.approx(2.5)
    o = next(x for x in p.overlays if x.link == c.id)
    assert (o.src_in, o.duration) == (pytest.approx(11.5), pytest.approx(2.5))
    # Other moment (R) with the angle selected works on its main clip
    assert window._moment_clip() == c.id


def test_unlinked_angles_are_left_alone_and_relink_back_in_sync():
    p, main, cam2, angles, syncs = _project()
    _store(p, angles, syncs, "pip")
    o = next(x for x in p.overlays if x.link == p.clips[0].id)
    assert multicam.set_linked(p, [o], False) == 1
    o.start, o.src_in = 1.25, 2.0                               # moved on its own
    p.clips[0].src_in += 1.0
    multicam.follow(p)
    o = next(x for x in p.overlays if x.link == p.clips[0].id)
    assert (o.start, o.src_in, o.linked) == (1.25, 2.0, False)  # not re-timed, not duplicated
    assert len(multicam.angles_of(p, p.clips[0].id)) == 1
    multicam.set_linked(p, [o], True)                            # back in sync
    o = next(x for x in p.overlays if x.link == p.clips[0].id)
    assert o.linked and o.start == 0.0 and o.src_in == pytest.approx(p.clips[0].src_in - 20.0)


def test_link_button_and_unlinked_drag(window, qapp):
    from PySide6.QtCore import QPointF
    p, main, cam2, angles, syncs = _project()
    p.clips = p.clips[:2]
    _store(p, angles, syncs, "pip")
    window.set_project(p)
    window.show()
    qapp.processEvents()
    tl, b = window.tl, window.timeline.btn_link
    tl.selected = p.clips[0].id
    tl.selection_changed.emit(p.clips[0].id)
    assert b.isEnabled() and b.isChecked()
    b.click()                                                    # unlink the clip's angles
    o = next(x for x in p.overlays if x.link == p.clips[0].id)
    assert not o.linked and not b.isChecked() and window._moment_clip() == p.clips[0].id
    tl.pps = 50.0
    tl.refresh()
    r = next(rr for c, rr in tl._overlay_rects() if c.id == o.id)
    _drag(tl, r.center(), r.center() + QPointF(50, 0))           # an unlinked angle moves on its own
    o = next(x for x in p.overlays if x.id == o.id)
    assert o.start == pytest.approx(1.0) and p.clips[0].src_in == 30.0
    tl.select_overlay(o.id)
    window.toggle_link()                                          # Ctrl+L on the angle: back in sync
    o = next(x for x in p.overlays if x.link == p.clips[0].id)
    assert o.linked and o.start == 0.0 and b.isChecked()
    window.undo()
    assert not next(x for x in p.overlays if x.link == p.clips[0].id).linked


# ----------------------------------------------------------------------------- transitions and switching
def test_angles_take_part_in_a_dissolve():
    p, main, cam2, angles, syncs = _project()
    p.clips = [TimelineClip(main.id, 30.0, 33.0), TimelineClip(main.id, 40.0, 44.0, transition="dissolve",
                                                                transition_dur=1.0)]
    _store(p, angles, syncs, "stack")
    a, b = sorted(p.overlays, key=lambda o: o.start)
    cut = p.clips[0].duration
    # the outgoing angle runs on past the cut, the incoming one starts early and fades in over it
    assert a.end == pytest.approx(cut + 0.5) and a.mix_out == 0.0
    assert b.start == pytest.approx(cut - 0.5) and b.mix_in == pytest.approx(1.0)
    assert b.src_in == pytest.approx(40.0 - 20.0 - 0.5)            # still in sync with the main clip
    assert p.overlays.index(b) > p.overlays.index(a)               # drawn on top
    assert b.alpha_at(cut - 0.5) == 0.0 and b.alpha_at(cut) == pytest.approx(0.5) and b.alpha_at(cut + 0.6) == 1.0
    assert b.fade_in == 0.0                                        # the user's own fades are separate


def test_angles_dip_with_a_dip_transition_and_follow_changes():
    p, main, cam2, angles, syncs = _project()
    p.clips = [TimelineClip(main.id, 30.0, 33.0), TimelineClip(main.id, 40.0, 44.0, transition="dip_black",
                                                                transition_dur=1.0)]
    _store(p, angles, syncs, "pip")
    a, b = sorted(p.overlays, key=lambda o: o.start)
    assert (a.mix_out, b.mix_in) == (0.5, 0.5) and a.end == pytest.approx(3.0) and b.start == pytest.approx(3.0)
    p.clips[1].transition = ""                                     # removed: the angles cut again
    multicam.follow(p)
    a, b = sorted(p.overlays, key=lambda o: o.start)
    assert (a.mix_out, b.mix_in) == (0.0, 0.0) and b.start == pytest.approx(3.0)


def test_switching_choose_cuts_cleanly():
    step = multicam.SWITCH_STEP
    t = np.arange(0, 30, step)
    a = np.where((t // 5) % 2 == 0, 1.0, -1.0)                   # who speaks changes every 5 s
    noise = np.random.default_rng(1).normal(0, 0.6, len(t))
    path = multicam._choose(np.vstack([a + noise, -a - noise]), step)
    runs = np.diff(np.flatnonzero(np.diff(path)))
    assert len(np.flatnonzero(np.diff(path))) == 5                # one cut per change of speaker, no flicker
    assert runs.min() * step >= multicam.MIN_SHOT
    # an angle that is always slightly better still gives way now and then (no endless single shot)
    path = multicam._choose(np.vstack([np.full(400, 0.5), np.full(400, 0.4)]), step)
    assert path.min() == 0 and path.max() == 1
    # an angle that isn't recording can't be chosen
    sc = np.vstack([np.zeros(100), np.full(100, 2.0)])
    sc[1, :50] = np.nan
    path = multicam._choose(sc, step)
    assert (path[:50] == 0).all() and (path[60:] == 1).all()


class _An:
    """A stand-in Analysis: loud while its own speaker talks."""

    def __init__(self, loud):
        self.times = np.arange(0, 60, 0.5)
        self.audio = np.where(loud(self.times), -12.0, -40.0)
        z = np.zeros_like(self.times)
        self.faces, self.activity, self.sharp, self.jerk, self.clipped = z + 0.1, z, z, z, z
        self.thumbs, self.thumb_times, self.wave = np.zeros((0, 9, 16, 3), np.uint8), np.zeros(0), np.zeros(0)


def test_switching_layout_follows_who_speaks():
    p, main, cam2, angles, syncs = _project()
    an = {main.id: _An(lambda t: (t // 10) % 2 == 0), cam2.id: _An(lambda t: ((t + 20) // 10) % 2 == 1)}
    plan = multicam.switch_plan(p, an, angles, syncs)
    shots = plan[main.id]
    assert [k for *_, k in shots][:4] == [1, 2, 1, 2] and shots[1][0] == pytest.approx(20.0, abs=0.3)
    p.clips = [TimelineClip(main.id, 25.0, 45.0)]                   # 25-30 main, 30-40 cam2, 40-45 main
    p.settings.multicam = {"switch": plan}
    _store(p, angles, syncs, "switch")
    p.settings.multicam["switch"] = plan               # (as the auto edit stores it)
    ov = p.overlays
    assert len(ov) == 1 and (ov[0].angle, ov[0].track) == (2, 0)
    assert ov[0].start == pytest.approx(5.0, abs=0.3) and ov[0].duration == pytest.approx(10.0, abs=0.3)
    assert ov[0].box_w == 1.0 and ov[0].box_h == 1.0 and ov[0].volume == 0.0       # full frame, main sound
    p.clips[0].src_in = 35.0                                        # trimmed: the shots follow
    multicam.follow(p)
    assert len(p.overlays) == 1 and p.overlays[0].start == pytest.approx(0.0, abs=0.3)
    assert p.overlays[0].duration == pytest.approx(5.0, abs=0.3)
    # to a layout and back
    assert multicam.relayout(p, "side") == 1 and p.overlays[0].duration == pytest.approx(10.0)
    assert multicam.relayout(p, "switch") == 1 and p.overlays[0].duration == pytest.approx(5.0, abs=0.3)


def test_dialog_switching_keeps_the_main_sound(qapp):
    from kangedit.autoedit import AutoEditSettings
    from kangedit.ui.dialogs import AutoEditDialog
    p, main, cam2, angles, syncs = _project()
    d = AutoEditDialog(p, [], AutoEditSettings())
    d.chk_multi.setChecked(True)
    d.angle_rows[cam2.id].setCurrentIndex(1)
    d.cb_layout.setCurrentIndex(d.cb_layout.findData("switch"))
    assert not d.cb_audio.isEnabled() and "one camera" in d.lbl_layout.text()
    assert d.multicam()["template"] == "switch"
    d.cb_layout.setCurrentIndex(d.cb_layout.findData("pip"))
    assert d.cb_audio.isEnabled()


def test_switching_from_the_layout_menu(window, qapp, monkeypatch):
    import threading
    p, main, cam2, angles, syncs = _project()
    p.clips = [TimelineClip(main.id, 25.0, 45.0)]
    _store(p, angles, syncs, "pip")
    window.set_project(p)
    window.analyses = {main.id: _An(lambda t: (t // 10) % 2 == 0), cam2.id: _An(lambda t: ((t + 20) // 10) % 2 == 1)}
    monkeypatch.setattr(window.tasks, "submit",
                        lambda name, job, done, err=None: done(job(lambda *a: None, threading.Event())))
    window.multicam_layout("switch")
    mc = window.project.settings.multicam
    assert mc["template"] == "switch" and mc["switch"][main.id]
    ov = window.project.overlays
    assert [o.angle for o in ov] == [2] and ov[0].track == 0 and ov[0].box_w == 1.0
    assert "switching" in window.lbl_status.text()
    window.undo()
    assert window.project.settings.multicam["template"] == "pip"
