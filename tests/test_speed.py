"""Slow / fast motion and speed ramps: the time map, the plan, the export graph, multicam and the UI."""

import os
import subprocess

import numpy as np
import pytest

from kangedit import multicam, speed
from kangedit.edits import plan
from kangedit.export import ExportSettings, _audio_chain, build_export, export
from kangedit.ffmpeg import FFmpegError, ffmpeg_bin, probe, run
from kangedit.models import Footage, OverlayClip, Project, TimelineClip
from kangedit.sync import SyncResult


# ----------------------------------------------------------------------------- time map
def test_constant_speed_changes_the_length_not_the_source():
    c = TimelineClip("f", 10.0, 14.0, speed=0.5)
    assert c.src_dur == 4.0 and c.duration == pytest.approx(8.0)
    assert c.src_at(2.0) == pytest.approx(11.0) and c.time_at(12.0) == pytest.approx(4.0)
    assert TimelineClip("f", 10.0, 14.0, speed=4.0).duration == pytest.approx(1.0)
    assert TimelineClip("f", 0.0, 1.0, speed=100).speed == 100      # stored as given, clamped when used
    assert TimelineClip("f", 0.0, 1.0, speed=100).duration == pytest.approx(1.0 / speed.MAX_SPEED)


@pytest.mark.parametrize("ramp", sorted(speed.RAMPS))
def test_ramps_cover_the_whole_source_and_invert(ramp):
    c = TimelineClip("f", 2.0, 8.0, ramp=ramp)
    tm = c.timemap()
    t = np.linspace(0.0, c.duration, 200)
    s = np.asarray(tm.src(t))
    assert s[0] == pytest.approx(0.0) and s[-1] == pytest.approx(6.0, abs=1e-6)
    assert np.all(np.diff(s) > 0)                                   # always moves forward
    assert np.allclose(tm.time(s), t, atol=1e-6)                    # time() inverts src()
    keys = speed.RAMPS[ramp][2]
    assert keys[0][0] == 0.0 and keys[-1][0] == 1.0
    # pieces for the sound cover the span in order, at speeds within range
    segs = tm.segments(0.0, c.duration)
    assert segs[0][0] == pytest.approx(0.0) and segs[-1][1] == pytest.approx(6.0)
    assert all(a[1] == pytest.approx(b[0]) for a, b in zip(segs, segs[1:]))
    assert all(speed.MIN_SPEED <= v <= speed.MAX_SPEED for _, _, v in segs)


def test_hero_ramp_is_slow_in_the_middle_and_fast_at_the_ends():
    tm = speed.TimeMap(10.0, 1.0, "hero")
    assert tm.v_in == pytest.approx(2.0) and tm.v_out == pytest.approx(2.0)
    mid = tm.time(5.2)
    rate_mid = (tm.src(mid + 0.1) - tm.src(mid - 0.1)) / 0.2
    assert rate_mid == pytest.approx(0.3, abs=0.02)
    assert tm.duration > 10.0 / 2.0                               # the slow part makes it longer than 2x
    # outside the clip (transition handles) it carries on at the edge speed
    assert tm.src(-0.5) == pytest.approx(-1.0) and tm.src(tm.duration + 1.0) == pytest.approx(12.0)


def test_setpts_expression_follows_the_map():
    tm = speed.TimeMap(4.0, 1.0, "bullet")
    expr = tm.setpts(0.0)
    assert expr.endswith("/TB") and expr.count("if(") == 48
    e = expr[:-3].replace("if(", "_if(").replace("lt(", "_lt(")
    for s in (0.0, 0.7, 1.75, 2.2, 3.9, 4.5):
        got = eval(e.replace("T", repr(s)), {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: a < b})
        assert got == pytest.approx(tm.time(s), abs=0.01)
    assert speed.TimeMap(4.0, 2.0, "").setpts(0.0) == "(PTS-STARTPTS)/2.000000"


def test_atempo_chains_stay_in_range():
    assert speed.atempo(1.0) == ""
    assert speed.atempo(0.25) == "atempo=0.500000,atempo=0.500000"
    parts = [float(x.split("=")[1]) for x in speed.atempo(6.0).split(",")]
    assert np.prod(parts) == pytest.approx(6.0) and all(0.5 <= p <= 2.0 for p in parts)
    parts = [float(x.split("=")[1]) for x in speed.atempo(0.12).split(",")]
    assert np.prod(parts) == pytest.approx(0.12) and all(0.5 <= p <= 2.0 for p in parts)


def test_labels():
    assert speed.label(1.0, "") == "" and speed.label(0.5, "") == "0.5×"
    assert speed.label(1.0, "hero") == "Hero moment" and speed.label(2.0, "pulse") == "Pulse 2×"
    assert speed.is_normal(1.0, "") and not speed.is_normal(1.0, "hero")


def test_speed_is_saved_and_undone(tmp_path):
    p = Project()
    f = Footage("a.mp4", 30.0, 1920, 1080, 30.0, True)
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 4.0, speed=0.5, ramp="hero")]
    p.overlays = [OverlayClip(f.id, 0.0, 0.0, 2.0, 0, speed=2.0, ramp="pulse")]
    path = str(tmp_path / "x.kgproj")
    p.save(path)
    q = Project.load(path)
    assert (q.clips[0].speed, q.clips[0].ramp) == (0.5, "hero")
    assert (q.overlays[0].speed, q.overlays[0].ramp) == (2.0, "pulse")
    assert q.duration == pytest.approx(p.clips[0].duration)


# ----------------------------------------------------------------------------- plan and export graph
def _project(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    f = Footage(str(video), 30.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 2.0, 4.0, speed=0.5),
               TimelineClip(f.id, 10.0, 16.0, ramp="hero", transition="dissolve", transition_dur=0.6),
               TimelineClip(f.id, 20.0, 22.0, speed=2.0, audio_split=-0.4)]
    return p, f


def test_plan_uses_timeline_lengths_and_source_handles(tmp_path):
    p, f = _project(tmp_path)
    fps = p.settings.fps
    plans = plan(p, fps)
    assert [cp.nfr for cp in plans] == [round(4.0 * fps), round(p.clips[1].duration * fps), round(1.0 * fps)]
    b = plans[1]
    assert b.pre_f > 0 and b.v_in < 10.0                      # handle before the in point, at the ramp's start speed
    assert b.v_end > 16.0                                   # and after it, at its end speed
    c = plans[2]                                            # J-cut at 2x: its sound starts 0.4 s (0.8 s source) early
    assert c.a_in == pytest.approx(20.0 - 0.8, abs=0.05) and c.a_t0 == pytest.approx(-0.4, abs=0.04)


def test_export_graph_speeds_picture_and_sound(tmp_path):
    p, f = _project(tmp_path)
    p.overlays = [OverlayClip(f.id, 1.0, 0.0, 2.0, 0, speed=0.5, volume=0.5)]
    build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    g = (tmp_path / "graph.txt").read_text(encoding="utf-8")
    assert "setpts='(PTS-STARTPTS)/0.500000'" in g and "setpts='(PTS-STARTPTS)/2.000000'" in g
    assert "setpts='(if(lt(T," in g                         # the ramp
    assert "framerate=fps=" in g                            # slow motion blends frames
    assert "atempo=0.500000" in g and "atempo=2.000000" in g
    assert "concat=n=" in g and ":v=0:a=1" in g              # the ramp's sound, in pieces
    build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False,
                                   blend_slowmo=False), str(tmp_path))
    assert "framerate=" not in (tmp_path / "graph.txt").read_text(encoding="utf-8")


def test_normal_speed_graph_is_unchanged(tmp_path):
    p, f = _project(tmp_path)
    for c in p.clips:
        c.speed, c.ramp = 1.0, ""
    build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    g = (tmp_path / "graph.txt").read_text(encoding="utf-8")
    assert "atempo" not in g and "framerate=" not in g and "setpts='" not in g


def test_audio_chain_single_and_ramp_segments():
    one = _audio_chain("0:a:0", "a0", 2.0, 1.0, 0.0, [(5.0, 6.0, 0.5)], 4.0)
    assert one.startswith("[0:a:0]atrim=start=1.0000") and "atempo=0.500000" in one
    one = _audio_chain("0:a:0", "a0", 2.0, 1.0, 0.0, [(4.0, 4.5, 0.5), (4.5, 6.0, 1.5)], 4.0)
    assert "atempo@a0t0=tempo=0.500000" in one and "atempo@a0t1" not in one     # 0.5..2x: one stage
    segs = [(4.0, 4.5, 0.3), (4.5, 6.0, 1.5)]
    ramp = _audio_chain("0:a:0", "a0", 2.0, 1.0, 0.0, segs, 4.0)              # pitch kept: one stretch
    assert "atempo@a0t0=tempo=0.547723" in ramp and "atempo@a0t1=tempo=0.547723" in ramp   # 2 stages
    # stage 1 switches at 0.5 s of source; stage 2 sees that moment 0.5 / 0.548 s into its input
    assert "asendcmd=c='0.5000 atempo@a0t0 tempo 1.224745'" in ramp
    assert "asendcmd=c='0.9129 atempo@a0t1 tempo 1.224745'" in ramp and ramp.endswith("[a0]")
    tape = _audio_chain("0:a:0", "a0", 2.0, 1.0, 0.0, segs, 4.0, keep_pitch=False)   # pitch follows: pieces
    assert "asplit=2[a0s0][a0s1]" in tape and "[a0p0][a0p1]concat=n=2:v=0:a=1[a0r]" in tape
    assert "asetrate=14400,aresample=48000" in tape and "asetrate=72000" in tape


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_real_export_shows_the_right_frames(tmp_path):
    """Export slow, fast and ramped clips of a clip whose every frame differs, then find each exported frame
    in the source: it is the one the time map says (within a frame), and the length is frame exact."""
    src = str(tmp_path / "src.mp4")
    try:
        run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=4",
             "-f", "lavfi", "-i", "sine=frequency=330:duration=4", "-shortest", "-pix_fmt", "yuv420p", "-y", src])
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"FFmpeg can't make the test clip: {e}")
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 1.0, speed=0.5), TimelineClip(f.id, 1.0, 3.0, speed=2.0),
               TimelineClip(f.id, 0.5, 3.5, ramp="hero")]
    out = str(tmp_path / "out.mp4")
    try:
        export(p, ExportSettings(out, p.settings.aspect, "720p", 30.0, use_gpu=False, blend_slowmo=False))
    except FFmpegError as e:   # an old system FFmpeg without the features the export uses
        pytest.skip(f"export needs a newer FFmpeg: {e}")

    def frames(path):
        raw = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", path, "-vf", "scale=64:36", "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, 36, 64).astype(np.float32)

    s, o = frames(src), frames(out)
    assert len(o) == sum(round(c.duration * 30) for c in p.clips)
    starts = p.clip_starts()
    errs = []
    for ci, c in enumerate(p.clips):
        for k in range(2, int(c.duration * 30) - 2, 4):
            t = starts[ci] + k / 30
            best = int(np.argmin([np.mean(np.abs(o[int(round(t * 30))] - x)) for x in s])) / 30
            errs.append(abs(best - c.src_at(t - starts[ci])))
    assert max(errs) <= 1.5 / 30
    assert os.path.getsize(out) > 0 and probe(out).has_audio
    # Blended slow motion into a transition, a J-cut and ramped layers: still frame exact.
    p.clips[1].transition, p.clips[2].transition, p.clips[2].audio_split = "dissolve", "dissolve", -0.3
    p.overlays = [OverlayClip(f.id, 0.5, 0.0, 2.0, 0, speed=0.5, scale=0.3), OverlayClip(f.id, 2.0, 0.5, 3.5, 1,
                                                                                      ramp="pulse", scale=0.3)]
    export(p, ExportSettings(out, p.settings.aspect, "720p", 30.0, use_gpu=False, blend_slowmo=True))
    assert len(frames(out)) == sum(round(c.duration * 30) for c in p.clips)


# ----------------------------------------------------------------------------- multicam
def _multicam():
    main = Footage("main.mp4", 120.0, 1920, 1080, 30.0, True)
    cam2 = Footage("cam2.mp4", 60.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {main.id: main, cam2.id: cam2}
    p.clips = [TimelineClip(main.id, 30.0, 33.0, speed=0.5), TimelineClip(main.id, 50.0, 54.0, ramp="hero")]
    angles = {main.id: 1, cam2.id: 2}
    syncs = {cam2.id: (main.id, SyncResult(20.0, 0.0, 1.0, 60.0))}
    p.overlays = multicam.build(p, p.clips, angles, syncs, "pip")
    p.settings.multicam = {"angles": angles, "template": "pip", "audio": "main",
                           "syncs": {f: [ref, r.offset, r.drift, r.confidence, r.overlap] for f, (ref, r) in syncs.items()},
                           "clips": [c.id for c in p.clips]}
    return p


def test_camera_angles_play_at_the_main_clips_speed():
    p = _multicam()
    ov = sorted(p.overlays, key=lambda o: o.start)
    starts = p.clip_starts()
    for o, c, t0 in zip(ov, p.clips, starts):
        assert (o.speed, o.ramp) == (c.speed, c.ramp)
        assert o.start == pytest.approx(t0) and o.duration == pytest.approx(c.duration, abs=1e-6)
        for x in (0.3, c.duration / 2, c.duration - 0.2):          # same moment all the way through
            assert o.src_at(x) + 20.0 == pytest.approx(c.src_at(x), abs=1e-6)
    p.clips[0].speed = 2.0
    assert multicam.follow(p)
    o = next(x for x in p.overlays if x.link == p.clips[0].id)
    assert o.speed == 2.0 and o.duration == pytest.approx(1.5)
    assert next(x for x in p.overlays if x.link == p.clips[1].id).start == pytest.approx(1.5)


def test_angle_covering_part_of_a_ramp_stays_in_step_at_both_ends():
    main = Footage("main.mp4", 120.0, 1920, 1080, 30.0, True)
    cam2 = Footage("cam2.mp4", 60.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {main.id: main, cam2.id: cam2}
    p.clips = [TimelineClip(main.id, 16.0, 26.0, ramp="hero")]        # cam2 starts at main's 20 s
    syncs = {cam2.id: (main.id, SyncResult(20.0, 0.0, 1.0, 60.0))}
    o = multicam.build(p, p.clips, {main.id: 1, cam2.id: 2}, syncs, "pip")[0]
    c = p.clips[0]
    assert o.ramp == "" and o.src_in == pytest.approx(0.0)
    assert o.start == pytest.approx(c.time_at(20.0), abs=1e-3) and o.end == pytest.approx(c.duration, abs=1e-3)


# ----------------------------------------------------------------------------- UI
@pytest.fixture
def window(qapp, tmp_path):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    p, f = _project(tmp_path)
    for c in p.clips:
        c.speed, c.ramp = 1.0, ""
    w.set_project(p)
    yield w
    w.project.dirty = False
    w.close()


def test_speed_menu_actions_and_undo(window):
    p = window.project
    a, b = p.clips[0], p.clips[1]
    window.clip_action("speed:0.5", a.id)
    assert a.speed == 0.5 and a.duration == pytest.approx(4.0)
    window.clip_action("ramp:bullet", b.id)
    assert b.ramp == "bullet" and b.duration > 6.0
    assert "Bullet time" in window.lbl_status.text()
    window.tl.select_all()
    window.clip_action("speed:2", a.id)                      # the whole selection
    assert all(c.speed == 2.0 for c in p.clips)
    window.undo()
    p = window.project
    assert p.clips[0].speed == 0.5 and p.clips[2].speed == 1.0


def test_split_keeps_the_picture_in_place(window):
    p = window.project
    a = p.clips[0]
    window.clip_action("speed:0.5", a.id)
    window.viewer.seek(1.0)                                  # 1 s into a half-speed clip = 0.5 s of source
    window.split_at_playhead()
    assert p.clips[0].src_out == pytest.approx(2.5, abs=1 / 30) and p.clips[1].src_in == p.clips[0].src_out
    assert p.clips[0].duration + p.clips[1].duration == pytest.approx(4.0)


def test_inspector_and_layer_panel_speed(window):
    p = window.project
    window.tl.selected = p.clips[0].id
    window.inspector.set_clip(p.clips[0])
    window.inspector.sp_speed.set_value(4.0, emit=True)
    assert p.clips[0].speed == 4.0 and p.clips[0].duration == pytest.approx(0.5)
    window.inspector.cb_ramp.setCurrentIndex(window.inspector.cb_ramp.findData("montage"))
    assert p.clips[0].ramp == "montage"
    f = next(iter(p.footage.values()))
    window.add_overlays([f.id], 0.0)
    o = p.overlays[-1]
    window.tl.select_overlay(o.id)
    lp = window.layer_panel
    lp.sl_speed.set_value(0.5, emit=True)
    assert o.speed == 0.5 and lp.sp_len.value() == pytest.approx(o.duration, abs=0.01)
    lp.sp_len.setValue(3.0)                                   # length on the timeline, at its speed
    assert o.duration == pytest.approx(3.0, abs=0.01) and o.src_dur == pytest.approx(1.5, abs=0.01)
    window._timeline_action("ov_ramp:pulse", o.id)
    assert o.ramp == "pulse"


def test_timeline_menu_lists_speeds_and_ramps(window):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    tl = window.tl
    seen = []

    def read_and_close():
        menu = QApplication.activePopupWidget()

        def walk(m):
            for act in m.actions():
                seen.append(act.text())
                if act.menu():
                    walk(act.menu())
        if menu is not None:
            walk(menu)
            menu.close()
    QTimer.singleShot(50, read_and_close)
    tl._context_menu(tl.mapToGlobal(tl.rect().center()), window.project.clips[0].id)
    assert "0.5× slow motion" in seen and "2× fast motion" in seen and "Custom speed…" in seen
    assert all(v[0] in seen for v in speed.RAMPS.values())
    assert "Split at playhead\tS" in seen and seen.count("Audio edit (J / L cut)") == 1


def test_viewer_shows_the_frame_the_speed_map_says(window):
    p = window.project
    window.clip_action("speed:0.5", p.clips[0].id)
    window.clip_action("ramp:hero", p.clips[1].id)
    shots = window.viewer._shots()
    starts = p.clip_starts()
    for sh, c, t0 in zip(shots, p.clips, starts):
        assert sh.start == pytest.approx(t0) and sh.dur == pytest.approx(c.duration)
        for off in (-0.2, 0.0, c.duration / 3, c.duration):
            assert sh.src_at(off) == pytest.approx(c.src_at(off))


# ----------------------------------------------------------------------------- the sound at another speed
def _chirp_project(tmp_path):
    """A 4 s clip whose tone rises 100 Hz a second from 200 Hz: its pitch tells which moment is heard."""
    src = str(tmp_path / "chirp.mp4")
    try:
        run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30:duration=4",
             "-f", "lavfi", "-i", "aevalsrc='0.5*sin(2*PI*(200*t+50*t*t))':s=48000:d=4", "-shortest",
             "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-y", src.replace(".mp4", ".mov")])
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"FFmpeg can't make the test clip: {e}")
    src = src.replace(".mp4", ".mov")
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    p = Project()
    p.footage = {f.id: f}
    return p, f


def _pitch(wav: str, t: float) -> float:
    import wave
    with wave.open(wav) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        w.setpos(int(t * rate))
        x = np.frombuffer(w.readframes(int(0.1 * rate)), dtype="<i2").reshape(-1, ch)[:, 0].astype(np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 1 << 16))
    return float(np.argmax(spec) * rate / (1 << 16))


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
@pytest.mark.parametrize("mode", ["sync", "pitch", "normal", "mute"])
def test_sound_modes_at_half_speed(tmp_path, mode):
    from kangedit.export import render_audio
    p, f = _chirp_project(tmp_path)
    c = TimelineClip(f.id, 1.0, 3.0, speed=0.5, audio_mode=mode)       # 4 s on the timeline
    p.clips = [c]
    try:
        wav = render_audio(p, False, str(tmp_path / "mix.wav"))
    except FFmpegError as e:
        pytest.skip(f"needs a newer FFmpeg: {e}")
    import wave
    with wave.open(wav) as w:
        assert w.getnframes() / w.getframerate() == pytest.approx(4.0, abs=0.05)
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if mode == "mute":
        assert np.abs(raw).max() < 50
        return
    for t in (0.5, 1.5, 2.5, 2.85):                # (at normal speed the file ends 3 s in)
        heard = c.sound_src_at(t + 0.05)           # source moment heard there (middle of the measured 0.1 s)
        want = 200 + 100 * heard
        if mode == "pitch":
            want *= c.speed                        # half speed: an octave down
        assert _pitch(wav, t) == pytest.approx(want, rel=0.04), (mode, t)


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_ramped_sound_follows_the_picture(tmp_path):
    from kangedit.export import render_audio
    p, f = _chirp_project(tmp_path)
    c = TimelineClip(f.id, 0.5, 3.5, ramp="hero")
    p.clips = [c]
    try:
        wav = render_audio(p, False, str(tmp_path / "mix.wav"))
    except FFmpegError as e:
        pytest.skip(f"needs a newer FFmpeg: {e}")
    # within ~0.05 s of the picture all along (restarting the time-stretch per piece drifted ~0.6 s)
    for t in np.linspace(0.2, c.duration - 0.25, 12):
        assert _pitch(wav, t) == pytest.approx(200 + 100 * c.src_at(t + 0.05), rel=0.03), t


def test_sound_mode_in_graph_captions_and_menus(tmp_path):
    p, f = _project(tmp_path)
    p.clips[0].audio_mode = "pitch"
    p.clips[2].audio_mode = "normal"
    build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0, use_gpu=False),
                 str(tmp_path))
    g = (tmp_path / "graph.txt").read_text(encoding="utf-8")
    assert "asetrate=24000,aresample=48000" in g             # half speed, pitch an octave down
    a2 = next(x for x in g.split(";") if x.endswith("[a2]"))
    assert "atempo" not in a2 and "asetrate" not in a2       # the 2x clip's sound plays at normal speed
    assert "asendcmd=c='" in g and "atempo@a1t0=tempo=" in g   # the ramp: one continuous time-stretch
    # captions: a word 1 s after the in point is heard 1 s in at normal speed, 2 s in synced at half speed
    from kangedit.captions import build_captions
    f.transcript = [{"w": "hello", "s": 3.0, "e": 3.4}]
    q = Project()
    q.footage = {f.id: f}
    q.clips = [TimelineClip(f.id, 2.0, 6.0, speed=0.5)]
    assert build_captions(q)[0].start == pytest.approx(2.0, abs=0.01)
    q.clips[0].audio_mode = "normal"
    assert build_captions(q)[0].start == pytest.approx(1.0, abs=0.01)
    q.clips[0].audio_mode = "mute"
    assert build_captions(q) == []
    assert speed.stretch(0.5, False) == "asetrate=24000,aresample=48000" and speed.stretch(1.0, False) == ""


def test_sound_mode_controls(window):
    p = window.project
    a = p.clips[0]
    window.clip_action("speed:0.5", a.id)
    window.clip_action("audio:normal", a.id)
    assert a.audio_mode == "normal" and "Not synced" in window.lbl_status.text()
    window.tl.selected = a.id
    window.inspector.set_clip(a)
    assert window.inspector.cb_sound.currentData() == "normal"
    window.inspector.cb_sound.setCurrentIndex(window.inspector.cb_sound.findData("pitch"))
    assert a.audio_mode == "pitch"
    window.undo()
    assert window.project.clips[0].audio_mode == "normal"
    f = next(iter(p.footage.values()))
    window.add_overlays([f.id], 0.0)
    o = window.project.overlays[-1]
    window._timeline_action("ov_audio:mute", o.id)
    assert o.audio_mode == "mute"
    window.tl.select_overlay(o.id)
    assert window.layer_panel.cb_sound.currentData() == "mute"
    window.layer_panel.cb_sound.setCurrentIndex(window.layer_panel.cb_sound.findData("sync"))
    assert o.audio_mode == "sync"
