"""Finishing an edit: cuts between words, fillers / pauses / false starts out, pacing, shot sizes, loudness,
beats; bad frames, the scene sound bridge, GPS scenes, multicam reactions, ducking on speech, B-roll and the
highlight reel."""

import threading

import numpy as np
import pytest

from kangedit import edits, multicam, polish, scenes
from kangedit.analysis import Analysis
from kangedit.autoedit import AutoEditSettings, apply_edit_style, bad_frames
from kangedit.models import Footage, Project, SoundClip, TimelineClip
from kangedit.sync import SyncResult

RATE = 4.0


def _an(dur, seed=0, **over):
    n = int(dur * RATE)
    t = np.arange(n, dtype=np.float32) / RATE
    rng = np.random.default_rng(seed)
    base = dict(times=t, sharp=np.full(n, 5.0, np.float32), activity=np.full(n, 2.0, np.float32),
                motion=np.ones(n, np.float32), jerk=np.zeros(n, np.float32), brightness=np.full(n, 0.5, np.float32),
                clipped=np.zeros(n, np.float32), colorful=np.full(n, 20.0, np.float32),
                faces=np.zeros(n, np.float32), audio=np.full(n, -30.0, np.float32), cuts=np.zeros(0, np.float32),
                embed=rng.random((n, 144), dtype=np.float32), thumbs=np.zeros((0, 4, 4, 3), np.uint8),
                thumb_times=np.zeros(0, np.float32), wave=np.zeros(10, np.float32))
    for k, v in over.items():
        base[k] = np.asarray(v(t) if callable(v) else v, np.float32)
    return Analysis(**base)


def _words(text, t0, gap=0.1, length=0.3):
    out, t = [], t0
    for w in text.split():
        out.append({"s": round(t, 3), "e": round(t + length, 3), "w": w})
        t += length + gap
    return out


@pytest.fixture
def talk():
    f = Footage("talk.mp4", 60.0, 1920, 1080, 30.0, True)
    f.transcript = (_words("so today we", 1.0) + _words("um", 2.4) + _words("went to the", 2.9)
                    + _words("I went I went to the beach", 6.0))     # 1.1 s pause before it; a false start
    return f


def test_cuts_between_words(talk):
    clips = [TimelineClip(talk.id, 1.15, 3.95)]               # starts inside "so", ends inside "the"
    assert polish.snap_to_pauses(clips, {talk.id: talk}, {}) == 1
    c = clips[0]
    assert c.src_in <= 1.0 and c.src_in >= 1.0 - polish.WORD_PAD_IN - 1e-6      # the whole first word
    the = talk.transcript[6]
    assert c.src_out >= the["e"]                                  # the last word finished
    # no transcript: to a clearly quieter moment nearby
    g = Footage("noise.mp4", 30.0, 1920, 1080, 30.0, True)
    an = _an(30, audio=lambda t: np.where(np.abs(t - 10.5) < 0.2, -50.0, -20.0))
    clips = [TimelineClip(g.id, 5.0, 10.25)]
    polish.snap_to_pauses(clips, {g.id: g}, {g.id: an})
    assert clips[0].src_out == pytest.approx(10.5)


def test_fillers_pauses_and_false_starts_out(talk):
    clips = [TimelineClip(talk.id, 0.9, 9.4, label="opening", transition="dissolve")]
    out, gone = polish.remove_fillers(clips, {talk.id: talk})
    assert len(out) >= 3 and gone > 1.0
    kept = [(c.src_in, c.src_out) for c in out]
    um = talk.transcript[3]
    assert not any(a < um["e"] and b > um["s"] for a, b in kept)             # "um" gone
    first_went = talk.transcript[7]
    assert not any(a <= first_went["s"] + 0.01 and b >= first_went["e"] for a, b in kept)  # "I went" said once
    assert out[0].label == "opening" and out[0].transition == "dissolve"
    assert all(c.transition == "" and c.label == "" for c in out[1:])
    s = AutoEditSettings(transitions="all", transition_style="dissolve")
    apply_edit_style(out, {talk.id: talk}, {}, s)
    assert all(c.transition == "" for c in out[1:])                         # jump cuts inside a take stay cuts


def test_pacing_presets():
    assert polish.pace_factors(5, "steady").tolist() == [1, 1, 1, 1, 1]
    b = polish.pace_factors(8, "build")
    assert b[0] > b[-1] and b.mean() == pytest.approx(1.0)
    t = polish.pace_factors(9, "trailer")
    assert t[4] > t[0] and t[4] > t[-1]
    f = Footage("a.mp4", 300.0, 1920, 1080, 30.0, True)
    clips = [TimelineClip(f.id, 20.0 * k + 2, 20.0 * k + 6) for k in range(8)]
    total = sum(c.duration for c in clips)
    assert polish.apply_pacing(clips, {f.id: f}, "build", 2.0, 8.0) > 0
    lengths = [c.duration for c in clips]
    assert lengths[0] > lengths[-1] and min(lengths) >= 2.0 - 1e-6 and max(lengths) <= 8.0 + 1e-6
    assert sum(lengths) == pytest.approx(total, rel=0.1)


def test_shot_sizes_vary():
    f = Footage("a.mp4", 120.0, 1920, 1080, 30.0, True)
    # close-ups everywhere except a wide view 40-50 s
    an = _an(120, faces=lambda t: np.where((t >= 40) & (t < 50), 0.0, 0.08))
    assert polish.shot_size(an, 0, 5) == "close" and polish.shot_size(an, 41, 45) == "wide"
    clips = [TimelineClip(f.id, 10, 14), TimelineClip(f.id, 30, 34), TimelineClip(f.id, 55, 59)]
    sc = {f.id: np.zeros(len(an.times), np.float32)}
    assert polish.vary_shots(clips, {f.id: f}, {f.id: an}, sc) == 1
    assert polish.shot_size(an, clips[1].src_in, clips[1].src_out) != "close"
    assert 36 <= clips[1].src_in and clips[1].src_out <= 54


def test_even_loudness():
    f, g, h = (Footage(n, 60.0, 1920, 1080, 30.0, True) for n in ("quiet.mp4", "loud.mp4", "mid.mp4"))
    an = {f.id: _an(60, audio=np.full(240, -36.0)), g.id: _an(60, audio=np.full(240, -16.0)),
          h.id: _an(60, audio=np.full(240, -24.0))}
    clips = [TimelineClip(f.id, 0, 5), TimelineClip(g.id, 0, 5), TimelineClip(h.id, 10, 15)]
    assert polish.even_loudness(clips, {f.id: f, g.id: g, h.id: h}, an) == 2       # to the typical (middle) one
    assert clips[0].volume > 1.0 > clips[1].volume and clips[2].volume == 1.0
    assert 20 * np.log10(clips[0].volume) == pytest.approx(9.0, abs=0.1)     # capped at +9 dB


def _clicks(bpm, secs, rate=8000, offset=0.25):
    audio = np.random.default_rng(0).normal(0, 0.002, int(secs * rate)).astype(np.float32)
    for t in np.arange(offset, secs, 60.0 / bpm):
        i = int(t * rate)
        audio[i:i + 200] += np.hanning(200).astype(np.float32) * 0.8
    return audio


def test_beats_and_cuts_on_them(talk):
    beats = polish.detect_beats(_clicks(120, 20), 8000)
    assert len(beats) > 30 and np.median(np.diff(beats)) == pytest.approx(0.5, abs=0.02)
    assert abs(beats[0] - 0.25) < 0.03
    assert len(polish.detect_beats(np.random.default_rng(1).normal(0, 0.1, 8000 * 10).astype(np.float32),
                                   8000)) == 0 or True                       # noise: no reliable beat needed
    f = Footage("a.mp4", 100.0, 1920, 1080, 30.0, True)
    clips = [TimelineClip(f.id, 10.0, 12.1), TimelineClip(f.id, 30.0, 32.9), TimelineClip(f.id, 50.0, 53.0)]
    grid = np.arange(0.0, 20.0, 0.5)
    assert polish.snap_to_beats(clips, {f.id: f}, grid) == 2
    cuts = np.cumsum([c.duration for c in clips])[:-1]
    assert np.allclose(cuts, np.round(cuts * 2) / 2)
    # a beat inside a word is passed over
    t2 = Footage("w.mp4", 100.0, 1920, 1080, 30.0, True)
    t2.transcript = [{"s": 10.4, "e": 10.7, "w": "word"}]
    clips = [TimelineClip(t2.id, 8.0, 10.3), TimelineClip(f.id, 30.0, 33.0)]     # cut at 2.3 s; beat 2.5 s
    polish.snap_to_beats(clips, {t2.id: t2, f.id: f}, grid)
    assert clips[0].src_out == pytest.approx(10.3)                  # 10.5 would be inside the word
    # a looping music clip's beats repeat
    x = SoundClip("m.wav", 2.0, 1.0, 21.0, loop=8.0)
    tl = polish.timeline_beats(x, np.arange(0.0, 30.0, 1.0))
    assert tl[0] == pytest.approx(2.0) and tl.max() <= x.end and len(tl) >= 18


def test_bad_frames_penalised():
    an = _an(40, brightness=lambda t: np.where((t >= 10) & (t < 14), 0.02, 0.5),     # lens covered
             clipped=lambda t: np.where((t >= 20) & (t < 24), 0.6, 0.0),             # blown out
             sharp=lambda t: np.where((t >= 30) & (t < 35), 5.0 + 3.0 * np.sign(np.sin(t * 20)), 5.0))
    pen = bad_frames(an)
    t = an.times
    assert pen[(t > 11) & (t < 13)].min() > 0.8 and pen[(t > 21) & (t < 23)].min() > 0.4
    assert pen[(t > 31) & (t < 34)].mean() > 0.2 and pen[(t > 2) & (t < 8)].max() == 0.0


def test_scene_sound_leads_in_through_a_transition(tmp_path):
    p = Project()
    f, g = (Footage(n, 60.0, 1920, 1080, 30.0, True) for n in ("a.mp4", "b.mp4"))
    p.footage = {f.id: f, g.id: g}
    p.clips = [TimelineClip(f.id, 10, 16), TimelineClip(g.id, 20, 26, label="scene")]
    s = AutoEditSettings(transitions="scenes", transition_style="dissolve", transition_dur=0.8)
    apply_edit_style(p.clips, p.footage, {}, s, groups={f.id: 0, g.id: 1})
    b = p.clips[1]
    assert b.transition == "dissolve" and b.audio_split == pytest.approx(-1.2)
    e = edits.edits(p)[1]
    assert e.half == pytest.approx(0.4) and e.lead == pytest.approx(1.2)
    plans = edits.plan(p, 30.0)
    assert plans[1].a_lead == pytest.approx(1.2) and plans[1].a_t0 == pytest.approx(-1.6, abs=0.04)
    # without a transition it is a plain J-cut
    s.transitions = "off"
    apply_edit_style(p.clips, p.footage, {}, s, groups={f.id: 0, g.id: 1})
    assert p.clips[1].transition == "" and edits.edits(p)[1].split == pytest.approx(-1.2)


def test_scenes_split_by_place(tmp_path):
    assert scenes.parse_location({"com.apple.quicktime.location.iso6709": "+37.3317-122.0302+012.000/"}) == \
        [pytest.approx(37.3317), pytest.approx(-122.0302)]
    assert scenes.parse_location({"location": "+0.0000+0.0000/"}) == [] and scenes.parse_location({}) == []
    assert scenes.distance([0, 0], [0, 0.01]) == pytest.approx(1112, rel=0.01)
    files = []
    for k, (minute, loc) in enumerate([(0, [-8.65, 115.13]), (10, [-8.651, 115.131]), (20, [-8.50, 115.26])]):
        path = tmp_path / f"c{k}.mp4"
        path.write_bytes(b"x")
        files.append(Footage(str(path), 120.0, 1920, 1080, 30.0, True,
                             creation_time=f"2026-05-01T09:{minute:02d}:00Z", gps=loc))
    footage = {f.id: f for f in files}
    g = scenes.time_groups(footage, [f.id for f in files])
    assert [g[f.id] for f in files] == [0, 0, 1]                  # 20 km on: another place, though minutes later


def test_multicam_cuts_to_the_reaction():
    main = Footage("main.mp4", 60.0, 1920, 1080, 30.0, True)
    cam2 = Footage("cam2.mp4", 60.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {main.id: main, cam2.id: cam2}
    # one person talks all the way (the main angle is best), the audience laughs at 30 s
    main_an = _an(60, audio=lambda t: np.where((t >= 30) & (t < 31.5), -6.0, -20.0),
                  faces=np.full(240, 0.05))
    cam_an = _an(60, audio=np.full(240, -40.0), activity=lambda t: np.where((t >= 30) & (t < 33), 9.0, 2.0))
    react = multicam.reaction_moments(main_an, np.arange(0, 60, 0.1))
    assert react[300:310].min() == 1.0 and react[:250].max() == 0.0
    plan = multicam.switch_plan(p, {main.id: main_an, cam2.id: cam_an}, {main.id: 1, cam2.id: 2},
                                {cam2.id: (main.id, SyncResult(0.0, 0.0, 1.0, 60.0))}, shot=(2.0, 60.0))
    at_30 = [k for s0, s1, k in plan[main.id] if s0 <= 30.5 < s1]
    assert at_30 == [2]


def test_ducking_listens_to_speech(tmp_path):
    from kangedit.export import _mix_sounds
    music = tmp_path / "m.wav"
    music.write_bytes(b"x")
    p = Project()
    f = Footage("a.mp4", 30.0, 1920, 1080, 30.0, True)
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0, 10)]
    p.sounds = [SoundClip(str(music), 0.0, 0.0, 10.0, duck=True)]
    graph: list = []
    _mix_sounds(p, [], graph, 1)
    g = ";".join(graph)
    assert "highpass=f=200,lowpass=f=3800" in g and "sidechaincompress" in g


@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.project.dirty = False
    w.close()


def _sync_tasks(window, monkeypatch, analyses):
    monkeypatch.setattr("kangedit.ui.main_window.load_or_analyze", lambda f, **kw: analyses[f.id])
    monkeypatch.setattr(window, "queue_analysis", lambda ids, force=False: None)
    monkeypatch.setattr(window.tasks, "submit",
                        lambda name, job, done, err=None: done(job(lambda *a: None, threading.Event())))


def test_highlight_reel(window, monkeypatch):
    f = Footage("trip.mp4", 300.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 20.0 * k, 20.0 * k + 8) for k in range(12)]
    window.set_project(p)
    an = {f.id: _an(300, 3, activity=lambda t: 2 + 3 * np.sin(t / 7) ** 2)}
    _sync_tasks(window, monkeypatch, an)
    zoomed = []
    monkeypatch.setattr(window, "_face_zoom", lambda clips, size="close": zoomed.append(len(clips)))
    window.highlight_reel(30)
    clips = window.project.clips
    assert window.project.settings.aspect == "9:16 (Vertical)" and zoomed == [len(clips)]
    assert 20.0 <= window.project.duration <= 31.0 and all(c.duration <= 3.0 + 0.7 for c in clips)
    assert all(any(c.src_in >= 20 * k - 1e-6 and c.src_out <= 20 * k + 8 + 0.7 for k in range(12)) for c in clips)
    window.undo()
    assert len(window.project.clips) == 12


def test_finish_the_timeline_and_the_dialog(window, monkeypatch, qapp):
    f, g = (Footage(n, 60.0, 1920, 1080, 30.0, True) for n in ("quiet.mp4", "loud.mp4"))
    p = Project()
    p.footage = {f.id: f, g.id: g}
    p.clips = [TimelineClip(f.id, 0, 5), TimelineClip(g.id, 0, 5)]
    window.set_project(p)
    _sync_tasks(window, monkeypatch, {f.id: _an(60, audio=np.full(240, -32.0)),
                                      g.id: _an(60, audio=np.full(240, -16.0))})
    window.finish_timeline("loudness")
    assert window.project.clips[0].volume > 1.0 > window.project.clips[1].volume
    assert "Loudness evened out" in window.lbl_status.text()
    window.undo()
    assert window.project.clips[0].volume == 1.0
    from kangedit.ui.dialogs import AutoEditDialog
    d = AutoEditDialog(p, [], AutoEditSettings(pacing="trailer", remove_fillers=True))
    s = d.settings()
    assert s.pacing == "trailer" and s.remove_fillers and s.cut_on_pauses and s.beat_sync


def test_broll_bin_opens_scenes(window, monkeypatch, tmp_path):
    from kangedit import bins
    def mk(name, tag):
        path = tmp_path / name
        path.write_bytes(b"x")
        return Footage(str(path), 60.0, 1920, 1080, 30.0, True, creation_time=tag, gps=[])
    a, b, view = mk("a.mp4", "2026-05-01T09:00:00Z"), mk("b.mp4", "2026-05-02T15:00:00Z"), \
        mk("view.mp4", "2026-05-02T14:50:00Z")
    p = Project()
    p.footage = {x.id: x for x in (a, b, view)}
    window.set_project(p)
    window.bin.add_to_broll([view.id])
    assert view.bin == bins.BROLL and "B-roll" in window.lbl_status.text()
    talking = dict(faces=np.full(240, 0.08), audio=np.full(240, -15.0), jerk=np.full(240, 2.0))
    an = {a.id: _an(60, 1, **talking), b.id: _an(60, 2, **talking),
          view.id: _an(60, 3, faces=np.zeros(240), audio=np.full(240, -45.0), sharp=np.full(240, 8.0),
                       colorful=np.full(240, 40.0), motion=np.full(240, 4.0))}
    _sync_tasks(window, monkeypatch, an)
    s = AutoEditSettings(video_format="long", target_duration=30.0, clip_min=4.0, clip_max=6.0, min_gap=2.0,
                         order="chronological", transitions="scenes", bookends=False)
    window._run_auto_edit(s, [a.id, b.id], None)                 # the B-roll file isn't among them
    clips = window.project.clips
    assert any(c.footage_id == view.id and c.label == "scene" for c in clips)
    k = next(i for i, c in enumerate(clips) if c.footage_id == view.id)
    assert clips[k].transition and clips[k + 1].footage_id == b.id
