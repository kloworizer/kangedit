"""Smarter analysis (what shots show, what is heard), the quick first look and the full analysis after it, and
the faster, more accurate captions: word times on the sound, transcripts kept for every project, footage
without speech skipped, the speech model chosen for the computer."""

import os
import subprocess
import threading

import cv2
import numpy as np
import pytest

from kangedit import analysis as an
from kangedit import captions, smart
from kangedit.analysis import Analysis
from kangedit.autoedit import AutoEditSettings, score_samples, select
from kangedit.ffmpeg import ffmpeg_bin, probe
from kangedit.models import Footage, Project

needs_models = pytest.mark.skipif(not smart.available(), reason="needs onnxruntime and the bundled models")


def _scene(seed, dx=0, dark=1.0):
    rng = np.random.default_rng(seed)
    img = np.zeros((360, 640, 3), np.uint8)
    img[:] = rng.integers(0, 255, 3)
    for _ in range(12):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(img, (int(rng.integers(0, 640)) + dx, int(rng.integers(0, 360))), int(rng.integers(20, 90)), c, -1)
        cv2.rectangle(img, (int(rng.integers(0, 640)) + dx, int(rng.integers(0, 360))),
                      (int(rng.integers(0, 640)) + dx, int(rng.integers(0, 360))), c, 3)
    return np.clip(img * dark, 0, 255).astype(np.uint8)


@needs_models
def test_the_same_view_is_found_again():
    e = smart.describe([_scene(1), _scene(1, dx=25), _scene(1, dark=0.6), _scene(2), _scene(3)]).astype(np.float32)
    sim = e @ e.T
    assert e.shape == (5, smart.EMBED_DIM) and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-2)
    assert min(sim[0, 1], sim[0, 2]) > max(sim[0, 3], sim[0, 4]) + 0.05     # moved / darker: still the same view


@needs_models
def test_sounds_are_heard():
    t = np.arange(16000 * 3) / 16000
    quiet = smart.hear(np.zeros_like(t, dtype=np.float32))
    assert quiet.shape == (6, len(smart.GROUP_NAMES)) and quiet.dtype == np.uint8
    assert smart.group(quiet, "silence").mean() > 0.6 and smart.group(quiet, "speech").max() < 0.1
    a = Analysis(**{**_plain(3).__dict__, "sound": quiet})
    assert smart.at_samples(a.sound, "silence", a.times).shape == a.times.shape
    assert smart.at_samples(np.zeros((0, len(smart.GROUP_NAMES)), np.uint8), "speech", a.times) is None


def _plain(dur, seed=0, sem=None, sound=None):
    n = int(dur * 4)
    t = np.arange(n, dtype=np.float32) / 4
    rng = np.random.default_rng(seed)
    m = int(np.ceil(dur))
    kw = {}
    if sem is not None:
        kw["sem"] = sem
    if sound is not None:
        kw["sound"] = sound
    return Analysis(times=t, sharp=np.full(n, 5.0, np.float32), activity=rng.random(n, dtype=np.float32) * 0.1 + 0.5,
                    motion=np.ones(n, np.float32), jerk=np.zeros(n, np.float32), brightness=np.full(n, 0.5, np.float32),
                    clipped=np.zeros(n, np.float32), colorful=np.full(n, 20.0, np.float32),
                    faces=np.zeros(n, np.float32), audio=np.full(n, -25.0, np.float32), cuts=np.zeros(0, np.float32),
                    embed=rng.random((n, 144), dtype=np.float32), thumbs=np.zeros((m, 9, 16, 3), np.uint8),
                    thumb_times=np.arange(m, dtype=np.float32), wave=np.zeros(10, np.float32), **kw)


def test_old_caches_load_and_get_the_smarter_analysis(tmp_path, monkeypatch):
    a = _plain(10)
    path = str(tmp_path / "a.npz")
    np.savez_compressed(path, version=an.CACHE_VERSION,
                        **{k: getattr(a, k) for k in ("times", "sharp", "activity", "motion", "jerk", "brightness",
                                                      "clipped", "colorful", "faces", "audio", "cuts", "embed",
                                                      "thumbs", "thumb_times", "wave")})
    old = Analysis.load(path)                                   # made before the smarter analysis existed
    assert old is not None and old.sem.shape == (0, smart.EMBED_DIM) and not old.quick and old.sem_at([1.0]) is None
    f = Footage(str(tmp_path / "a.mp4"), 10.0, 1920, 1080, 30.0, False)
    monkeypatch.setattr(smart, "enabled", lambda: True)
    monkeypatch.setattr(smart, "describe", lambda imgs: np.ones((len(imgs), smart.EMBED_DIM), np.float16))
    assert an.needs_smart(f, old)
    assert an.make_smart(f, old) and old.sem.shape == (10, smart.EMBED_DIM)       # from its thumbnails
    assert not an.needs_smart(f, old) and old.sem_at([2.2, 7.9]).shape == (2, smart.EMBED_DIM)
    monkeypatch.setattr(smart, "enabled", lambda: False)                           # the lighter analysis
    assert not an.needs_smart(f, _plain(10))


def _same(dur, vec):
    m = int(np.ceil(dur))
    return np.tile(vec.astype(np.float16), (m, 1))


def test_the_same_view_from_another_file_is_avoided():
    rng = np.random.default_rng(3)
    view = rng.normal(size=smart.EMBED_DIM)
    view /= np.linalg.norm(view)
    other = rng.normal(size=smart.EMBED_DIM)
    other /= np.linalg.norm(other)
    files = [Footage(f"{n}.mp4", 30.0, 1920, 1080, 30.0, True) for n in ("phone", "camera", "drone")]
    footage = {f.id: f for f in files}
    # the phone and the camera filmed the same view; the drone something else, its moments a little weaker
    analyses = {files[0].id: _plain(30, 1, _same(30, view)), files[1].id: _plain(30, 2, _same(30, view)),
                files[2].id: _plain(30, 3, _same(30, other))}
    analyses[files[2].id].sharp[:] = 4.6
    s = AutoEditSettings(target_duration=8.0, clip_min=4.0, clip_max=4.0, min_gap=2.0, use_every_footage=False,
                         order="chronological")
    picked = {c.footage_id for c in select(footage, analyses, s, [f.id for f in files])}
    assert files[2].id in picked and not {files[0].id, files[1].id} <= picked
    # without the smarter analysis the weaker drone shot isn't picked: the same view twice instead
    plain = {fid: _plain(30, k + 1) for k, fid in enumerate(analyses)}
    plain[files[2].id].sharp[:] = 4.6
    assert files[2].id not in {c.footage_id for c in select(footage, plain, s, [f.id for f in files])}


def test_laughter_lifts_a_moment_and_wind_lowers_it():
    f = Footage("a.mp4", 20.0, 1920, 1080, 30.0, True)
    k = int(20 * smart.SOUND_RATE)
    sound = np.zeros((k, len(smart.GROUP_NAMES)), np.uint8)
    sound[10:16, smart.GROUP_NAMES.index("laughter")] = 230       # 5-8 s: laughing
    sound[30:36, smart.GROUP_NAMES.index("wind")] = 230           # 15-18 s: wind on the microphone
    a = _plain(20, 1, sound=sound)
    plain = score_samples({f.id: _plain(20, 1)}, {f.id: f}, "balanced")[f.id]
    heard = score_samples({f.id: a}, {f.id: f}, "balanced")[f.id]
    t = a.times
    assert (heard - plain)[(t > 5.5) & (t < 7.5)].min() > 0.1
    assert (heard - plain)[(t > 15.5) & (t < 17.5)].max() < -0.1
    assert np.allclose((heard - plain)[(t > 9) & (t < 14)], 0, atol=1e-6)


def test_a_loud_view_without_speech_is_scenic():
    from kangedit.scenes import scenic_scores
    f = Footage("a.mp4", 20.0, 1920, 1080, 30.0, True)
    k = int(20 * smart.SOUND_RATE)
    loud = _plain(20, 1)
    loud.audio[:] = -12.0                                          # waves crashing
    sound = np.zeros((k, len(smart.GROUP_NAMES)), np.uint8)
    sound[:, smart.GROUP_NAMES.index("nature")] = 220
    heard = Analysis(**{**loud.__dict__, "sound": sound})
    plain = scenic_scores({f.id: loud}, {f.id: f})[f.id]
    smart_sc = scenic_scores({f.id: heard}, {f.id: f})[f.id]
    assert smart_sc[8:-8].mean() > plain[8:-8].mean() + 0.05


def _clip(path, seconds=20, gop=30, size="640x360"):
    try:
        subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=size={size}:rate=30:duration={seconds}", "-f", "lavfi", "-i",
                        f"sine=frequency=300:duration={seconds}", "-c:v", "libx264", "-preset", "veryfast",
                        "-g", str(gop), "-pix_fmt", "yuv420p", "-shortest", path], check=True, capture_output=True)
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"FFmpeg can't make the test clip: {e}")
    i = probe(path)
    return Footage(path, i.duration, i.width, i.height, i.fps, i.has_audio)


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_quick_look_from_the_keyframes(tmp_path):
    f = _clip(str(tmp_path / "a.mp4"))
    q = an.quick_look(f)
    full = an.analyze(f, use_gpu=False)
    assert q.quick and not full.quick
    assert len(q.times) == len(full.times) and len(q.thumbs) == len(full.thumbs)
    assert q.thumb_times[1] - q.thumb_times[0] == 1.0 and len(q.embed) == len(q.times)
    # measured on the keyframes only, the same picture on average (moment by moment it can differ)
    assert abs(q.sharp.mean() - full.sharp.mean()) < 0.2 * full.sharp.mean()
    assert abs(q.brightness.mean() - full.brightness.mean()) < 0.05 and abs(q.audio.mean() - full.audio.mean()) < 1.0
    q.save(an.quick_path_for(f))
    assert an.load_plain(f).quick is False                       # the full analysis is cached: it wins
    os.remove(an.analysis_path_for(f))
    assert an.load_plain(f).quick                                # else the quick look


def test_two_passes_in_the_app(qapp, monkeypatch, tmp_path):
    from kangedit.ui import main_window as mw
    w = mw.MainWindow()
    try:
        files = [Footage(str(tmp_path / f"c{k}.mp4"), 10.0, 1920, 1080, 30.0, True) for k in range(3)]
        for f in files:
            open(f.path, "wb").close()
        p = Project()
        p.footage = {f.id: f for f in files}
        # No analysis while the project opens; then the real queue_analysis. (Not monkeypatch.undo(): that would
        # also drop conftest's guards against message boxes, and a late box from an earlier test would block.)
        w.queue_analysis = lambda ids, force=False: None
        w.set_project(p)
        del w.queue_analysis
        looked, full_runs = [], []

        def quick(f, cancel=None):
            looked.append(f.name)
            return Analysis(**{**_plain(10).__dict__, "quick": True})

        def full(batch, progress=None, cancel=None, on_result=None, on_error=None, use_gpu=True):
            full_runs.append([f.name for f in batch])
            for f in batch:
                on_result(f, _plain(10))
            return {}
        monkeypatch.setattr(mw, "quick_look", quick)
        monkeypatch.setattr(mw, "analyze_many", full)
        monkeypatch.setattr(w.tasks, "submit", lambda name, job, done, err=None: done(job(lambda *a: None,
                                                                                            threading.Event())))
        w.queue_analysis([f.id for f in files])
        assert sorted(looked) == ["c0.mp4", "c1.mp4", "c2.mp4"]
        w._refine_thread.join(5)                                   # the full analysis, in the background
        qapp.processEvents()
        assert sorted(full_runs[0]) == ["c0.mp4", "c1.mp4", "c2.mp4"]
        assert all(not w.analyses[f.id].quick for f in files)
    finally:
        w.project.dirty = False
        w.close()


# ----------------------------------------------------------------------------- captions
def _bursts(spans, total=6.0, rate=16000):
    t = np.arange(int(total * rate)) / rate
    x = np.random.default_rng(0).normal(0, 0.002, t.size)        # a quiet room
    for a, b in spans:
        m = (t >= a) & (t < b)
        x[m] += 0.3 * np.sin(2 * np.pi * 220 * t[m])
    return x.astype(np.float32)


def test_word_times_are_set_on_the_sound():
    audio = _bursts([(1.00, 1.40), (1.60, 2.10), (3.00, 3.50)])
    words = [{"s": 0.90, "e": 1.30, "w": "halo"},               # Whisper: 0.1 s early, ends 0.1 s too soon
             {"s": 1.70, "e": 2.25, "w": "semua"},              # starts late, ends late
             {"s": 3.00, "e": 3.50, "w": "ayo"}]
    out = captions.refine_word_times(words, audio)
    assert abs(out[0]["s"] - 1.0) < 0.04 and abs(out[0]["e"] - 1.40) < 0.06
    assert abs(out[1]["s"] - 1.6) < 0.04 and abs(out[1]["e"] - 2.10) < 0.06
    assert out[0]["e"] < out[1]["s"] and [w["w"] for w in out] == ["halo", "semua", "ayo"]
    noisy = np.random.default_rng(1).normal(0, 0.2, audio.size).astype(np.float32)   # wind: no pause to find
    assert captions.refine_word_times(words, noisy) == words


def test_transcripts_are_kept_for_every_project_and_silent_footage_is_skipped(tmp_path, monkeypatch):
    f = Footage(str(tmp_path / "talk.mp4"), 6.0, 1920, 1080, 30.0, True)
    open(f.path, "wb").close()
    loads = []
    monkeypatch.setattr(captions, "load_model", lambda name, progress=None: loads.append(name))
    monkeypatch.setattr(captions, "_load_audio_16k", lambda path, *a: _bursts([(1.0, 3.0)]))
    monkeypatch.setattr(captions, "speech_seconds", lambda audio: 2.0)
    monkeypatch.setattr(captions, "transcribe_audio",
                        lambda audio, model, lang, progress=None, cancel=None: ([{"s": 1.0, "e": 1.5, "w": "halo"}], "id"))
    assert captions.transcribe(f, "large-v3", "")[0]["w"] == "halo" and loads == ["large-v3"]
    g = Footage(f.path, 6.0, 1920, 1080, 30.0, True)                 # the same file in another project
    assert captions.transcribe(g, "large-v3", "")[0]["w"] == "halo" and loads == ["large-v3"]   # no model
    assert g.transcript_lang == "id"
    assert captions.cached_transcript(g, "small", "", any_model=True) is not None
    assert captions.cached_transcript(g, "small", "") is None
    captions.transcribe(g, "large-v3", "", use_cache=False)          # Transcribe again
    assert loads == ["large-v3", "large-v3"]
    # a view with nothing said: no speech model at all
    h = Footage(str(tmp_path / "view.mp4"), 6.0, 1920, 1080, 30.0, True)
    open(h.path, "wb").close()
    monkeypatch.setattr(captions, "speech_seconds", lambda audio: 0.2)
    assert captions.transcribe(h, "large-v3", "") == [] and h.transcript == [] and len(loads) == 2


@pytest.mark.skipif(not smart.available(), reason="needs onnxruntime (voice activity detection)")
def test_speech_seconds_on_silence():
    assert captions.speech_seconds(np.zeros(16000 * 3, np.float32)) < captions.MIN_SPEECH


def test_the_speech_model_suits_the_computer(monkeypatch):
    monkeypatch.setattr(captions, "_cuda", False)
    assert captions.resolve_model("auto") == "large-v3-turbo" and captions.resolve_model("small") == "small"
    monkeypatch.setattr(captions, "_cuda", True)
    assert captions.resolve_model("auto") == "large-v3"
    monkeypatch.setattr(os, "cpu_count", lambda: 12)               # a 6-core, 12-thread Ryzen 5 3600
    assert captions.cpu_threads() == 6
    assert Project().settings.whisper_model == "auto"


@needs_models
def test_smarter_analysis_is_on_by_default(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        assert w.a_smart.isChecked() and w.a_smart.isEnabled() and smart.enabled()
    finally:
        w.project.dirty = False
        w.close()


def test_opening_a_big_older_project_stays_responsive(qapp, monkeypatch, tmp_path):
    """An older project (analyses made before the smarter analysis) opens without freezing: the upgrade runs in
    the background, outside the task queue, and the window refreshes a few times, not twice per file."""
    import time

    from PySide6.QtTest import QTest

    from kangedit.ui import main_window as mw
    files = [Footage(str(tmp_path / f"c{k:03d}.mp4"), 10.0, 1920, 1080, 30.0, False) for k in range(150)]
    for f in files:
        open(f.path, "wb").close()
        _plain(10).save(an.analysis_path_for(f))                    # no smarter analysis yet
    monkeypatch.setattr(smart, "enabled", lambda: True)
    upgraded = []

    def fake_smart(f, a, cancel=None):
        a.sem = np.ones((len(a.thumbs), smart.EMBED_DIM), np.float16)
        upgraded.append(f.id)
        return True
    monkeypatch.setattr(mw, "make_smart", fake_smart)
    w = mw.MainWindow()
    try:
        refreshes = []
        real = w.refresh_all
        monkeypatch.setattr(w, "refresh_all", lambda *a, **k: (refreshes.append(1), real(*a, **k)))
        queued = []
        monkeypatch.setattr(w.tasks, "submit", lambda name, job, done, err=None: (queued.append(name),
                                                                                   done(job(lambda *a: None,
                                                                                            threading.Event()))))
        p = Project()
        p.footage = {f.id: f for f in files}
        t = time.perf_counter()
        w.set_project(p)
        assert time.perf_counter() - t < 10.0
        w._refine_thread.join(20)
        QTest.qWait(800)
        assert len(upgraded) == 150 and all(len(w.analyses[f.id].sem) for f in files)
        assert len(refreshes) <= 6                                   # was two full refreshes per file
    finally:
        w.project.dirty = False
        w.close()
