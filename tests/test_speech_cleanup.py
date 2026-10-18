"""Background noise cleaned out of the sound before speech recognition (road, wind, crowd)."""

import numpy as np
import pytest

from kangedit import captions
from kangedit.ffmpeg import ffmpeg_bin, probe, run
from kangedit.models import Footage

RATE = 16000


def _talk(seconds=6.0, noise=0.0, seed=0):
    """A "voice" (a tone in bursts, like syllables, pauses between phrases) over brown "road" noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    voice = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0) * ((t % 2.0) < 1.4)
    road = np.cumsum(rng.normal(0, 1, t.size))
    road = road - np.convolve(road, np.ones(400) / 400, mode="same")       # rumble, no drift
    road = noise * road / (np.abs(road).max() + 1e-9)
    return (voice + road).astype(np.float32)


def test_cleanup_is_chosen_for_loud_noise():
    quiet = _talk(noise=0.003)
    loud = _talk(noise=0.25)
    assert captions.needs_cleanup(quiet) is None                     # a clear voice: as recorded
    floor = captions.needs_cleanup(loud)
    assert floor is not None and -40 < floor < -10
    assert captions.needs_cleanup(loud, "off") is None
    assert captions.needs_cleanup(quiet, "on") is not None
    assert "afftdn" in captions.cleanup_filter(-30.0) and "nf=-30.0" in captions.cleanup_filter(-30.0)


def _band(x, lo, hi):
    sp = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / RATE)
    return float(sp[(f >= lo) & (f < hi)].sum())


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_transcription_hears_the_cleaned_sound(tmp_path, monkeypatch):
    wav = str(tmp_path / "road.wav")
    import wave
    a = _talk(noise=0.25)
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    src = str(tmp_path / "road.mp4")
    run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=c=gray:size=160x90:rate=30:duration=6",
         "-i", wav, "-shortest", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-y", src])
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    heard = {}

    def fake(audio, model_name, language="", progress=None, cancel=None, offset=0.0, quick=False, vad=True):
        heard.setdefault("audio", []).append(audio)
        return [{"s": 0.1, "e": 0.5, "w": "halo"}], "id"

    monkeypatch.setattr(captions, "transcribe_audio", fake)
    monkeypatch.setattr(captions, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(captions, "save_transcript", lambda *a, **k: None)
    monkeypatch.setattr(captions, "speech_seconds", lambda audio: 5.0)
    said = []
    captions.transcribe(f, "small", "", lambda p, m="": said.append(m), use_cache=False, cleanup="auto")
    raw = captions._load_audio_16k(src)
    got = heard["audio"][0]
    assert any("background noise" in m for m in said)

    def snr(x):           # the voice's band against the rumble below it
        return _band(x, 200, 240) / _band(x, 20, 90)

    assert snr(got) > 10 * snr(raw)
    heard.clear()
    captions.transcribe(f, "small", "", None, use_cache=False, cleanup="off")
    assert np.allclose(heard["audio"][0], raw)


def test_panel_setting(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        cb = w.caps_panel.cb_clean
        assert cb.currentData() == "auto" == w.project.settings.speech_cleanup
        cb.setCurrentIndex(cb.findData("on"))
        assert w.project.settings.speech_cleanup == "on"
    finally:
        w.project.dirty = False
        w.close()


def test_noisy_sound_is_heard_whole_not_skipped(tmp_path, monkeypatch):
    """Voice activity detection finds no voice in loud noise (a car, wind) where the speech is still clear: the
    file isn't skipped as "nobody speaks", and the recogniser hears all of it."""
    f = Footage(str(tmp_path / "x.mp4"), 6.0, 160, 90, 30.0, True)
    loud = _talk(noise=0.25)
    quiet = np.zeros(RATE * 6, np.float32)
    calls = []
    monkeypatch.setattr(captions, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(captions, "save_transcript", lambda *a, **k: None)
    monkeypatch.setattr(captions, "speech_seconds", lambda audio: 0.0)            # VAD hears nothing
    monkeypatch.setattr(captions, "transcribe_audio",
                        lambda audio, *a, vad=True, **k: calls.append(vad) or ([{"s": 1, "e": 2, "w": "halo"}], "id"))
    for audio, want in ((loud, [False]), (quiet, [])):
        calls.clear()
        monkeypatch.setattr(captions, "_load_audio_16k", lambda *a, audio=audio, **k: audio)
        words = captions.transcribe(f, "small", "", use_cache=False, cleanup="off")
        assert calls == want and bool(words) == bool(want)


def test_old_empty_transcripts_are_redone(tmp_path, monkeypatch):
    import json
    f = Footage(str(tmp_path / "x.mp4"), 6.0, 160, 90, 30.0, True)
    path = tmp_path / "t.json"
    monkeypatch.setattr(captions, "transcript_path", lambda *a: str(path))
    path.write_text(json.dumps({"version": 1, "language": "", "words": []}))
    assert captions.cached_transcript(f, "small", "") is None                    # maybe missed in noise
    path.write_text(json.dumps({"version": 1, "language": "id", "words": [{"s": 0, "e": 1, "w": "a"}]}))
    assert captions.cached_transcript(f, "small", "")[0][0]["w"] == "a"          # kept
    path.write_text(json.dumps({"version": captions.TRANSCRIPT_VERSION, "language": "", "words": []}))
    assert captions.cached_transcript(f, "small", "") == ([], "")


def test_sign_off_invented_at_the_end_of_noise_is_dropped(monkeypatch):
    from types import SimpleNamespace as NS

    def seg(s, e, text):
        return NS(start=s, end=e, text=text, no_speech_prob=0.0, avg_logprob=-0.6,     # noise: Whisper is "sure"
                  words=[NS(start=s, end=e, word=w, probability=0.9) for w in text.split()])
    segs = [seg(1.0, 3.0, " kita mau ke mana"), seg(4.0, 5.0, " terima kasih"),
            seg(8.0, 10.0, " Terima kasih telah menonton!")]
    model = NS(transcribe=lambda audio, **k: (iter(segs), NS(duration=10.0, language="id")))
    monkeypatch.setattr(captions, "_get_model", lambda name: model)
    monkeypatch.setattr(captions, "_batched", lambda m: model)
    for vad, want in ((False, "kita mau ke mana terima kasih"), (True, "kita mau ke mana terima kasih Terima kasih telah menonton!")):
        words, _ = captions.transcribe_audio(np.zeros(RATE * 10, np.float32), "small", "id", vad=vad)
        assert " ".join(w["w"] for w in words) == want
