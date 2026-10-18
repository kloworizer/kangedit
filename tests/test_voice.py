"""Voice clean-up: noise and wind rumble taken out of a clip's sound, in the preview mix as in the export."""

import wave

import numpy as np
import pytest

from kangedit import voice
from kangedit.export import render_audio
from kangedit.ffmpeg import FFmpegError, ffmpeg_bin, probe, run
from kangedit.models import Footage, Project, TimelineClip


def test_filter_strength():
    assert voice.ffmpeg_filter(0.0) == ""
    weak, strong = voice.ffmpeg_filter(0.2), voice.ffmpeg_filter(1.0)
    assert "afftdn" in weak and "highpass" in weak and weak.endswith(",")
    nr = lambda f: float(f.split("nr=")[1].split(":")[0])   # noqa: E731
    assert nr(strong) > nr(weak)


def test_suggests_it_where_wind_was_heard():
    class A:
        times = np.arange(0, 10, 0.25, dtype=np.float32)
        sound = None

    a = A()
    c = TimelineClip("x", 2.0, 6.0)
    assert voice.suggest(c, a) is None and voice.suggest(c, None) is None     # nothing heard: nothing to say
    from kangedit import smart
    n = int(10 * smart.SOUND_RATE)
    scores = np.zeros((n, len(smart.GROUP_NAMES)), np.uint8)
    scores[int(2 * smart.SOUND_RATE):int(6 * smart.SOUND_RATE), smart.GROUP_NAMES.index("wind")] = 150
    a.sound = scores
    assert voice.suggest(c, a) == voice.DEFAULT
    assert voice.suggest(TimelineClip("x", 7.0, 9.0), a) is None


def _band(x, rate, lo, hi):
    sp = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / rate)
    return float(sp[(f >= lo) & (f < hi)].sum())


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_cleaned_voice_keeps_the_voice_and_drops_the_noise(tmp_path):
    src = str(tmp_path / "noisy.mp4")
    try:   # a "voice" (440 Hz), quiet at first, in hiss and rumble (40 Hz)
        run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=c=gray:size=160x90:rate=30:duration=4",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=4,volume='if(lt(t,1),0,1)':eval=frame",
             "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.08:duration=4",
             "-f", "lavfi", "-i", "sine=frequency=40:duration=4",
             "-filter_complex", "[1:a][2:a][3:a]amix=inputs=3:normalize=0[a]", "-map", "0:v", "-map", "[a]",
             "-shortest", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-y", src])
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"FFmpeg can't make the test clip: {e}")
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)

    from kangedit.analysis import analyze
    floor = voice.noise_floor(analyze(f), 0.0, 4.0)        # the noise alone, before the voice starts
    assert -45.0 < floor < -20.0

    def mix(clean):
        p = Project()
        p.footage = {f.id: f}
        p.clips = [TimelineClip(f.id, 0.5, 3.5, voice_clean=clean, voice_floor=floor)]
        out = str(tmp_path / f"mix_{clean}.wav")
        try:
            render_audio(p, False, out)
        except FFmpegError as e:
            pytest.skip(f"needs a newer FFmpeg: {e}")
        w = wave.open(out)
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())[:, 0]
        return a[36000:-24000].astype(np.float64), w.getframerate()

    raw, rate = mix(0.0)
    clean, _ = mix(1.0)
    voice_kept = _band(clean, rate, 400, 480) / _band(raw, rate, 400, 480)
    hiss_left = _band(clean, rate, 2000, 12000) / _band(raw, rate, 2000, 12000)
    rumble_left = _band(clean, rate, 20, 60) / _band(raw, rate, 20, 60)
    assert voice_kept > 0.5 and hiss_left < 0.2 and rumble_left < 0.1
