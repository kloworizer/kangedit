"""A long timeline (hundreds of clips) is made in parts and joined: the preview sound and the export come out
as if made in one FFmpeg run."""

import os
import subprocess
import wave

import numpy as np
import pytest

from kangedit import export as ex
from kangedit.ffmpeg import FFmpegError, ffmpeg_bin, probe, run
from kangedit.models import Footage, OverlayClip, Project, SoundClip, TimelineClip


def _fake_project(n=400):
    f = Footage("/media/very/long/path/to/the/footage/VID_20260501_091005.mp4", 600.0, 1920, 1080, 30.0, True)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 1.0 + k, 1.5 + k) for k in range(n)]
    return p, f


def test_a_short_timeline_is_one_run():
    p, _ = _fake_project(20)
    assert ex._parts(p, 30.0, False, video=False) is None


def test_parts_end_only_at_straight_cuts():
    p, f = _fake_project(400)
    for k in range(70, 400, 7):
        p.clips[k].transition = "dissolve"                # transitions into these clips
    p.clips[150].audio_split = -0.3                       # a J-cut
    starts = p.clip_starts()
    spanning = OverlayClip(f.id, starts[160] - 0.25, src_in=0.0, src_out=2.0)   # over the cut into clip 160
    inside = OverlayClip(f.id, starts[5] + 0.1, src_in=0.0, src_out=0.2)
    p.overlays = [spanning, inside]
    parts, rest = ex._parts(p, 30.0, False, video=True)
    assert len(parts) >= 3
    assert [i for part in parts for i in part.rng] == list(range(400))       # every clip once, in order
    es = ex.edits(p)
    for part in parts[1:]:
        a = part.rng.start
        assert es[a].half == 0 and es[a].split == 0
    for part in parts:                                    # each overlay in one part, or in the final pass
        for o in part.overlays:
            assert 0 <= o.start and o.end <= (part.f1 - part.f0) / 30.0 + 1e-3
    assert sum(len(part.overlays) for part in parts) + len(rest) == 2
    # an overlay across the whole timeline doesn't stop it being made in parts: it goes to the final pass
    p.overlays = [OverlayClip(f.id, 0.0, src_in=0.0, src_out=599.0)]
    parts2, rest2 = ex._parts(p, 30.0, False, video=True)
    assert len(parts2) >= 3 and [o.id for o in rest2] == [p.overlays[0].id]
    first = parts[0]
    assert len(first.overlays) == 1 and first.overlays[0].start == pytest.approx(inside.start - first.f0 / 30.0)
    assert all(b.f0 == a.f1 for a, b in zip(parts, parts[1:]))                # no gap, no overlap


def _chirp(tmp_path):
    src = str(tmp_path / "chirp.mov")
    try:
        run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30:duration=8",
             "-f", "lavfi", "-i", "aevalsrc='0.5*sin(2*PI*(200*t+50*t*t))':s=48000:d=8", "-shortest",
             "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-y", src])
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"FFmpeg can't make the test clip: {e}")
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    p = Project()
    p.footage = {f.id: f}
    p.settings.fps = 30.0
    # 24 short clips, with a transition and a J-cut among them, and quiet music over it all
    p.clips = [TimelineClip(f.id, 0.2 + 0.3 * k, 0.6 + 0.3 * k) for k in range(24)]
    p.clips[6].transition = "dissolve"
    p.clips[13].audio_split = -0.1
    p.sounds = [SoundClip(src, start=0.0, src_in=0.0, src_out=7.0, volume=0.1)]
    return p, f


def _read(wav):
    with wave.open(wav) as w:
        return w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)[:, 0]


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_sound_in_parts_matches_one_run(tmp_path, monkeypatch):
    p, f = _chirp(tmp_path)
    try:
        whole = ex.render_audio(p, False, str(tmp_path / "whole.wav"))
    except FFmpegError as e:
        pytest.skip(f"needs a newer FFmpeg: {e}")
    monkeypatch.setattr(ex, "MAX_INPUTS", 4)
    monkeypatch.setattr(ex, "PART_INPUTS", 5)
    assert len(ex._parts(p, 30.0, False, video=False)[0]) >= 3
    parted = ex.render_audio(p, False, str(tmp_path / "parts.wav"))
    rate, a = _read(whole)
    _, b = _read(parted)
    assert abs(len(a) - len(b)) <= rate / 30                       # the same length, within a frame
    n = min(len(a), len(b))
    # the same sound, sample for sample (a few samples of resampling slack at the joins)
    assert np.mean(np.abs(a[:n].astype(np.int32) - b[:n])) < 200
    assert np.abs(b).max() > 1000


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_export_in_parts(tmp_path, monkeypatch):
    p, f = _chirp(tmp_path)
    p.overlays = [OverlayClip(f.id, 0.5, src_in=1.0, src_out=1.4, scale=0.3)]     # inside the first part
    monkeypatch.setattr(ex, "MAX_INPUTS", 4)
    monkeypatch.setattr(ex, "PART_INPUTS", 5)
    out = str(tmp_path / "out.mp4")
    seen = []
    try:
        ex.export(p, ex.ExportSettings(out, p.settings.aspect, "720p", 30.0, use_gpu=False),
                  progress=lambda v, m="": seen.append(v))
    except FFmpegError as e:
        pytest.skip(f"export needs a newer FFmpeg: {e}")
    raw = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", out, "-vf", "scale=32:18", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    frames = len(raw) // (32 * 18)
    assert frames == ex.plan(p, 30.0)[-1].start_f + ex.plan(p, 30.0)[-1].nfr
    i = probe(out)
    assert i.has_audio and i.duration == pytest.approx(frames / 30.0, abs=0.1)
    assert seen and max(seen) <= 1.0 and os.path.getsize(out) > 0
    assert not [d for d in os.listdir(ex.cache_dir("tmp")) if d.startswith("export_")]    # parts cleaned up
