import os
import sys
import threading
import time

import pytest

from kangedit import proxy
from kangedit.ffmpeg import FFmpegError, ffmpeg_bin, probe, run
from kangedit.models import Footage

F = Footage("in.mp4", 10.0, 3840, 2160, 30.0, True)


def test_gpu_args_fail_on_decode_errors():
    args = proxy.proxy_args(F, 540, "out.mp4", "d3d11va", "h264_amf")
    assert args[:3] == ["-xerror", "-hwaccel", "d3d11va"]   # garbage frames must not pass silently
    assert "h264_amf" in args and "scale=960:540" in args[args.index("-vf") + 1]
    assert args[args.index("-g") + 1] == str(proxy.GOP) and args[args.index("-bf") + 1] == "0"


def test_cpu_args():
    args = proxy.proxy_args(F, 540, "out.mp4")
    assert "-hwaccel" not in args and "libx264" in args and "format=yuv420p" in args[args.index("-vf") + 1]


@pytest.fixture
def attempts(monkeypatch, tmp_path):
    out = tmp_path / "p.mp4"
    monkeypatch.setattr(proxy, "proxy_path_for", lambda f, h: str(out))
    monkeypatch.setattr(proxy, "hw_decoder", lambda: "d3d11va")
    monkeypatch.setattr(proxy, "hw_encoder", lambda codec: "h264_amf")
    seen = []

    def run(args, duration, progress, cancel, label="", stall=None):
        dec = args[args.index("-hwaccel") + 1] if "-hwaccel" in args else None
        enc = args[args.index("-c:v") + 1]
        seen.append((dec, enc))
        if len(seen) <= fail_first[0]:
            raise FFmpegError("hardware accelerator failed to decode picture")
        open(args[-1], "wb").write(b"x")

    fail_first = [0]
    monkeypatch.setattr(proxy, "run_ffmpeg", run)
    monkeypatch.setattr(proxy, "_stalled", set())
    return seen, fail_first


def test_gpu_decode_failure_falls_back_to_cpu_decode_then_cpu(attempts):
    seen, fail_first = attempts
    fail_first[0] = 2
    proxy.generate_proxy(F, 540, use_gpu=True)
    assert seen == [("d3d11va", "h264_amf"), (None, "h264_amf"), (None, "libx264")]


def test_cpu_only_when_gpu_off(attempts):
    seen, _ = attempts
    proxy.generate_proxy(F, 540, use_gpu=False)
    assert seen == [(None, "libx264")]


# ----------------------------------------------------------------------------- a GPU attempt that hangs
def _hanging_args(tmp_path):
    """ffmpeg arguments that block forever before any output (reading a pipe nobody writes): a stand-in for
    a GPU driver that never starts."""
    fifo = str(tmp_path / "never.fifo")
    if not os.path.exists(fifo):
        os.mkfifo(fifo)
    return ["-f", "rawvideo", "-pix_fmt", "gray", "-s", "16x16", "-i", fifo, "-f", "null", "-"]


@pytest.mark.skipif(sys.platform == "win32" or not ffmpeg_bin(), reason="needs FFmpeg and a POSIX fifo")
def test_a_hanging_gpu_attempt_falls_back_to_the_cpu(tmp_path, monkeypatch):
    from kangedit import proxy
    src = str(tmp_path / "clip.mp4")
    run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=2",
         "-pix_fmt", "yuv420p", "-y", src])
    i = probe(src)
    f = Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)
    monkeypatch.setattr(proxy, "STALL", 1.5)
    monkeypatch.setattr(proxy, "_stalled", set())
    monkeypatch.setattr(proxy, "hw_decoder", lambda: "fakegpu")
    monkeypatch.setattr(proxy, "hw_encoder", lambda codec: None)
    monkeypatch.setattr(proxy, "cache_dir", lambda kind: tmp_path)
    real_args = proxy.proxy_args
    tried = []

    def args(footage, height, out, hw=None, encoder=None):
        tried.append(hw)
        return _hanging_args(tmp_path) if hw == "fakegpu" else real_args(footage, height, out, hw, encoder)
    monkeypatch.setattr(proxy, "proxy_args", args)
    labels = []
    t0 = time.monotonic()
    out = proxy.generate_proxy(f, 120, lambda p, m: labels.append(m))
    assert os.path.getsize(out) > 0 and time.monotonic() - t0 < 30
    assert tried == ["fakegpu", None] and ("fakegpu", None) in proxy._stalled
    assert any("GPU" in m for m in labels) and any("(CPU)" in m for m in labels)   # the status shows each try
    os.remove(out)
    tried.clear()
    proxy.generate_proxy(f, 120)                     # the hanging way isn't tried again this session
    assert tried == [None]


@pytest.mark.skipif(sys.platform == "win32" or not ffmpeg_bin(), reason="needs FFmpeg and a POSIX fifo")
def test_cancel_stops_a_silent_ffmpeg(tmp_path):
    from kangedit.ffmpeg import Cancelled, run_ffmpeg
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        run_ffmpeg(_hanging_args(tmp_path), 10.0, None, cancel)
    assert time.monotonic() - t0 < 5
