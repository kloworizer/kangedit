"""Decoding the preview on the graphics card: the Auto / Off rule, and the silent fall back to the CPU (this
machine has no GPU, so the card's side is simulated: a failing open, a failure mid-stream)."""

import dataclasses
import logging
import time

import av
import numpy as np
import pytest

from kangedit import decoder
from kangedit.decoder import DecoderPool, FrameDecoder, wants_gpu
from kangedit.models import Footage

FPS = 25


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """1 s of 1920x1080 moving test pattern: 10-bit HEVC when PyAV can encode it, else H.264."""
    path = str(tmp_path_factory.mktemp("media") / "big.mp4")
    try:
        av.codec.Codec("libx265", "w")
        codec, pix = "libx265", "yuv420p10le"
    except Exception:  # noqa: BLE001
        codec, pix = "libx264", "yuv420p"
    w, h = 1920, 1080
    with av.open(path, "w") as out:
        st = out.add_stream(codec, rate=FPS)
        st.width, st.height, st.pix_fmt = w, h, pix
        st.codec_context.gop_size = 12
        if codec == "libx265":
            st.codec_context.options = {"x265-params": "log-level=error", "preset": "ultrafast"}
        else:
            st.codec_context.options = {"preset": "ultrafast"}
        x = np.linspace(0, 255, w, dtype=np.float32)
        for i in range(FPS):
            img = np.empty((h, w, 3), np.uint8)
            img[..., 0] = (x + i * 7) % 256
            img[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
            img[..., 2] = 255 - img[..., 0]
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format=pix)):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    hevc = codec == "libx265"
    return Footage(path, 1.0, w, h, FPS, False, codec="hevc" if hevc else "h264",
                   bit_depth=10 if hevc else 8)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Each test starts in Auto with no remembered failures."""
    monkeypatch.setattr(decoder, "_gpu_mode", "auto")
    monkeypatch.setattr(decoder, "_hw_failed", set())
    monkeypatch.setattr(decoder, "_hw_logged", set())


def frames(dec, times):
    return [dec.frame_at(t) for t in times]


TIMES = [0.0, 0.4, 0.2, 0.92]


def test_auto_rule_picks_the_gpu_only_for_big_or_heavy_footage(tmp_path):
    def f(w, h, codec="h264", depth=8, proxy=None):
        return Footage("x.mp4", 10, w, h, 25, False, codec=codec, bit_depth=depth, proxy_path=proxy)

    assert wants_gpu(f(3840, 2160), False)
    assert wants_gpu(f(2160, 3840), False)                      # vertical 4K
    assert wants_gpu(f(1920, 1080, "hevc", 10), False)          # 10-bit HEVC
    assert wants_gpu(f(2560, 1440), False)                      # 1440p without a proxy
    assert not wants_gpu(f(1920, 1080), False)                  # ordinary HD: CPU
    assert not wants_gpu(f(1280, 720, "hevc", 10), False)       # small, even if 10-bit
    assert not wants_gpu(f(3840, 2160), True)                   # the proxy is in use: no change
    p = tmp_path / "proxy.mp4"
    p.write_bytes(b"")
    assert not wants_gpu(f(2560, 1440, proxy=str(p)), False)    # a proxy exists (proxies off): CPU is fine
    img = f(3840, 2160)
    img.kind = "image"
    assert not wants_gpu(img, False)
    decoder.set_gpu_mode("off")
    assert not wants_gpu(f(3840, 2160), False)
    assert not wants_gpu(f(1920, 1080, "hevc", 10), False)


def test_off_never_tries_the_gpu(clip, monkeypatch):
    calls = []
    monkeypatch.setattr(decoder, "hw_devices", lambda: ["cuda"])
    monkeypatch.setattr(decoder, "_open_hw", lambda path, dev: calls.append(dev) or av.open(path))
    big = dataclasses.replace(clip, width=3840, height=2160)
    decoder.set_gpu_mode("off")
    pool = DecoderPool(gpu=True)
    dec = pool.get(big, False, 360)
    assert dec.hw_device is None and calls == []
    pool.close()
    decoder.set_gpu_mode("auto")          # Auto does try it (and here falls back: no hardware decoder)
    pool = DecoderPool(gpu=True)
    pool.get(big, False, 360)
    assert calls == ["cuda"]
    pool.close()
    cpu_pool = DecoderPool()               # pools other than the preview's never use the card
    cpu_pool.get(big, False, 360)
    assert calls == ["cuda"]
    cpu_pool.close()


def test_failing_gpu_open_falls_back_with_identical_frames(clip, monkeypatch, caplog):
    ref = FrameDecoder(clip.path, max_height=540)
    want = frames(ref, TIMES)
    ref.close()

    calls = []

    def broken(path, dev):
        calls.append(dev)
        raise av.error.PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(decoder, "hw_devices", lambda: ["cuda", "vaapi"])
    monkeypatch.setattr(decoder, "_open_hw", broken)
    with caplog.at_level(logging.INFO, logger="kangedit.decoder"):
        dec = FrameDecoder(clip.path, max_height=540, gpu=True)
        assert dec.hw_device is None and dec.decoding_on == "CPU"
        got = frames(dec, TIMES)
        dec.close()
        assert calls == ["cuda", "vaapi"]
        FrameDecoder(clip.path, max_height=540, gpu=True).close()   # remembered: not tried again
        assert calls == ["cuda", "vaapi"]
    assert len([r for r in caplog.records if r.name == "kangedit.decoder"]) == 2   # once per device
    for a, b in zip(want, got):
        assert np.array_equal(a, b)


def test_real_gpu_attempt_here_still_decodes_the_same(clip):
    """Whatever this machine has (no GPU in CI: cuda fails to open), gpu=True gives the software frames."""
    ref = FrameDecoder(clip.path, max_height=540)
    dec = FrameDecoder(clip.path, max_height=540, gpu=True)
    print(f"\nPyAV {av.__version__}: hwaccel {'available' if decoder.hwaccel_supported() else 'missing'}, "
          f"devices {decoder.hw_devices()}, decoding on {dec.decoding_on}")
    for t in TIMES:
        a, b = ref.frame_at(t), dec.frame_at(t)
        if dec.hw_device is None:
            assert np.array_equal(a, b)
        else:   # a real card: same picture, give or take rounding in its output
            assert np.abs(a.astype(int) - b.astype(int)).mean() < 2
    ref.close()
    dec.close()


def test_gpu_failing_mid_stream_hands_over_to_the_cpu(clip, monkeypatch):
    ref = FrameDecoder(clip.path, max_height=540)
    want = frames(ref, TIMES)
    ref.close()
    dec = FrameDecoder(clip.path, max_height=540)
    dec.hw_device = "cuda"    # pretend the card opened it, then fails on the first packet

    def failing():
        del dec._frames     # once: the CPU decoder that takes over reads normally
        raise av.error.InvalidDataError(1094995529, "hardware decoder error")
        yield

    dec._frames = failing
    got = frames(dec, TIMES)
    assert dec.hw_device is None
    assert ("hevc" if clip.bit_depth == 10 else "h264", "cuda") in decoder._hw_failed
    for a, b in zip(want, got):
        assert np.array_equal(a, b)
    dec.close()


def test_without_hwaccel_in_pyav_the_gpu_is_never_asked_for(clip, monkeypatch):
    monkeypatch.setattr(decoder, "HWAccel", None)
    assert not decoder.hwaccel_supported()
    assert decoder.hw_devices() == []
    dec = FrameDecoder(clip.path, max_height=360, gpu=True)
    assert dec.hw_device is None and dec.frame_at(0.5).shape == (360, 640, 3)
    dec.close()


def test_cpu_decode_speed(clip):
    """Not asserted: how fast this machine plays the clip on the CPU (frame threads on every core)."""
    dec = FrameDecoder(clip.path, max_height=1080)
    dec.raw_at(0.0)
    t0 = time.perf_counter()
    n = 0
    for i in range(1, FPS):
        frame, _ = dec.raw_at(i / FPS)
        dec.to_rgb(frame)
        n += 1
    fps = n / (time.perf_counter() - t0)
    print(f"\nCPU decode + RGB of {clip.width}x{clip.height} {clip.codec} {clip.bit_depth}-bit: {fps:.0f} fps")
    dec.close()
