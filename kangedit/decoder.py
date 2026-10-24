"""Frame-accurate preview decoding with PyAV."""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Callable, Optional

import av
import numpy as np
from av.video.reformatter import VideoReformatter

from .models import Footage


_local = threading.local()
log = logging.getLogger("kangedit.decoder")

try:   # PyAV 13+: decoding on the graphics card
    from av.codec.hwaccel import HWAccel, hwdevices_available
except Exception:  # noqa: BLE001 — older PyAV: the preview decodes on the CPU only
    HWAccel = None
    hwdevices_available = None

# Graphics-card decoders to try, best first. cuda (NVIDIA) works on Windows and Linux alike.
_HW_CANDIDATES = {
    "win32": ["cuda", "d3d11va", "dxva2"],
    "darwin": ["videotoolbox"],
    "linux": ["cuda", "vaapi"],
}

GPU_MODES = ("auto", "off")
_gpu_mode = "auto"
_hw_lock = threading.Lock()
_hw_failed: set[tuple[str, str]] = set()   # (codec, device) that failed this session: not tried again
_hw_logged: set[tuple[str, str]] = set()


def hwaccel_supported() -> bool:
    """Whether this PyAV can decode on the graphics card at all."""
    return HWAccel is not None


def set_gpu_mode(mode: str) -> None:
    """The app option "GPU video decoding for the preview": "auto" or "off"."""
    global _gpu_mode
    _gpu_mode = mode if mode in GPU_MODES else "auto"


def gpu_mode() -> str:
    return _gpu_mode


def hw_devices() -> list[str]:
    """This platform's graphics-card decoders that this PyAV / FFmpeg build knows, best first."""
    if not hwaccel_supported():
        return []
    plat = "linux" if sys.platform.startswith("linux") else sys.platform
    try:
        built = set(hwdevices_available())
    except Exception:  # noqa: BLE001
        return []
    return [d for d in _HW_CANDIDATES.get(plat, []) if d in built]


def wants_gpu(footage: Footage, using_proxy: bool) -> bool:
    """The Auto rule: big or heavy footage decodes on the graphics card; small footage stays on the CPU,
    where copying each frame back from the card would cost more than it saves. A proxy in use is small
    by design, so it never needs the card."""
    if _gpu_mode == "off" or using_proxy or footage.is_image:
        return False
    short = min(footage.width or 0, footage.height or 0)
    codec = (footage.codec or "").lower()
    if short >= 2160:                                           # 4K and up
        return True
    if short >= 1080 and footage.bit_depth >= 10 and codec in ("hevc", "h265", "av1"):
        return True                                             # 10-bit HEVC / AV1: slow in software
    return short > 1080 and not footage.has_proxy()             # 1440p etc. with nothing smaller to use


def _hw_failure(codec: str, device: str, err: Exception) -> None:
    """Remember a graphics-card decoder that didn't work, logging it once."""
    with _hw_lock:
        _hw_failed.add((codec, device))
        if (codec, device) in _hw_logged:
            return
        _hw_logged.add((codec, device))
    log.info("GPU decoding (%s) unavailable for %s, using the CPU: %s", device, codec or "this video", err)


def _hw_ok(codec: str, device: str) -> bool:
    with _hw_lock:
        return (codec, device) not in _hw_failed


def _reformatter() -> VideoReformatter:
    """This thread's scaler. frame.reformat() keeps one per frame, and two threads converting the same
    frame (a still shown twice, both sides of a transition) would share it: a crash on PyAV 17."""
    r = getattr(_local, "reformatter", None)
    if r is None:
        r = _local.reformatter = VideoReformatter()
    return r


def _colorspace(cs: str, height: int) -> str:
    cs = (cs or "").lower()
    if cs.startswith("bt2020"):
        return "bt2020"
    if cs in ("smpte170m", "bt470bg"):
        return "itu601"
    if cs == "bt709":
        return "itu709"
    return "itu709" if height >= 576 else "itu601"


class FrameDecoder:
    """Sequential-friendly decoder: reads forward when possible, seeks otherwise."""

    def __init__(self, path: str, rotation: int = 0, color_space: str = "", max_height: int = 720,
                 gpu: bool = False):
        """`gpu` asks for the graphics card's video decoder (when there is one that works); anything that
        goes wrong with it falls back to the CPU without a word."""
        self.path = path
        self.rotation = rotation % 360
        self.hw_device: Optional[str] = None     # the graphics-card decoder in use, None = the CPU
        self._open(gpu)
        self.tb = float(self.stream.time_base) if self.stream.time_base else 1 / 90000
        # Source time is counted from the start of the file (its earliest stream), as FFmpeg's -ss does for the
        # export, the sound and the transcripts: a video stream that starts later (audio priming, edit lists)
        # then shows the same frame at the same time in the preview and in the export.
        if self.container.start_time is not None:
            self.start = self.container.start_time / 1_000_000
        elif self.stream.start_time is not None:
            self.start = float(self.stream.start_time * self.stream.time_base)
        else:
            self.start = 0.0
        rate = self.stream.average_rate or self.stream.guessed_rate or 30
        self.frame_dur = 1.0 / float(rate)
        cc = self.stream.codec_context
        self.src_w, self.src_h = cc.width, cc.height
        self.colorspace = _colorspace(color_space, self.src_h)
        scale = min(1.0, max_height / float(min(self.src_w, self.src_h) if self.rotation in (90, 270) else self.src_h))
        self.out_w = max(2, int(self.src_w * scale) // 2 * 2)
        self.out_h = max(2, int(self.src_h * scale) // 2 * 2)
        self._iter = None
        self._pos_t: Optional[float] = None      # time of the last frame consumed from the stream
        self._last_t: Optional[float] = None     # time of the last frame returned (_last_frame)
        self._last_frame = None                  # av.VideoFrame
        self._last_img: Optional[tuple] = None   # (frame, its RGB image): conversion cache for frame_at
        self._next_frame = None
        self._closing = False
        # While catching up to a target far ahead (after a keyframe seek), non-reference frames are
        # skipped: nothing depends on them and they're about half the frames of a long GOP. Frame
        # threads keep packets in flight, so full decoding resumes this far before the target.
        self._skip_margin = ((1 if self.hw_device else (os.cpu_count() or 4)) + 4) * self.frame_dur
        self._skip_until: Optional[float] = None
        self.lock = threading.Lock()

    def _open(self, gpu: bool):
        """Open the file, on the graphics card when asked and it works, else on the CPU (frame and slice
        threads on every core)."""
        if gpu:
            codec = ""
            for device in hw_devices():
                container = None
                try:
                    if not codec:
                        with av.open(self.path) as probe:
                            codec = probe.streams.video[0].codec_context.name
                    if not _hw_ok(codec, device):
                        continue
                    container = _open_hw(self.path, device)
                    stream = container.streams.video[0]
                    if not stream.codec_context.is_hwaccel:
                        raise RuntimeError("no hardware decoder for this codec")
                    container.seek(0, stream=stream)   # one frame proves the card decodes it
                    next(container.decode(stream))
                    container.seek(0, stream=stream)
                except Exception as e:  # noqa: BLE001 — any failure: try the next one, then the CPU
                    if container is not None:
                        try:
                            container.close()
                        except Exception:  # noqa: BLE001
                            pass
                    _hw_failure(codec, device, e)
                    continue
                self.container, self.stream, self.hw_device = container, stream, device
                return
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.hw_device = None

    @property
    def decoding_on(self) -> str:
        """"GPU (d3d11va)" or "CPU": where this file is decoded, for the viewer's note."""
        return f"GPU ({self.hw_device})" if self.hw_device else "CPU"

    def _to_software(self, err: Exception):
        """The graphics card failed mid-stream: carry on on the CPU, positioned to seek again."""
        _hw_failure(self.stream.codec_context.name, self.hw_device or "", err)
        try:
            self.container.close()
        except Exception:  # noqa: BLE001
            pass
        self._open(False)
        self._iter = None
        self._pos_t = None
        self._next_frame = None
        self._skip_until = None
        self._skip_margin = ((os.cpu_count() or 4) + 4) * self.frame_dur

    def close(self):
        self._closing = True   # makes a decode running in another thread bail out quickly
        with self.lock:
            try:
                self.container.close()
            except Exception:
                pass
            self._iter = None

    def _frames(self):
        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                yield frame

    def to_rgb(self, frame) -> np.ndarray:
        """Convert a frame from raw_at to RGB. Thread-safe: playback converts on several threads."""
        rng = frame.color_range
        src_range = "JPEG" if rng == 2 else "MPEG"
        f = _reformatter().reformat(frame, width=self.out_w, height=self.out_h, format="rgb24",
                                    src_colorspace=self.colorspace, src_color_range=src_range,
                                    dst_color_range="JPEG", interpolation="BILINEAR")
        img = f.to_ndarray()
        if self.rotation:
            img = np.ascontiguousarray(np.rot90(img, self.rotation // 90))
        return img

    def _seek(self, t: float):
        target = int((t + self.start) / self.tb)
        self.container.seek(max(0, target), stream=self.stream, backward=True, any_frame=False)
        self._iter = self._frames()
        self._next_frame = None
        self._skip_nonref(t - self._skip_margin)

    def _skip_nonref(self, until: Optional[float]):
        if until is not None and until <= 0:
            until = None
        if (until is None) != (self._skip_until is None):
            try:
                self.stream.codec_context.skip_frame = "NONREF" if until is not None else "DEFAULT"
            except Exception:  # noqa: BLE001 — an optimisation only
                until = None
        self._skip_until = until

    def frame_at(self, t: float, interrupt: Optional[Callable[[], bool]] = None) -> Optional[np.ndarray]:
        """RGB uint8 frame shown at source time t (seconds from file start).

        `interrupt` is polled while decoding forward to t (long-GOP seeks can take a second). When it
        returns True the nearest frame decoded so far is returned instead, so a newer request isn't
        stuck behind a stale one; the decoder stays positioned for the next call.
        """
        frame, _ = self.raw_at(t, interrupt)
        return None if frame is None else self.rgb_of(frame)

    def rgb_of(self, frame) -> np.ndarray:
        """to_rgb, reusing the last conversion when the same frame is shown again (grading a still)."""
        cached = self._last_img
        if cached is not None and cached[0] is frame:
            return cached[1]
        img = self.to_rgb(frame)
        self._last_img = (frame, img)
        return img

    def rgba_of(self, frame) -> np.ndarray:
        """rgb_of with an opaque alpha channel (overlay layers)."""
        rgb = self.rgb_of(frame)
        return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])

    def raw_at(self, t: float, interrupt: Optional[Callable[[], bool]] = None) -> tuple:
        """(av.VideoFrame shown at source time t or None, whether t was reached without an interrupt).

        Only decodes: convert with to_rgb, possibly on another thread, while this decoder moves on.
        """
        with self.lock:
            if self.hw_device is None:
                return self._raw_at(t, interrupt)
            try:
                return self._raw_at(t, interrupt)
            except Exception as e:  # noqa: BLE001 — the graphics card gave up: the CPU takes over
                if self._closing:
                    return None, False
                self._to_software(e)
                return self._raw_at(t, interrupt)

    def _raw_at(self, t: float, interrupt: Optional[Callable[[], bool]]) -> tuple:
        if self._closing:
            return None, False
        if self._last_t is not None and abs(t - self._last_t) < self.frame_dur * 0.5 and self._last_frame is not None:
            return self._last_frame, True
        forward_ok = (self._iter is not None and self._pos_t is not None
                      and self._pos_t - self.frame_dur * 0.5 <= t <= self._pos_t + 2.0)
        if not forward_ok:
            self._seek(t)
        elif t - self._pos_t > 2 * self._skip_margin:
            self._skip_nonref(t - self._skip_margin)
        best = None
        interrupted = False
        try:
            while True:
                frame = self._next_frame if self._next_frame is not None else next(self._iter)
                self._next_frame = None
                ft = (frame.time if frame.time is not None else 0.0) - self.start
                if self._skip_until is not None and ft >= self._skip_until:
                    self._skip_nonref(None)
                if ft + self.frame_dur * 0.5 < t:
                    best = frame
                    if self._closing:
                        return None, False
                    if interrupt is not None and interrupt():
                        interrupted = True
                        break
                    continue
                if ft > t + self.frame_dur * 0.5 and best is not None:
                    self._next_frame = frame
                    break
                best = frame
                break
        except StopIteration:
            self._iter = None
        except av.error.FFmpegError:
            if self.hw_device:
                raise          # the graphics card failed: raw_at hands over to the CPU
            self._iter = None
        if self._skip_until is not None and (self._iter is None or interrupted):
            self._skip_nonref(None)
        if best is None:
            return self._last_frame, False
        bt = (best.time or 0.0) - self.start
        self._pos_t = bt
        self._last_t = bt if interrupted or abs(bt - t) < 1.0 else t
        self._last_frame = best
        return best, not interrupted


def _open_hw(path: str, device: str):
    """av.open on a graphics-card decoder; frames are copied back to system memory (is_hw_owned off)."""
    return av.open(path, hwaccel=HWAccel(device, allow_software_fallback=True))


class DecoderPool:
    """Keeps one open decoder per media file (proxy or original). Safe to use from several threads.

    `gpu` lets big footage decode on the graphics card (the preview's pool; see wants_gpu)."""

    def __init__(self, limit: int = 24, gpu: bool = False):
        self._pool: dict[str, FrameDecoder] = {}
        self._limit = limit
        self._gpu = gpu
        self._lock = threading.Lock()

    def get(self, footage: Footage, use_proxy: bool, max_height: int = 720, slot: int = 0):
        """`slot` > 0 opens extra decoders on the same file, so two positions can be read independently.
        Images (see images.py) get an ImageDecoder, shared by every slot."""
        if footage.is_image:
            from .images import ImageDecoder
            key = f"{footage.path}|{max_height}|image"
            with self._lock:
                dec = self._pool.get(key)
                if dec is not None:
                    return dec
            dec = ImageDecoder(footage.path, max_height)
            with self._lock:
                self._pool[key] = dec
            return dec
        proxy = use_proxy and footage.has_proxy()
        path = footage.proxy_path if proxy else footage.path
        gpu = self._gpu and wants_gpu(footage, proxy)
        key = f"{path}|{max_height}|{slot}" + ("|gpu" if gpu else "")
        evicted = None
        with self._lock:
            dec = self._pool.get(key)
            if dec is None:
                if len(self._pool) >= self._limit:
                    evicted = self._pool.pop(next(iter(self._pool)))
                dec = FrameDecoder(path, 0 if proxy else footage.rotation,
                                   "bt709" if proxy else footage.color_space, max_height, gpu)
                self._pool[key] = dec
            else:
                self._pool[key] = self._pool.pop(key)  # LRU bump
        if evicted is not None:
            evicted.close()
        return dec

    def drop(self, path_prefix: str = "") -> None:
        with self._lock:
            gone = [self._pool.pop(k) for k in list(self._pool) if k.startswith(path_prefix)]
        for dec in gone:
            dec.close()

    def close(self):
        self.drop("")
