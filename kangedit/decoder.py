"""Frame-accurate preview decoding with PyAV."""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional

import av
import numpy as np
from av.video.reformatter import VideoReformatter

from .models import Footage


_local = threading.local()


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

    def __init__(self, path: str, rotation: int = 0, color_space: str = "", max_height: int = 720):
        self.path = path
        self.rotation = rotation % 360
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.tb = float(self.stream.time_base) if self.stream.time_base else 1 / 90000
        self.start = float(self.stream.start_time * self.stream.time_base) if self.stream.start_time is not None else 0.0
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
        self._skip_margin = ((os.cpu_count() or 4) + 4) * self.frame_dur
        self._skip_until: Optional[float] = None
        self.lock = threading.Lock()

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


class DecoderPool:
    """Keeps one open decoder per media file (proxy or original). Safe to use from several threads."""

    def __init__(self, limit: int = 24):
        self._pool: dict[str, FrameDecoder] = {}
        self._limit = limit
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
        key = f"{path}|{max_height}|{slot}"
        evicted = None
        with self._lock:
            dec = self._pool.get(key)
            if dec is None:
                if len(self._pool) >= self._limit:
                    evicted = self._pool.pop(next(iter(self._pool)))
                dec = FrameDecoder(path, 0 if proxy else footage.rotation,
                                   "bt709" if proxy else footage.color_space, max_height)
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
