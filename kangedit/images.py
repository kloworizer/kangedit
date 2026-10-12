"""Still images and animated GIFs as footage.

An image has no length of its own: it is footage STILL_LENGTH seconds long, so a clip of it can be trimmed
to any length (DEFAULT_LENGTH when it is first put on the timeline). An animated GIF loops for as long as
its clip lasts. Pictures keep their transparency, so a PNG logo laid on an overlay layer shows the video
around it.
"""

from __future__ import annotations

import functools
import os
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif")
STILL_LENGTH = 3600.0       # source length of an image: any clip length fits in it
DEFAULT_LENGTH = 5.0        # length of a new image clip
MAX_FRAMES = 1000           # animated GIF frames kept in memory


def is_image(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTENSIONS)


@dataclass
class ImageInfo:
    width: int
    height: int
    frames: int             # 1 for a still
    loop: float             # length of one loop of an animated GIF (0 for a still)
    fps: float
    has_alpha: bool
    codec: str


def _decode_av(path: str, max_frames: int = MAX_FRAMES) -> tuple[list[np.ndarray], list[float]]:
    """RGBA frames and their start times, decoded with PyAV (GIF, WebP, PNG, JPEG, BMP, TIFF...)."""
    import av
    frames, times = [], []
    with av.open(path) as c:
        stream = c.streams.video[0]
        for frame in c.decode(stream):
            frames.append(frame.to_ndarray(format="rgba"))
            times.append(float(frame.time) if frame.time is not None else len(times) * 0.1)
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError("no picture in the file")
    t0 = times[0]
    return frames, [t - t0 for t in times]


def _decode_cv(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)   # imread can't open non-ASCII paths on Windows
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("not a readable image")
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


def load(path: str) -> tuple[list[np.ndarray], list[float], float]:
    """(RGBA frames, their start times, loop length). A still is one frame and loop length 0."""
    try:
        frames, times = _decode_av(path)
    except Exception:  # noqa: BLE001 - PyAV can't read it: try OpenCV
        frames, times = [_decode_cv(path)], [0.0]
    if len(frames) == 1:
        return frames, [0.0], 0.0
    step = (times[-1] - times[0]) / max(1, len(times) - 1) if times[-1] > 0 else 0.1
    return frames, times, times[-1] + max(0.02, step)


def probe(path: str) -> ImageInfo:
    frames, times, loop = load(path)
    h, w = frames[0].shape[:2]
    alpha = any(bool((f[..., 3] < 255).any()) for f in frames[:5])
    fps = (len(frames) / loop) if loop > 0 else 25.0
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ImageInfo(w, h, len(frames), loop, fps, alpha, {"jpg": "jpeg", "jfif": "jpeg", "tif": "tiff"}.get(ext, ext))


class ImageDecoder:
    """Decoder interface (raw_at / rgb_of / rgba_of) over a picture held in memory."""

    def __init__(self, path: str, max_height: int = 720):
        frames, self.times, self.loop = load(path)
        h, w = frames[0].shape[:2]
        k = min(1.0, max(2, max_height) / float(min(w, h)))
        if k < 1.0:
            size = (max(2, int(w * k)), max(2, int(h * k)))
            frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
        self.frames = frames
        self.path = path
        self._rgb: dict[int, np.ndarray] = {}
        self.lock = threading.Lock()

    def close(self):
        self.frames = []
        self._rgb.clear()

    def index_at(self, t: float) -> int:
        if self.loop <= 0 or len(self.frames) < 2:
            return 0
        t = t % self.loop
        return max(0, int(np.searchsorted(self.times, t, side="right")) - 1)

    def raw_at(self, t: float, interrupt=None) -> tuple:
        if not self.frames:
            return None, False
        return self.index_at(t), True

    def rgba_of(self, frame: int) -> np.ndarray:
        return self.frames[frame]

    def rgb_of(self, frame: int) -> np.ndarray:
        """The picture over black (a transparent PNG on the main track)."""
        img = self._rgb.get(frame)
        if img is None:
            rgba = self.frames[frame]
            if (rgba[..., 3] < 255).any():
                a = rgba[..., 3:4].astype(np.uint16)
                img = (rgba[..., :3].astype(np.uint16) * a // 255).astype(np.uint8)
            else:
                img = np.ascontiguousarray(rgba[..., :3])
            self._rgb[frame] = img
        return img

    def frame_at(self, t: float, interrupt=None) -> Optional[np.ndarray]:
        frame, _ = self.raw_at(t)
        return None if frame is None else self.rgb_of(frame)


@functools.lru_cache(maxsize=96)
def thumbnail(path: str, height: int = 90) -> Optional[np.ndarray]:
    """A small RGB picture of the image's first frame (transparency shown over a dark checkerboard), or None
    when it can't be read."""
    try:
        frames, _, _ = load(path)
    except Exception:  # noqa: BLE001
        return None
    rgba = frames[0]
    h, w = rgba.shape[:2]
    k = height / float(max(1, h))
    rgba = cv2.resize(rgba, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA)
    th, tw = rgba.shape[:2]
    yy, xx = np.mgrid[0:th, 0:tw]
    checker = np.where(((yy // 8) + (xx // 8)) % 2 == 0, 58, 40).astype(np.float32)[..., None]
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.ascontiguousarray((rgba[..., :3] * a + checker * (1.0 - a)).astype(np.uint8))
