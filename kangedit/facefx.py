"""Fun face filters: a big head, a tiny head, big eyes and other funny faces.

Each frame's face is found (faces.py's YuNet detector, with its eye and mouth points) and the picture is warped
around it: a smooth bulge or pinch that fades to nothing at its edge, so the rest of the picture stays as it was.
The viewer warps each frame it shows (`apply_frame`, detections cached per frame); the export warps the clip's
frames the same way before FFmpeg grades and frames them (`render_clip`), following the face through the clip.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

FACE_FX = {
    "": "Off",
    "big_head": "Big head",
    "tiny_head": "Tiny head",
    "big_eyes": "Funny face: big eyes",
    "big_mouth": "Funny face: big mouth",
    "wide": "Funny face: wide face",
    "long": "Funny face: long face",
    "alien": "Funny face: alien (big eyes, tiny mouth)",
}
DEFAULT_AMOUNT = 0.6        # how strong (0..1)
DETECT_LONG = 480           # faces are looked for in a copy at most this big (long side): fast enough to play
HOLD = 0.8                  # seconds a lost face keeps its effect (fading out) before it is dropped


@dataclass(frozen=True)
class Face:
    """A face as parts of the picture: centre, size, and its eyes and mouth (centres)."""
    cx: float
    cy: float
    w: float
    h: float
    eyes: tuple = ()            # ((x, y), (x, y)) left and right eye
    mouth: tuple = ()           # (x, y)

    def mixed(self, other: "Face", k: float) -> "Face":
        """This face moved `k` (0..1) of the way to `other` (smoothing between frames)."""
        def m(a, b):
            return a + (b - a) * k

        eyes = tuple((m(a[0], b[0]), m(a[1], b[1])) for a, b in zip(self.eyes, other.eyes)) \
            if self.eyes and other.eyes else other.eyes
        mouth = (m(self.mouth[0], other.mouth[0]), m(self.mouth[1], other.mouth[1])) \
            if self.mouth and other.mouth else other.mouth
        return Face(m(self.cx, other.cx), m(self.cy, other.cy), m(self.w, other.w), m(self.h, other.h),
                    eyes, mouth)


# ----------------------------------------------------------------------------- finding the face


def _yunet():
    from .faces import _yunet as make
    return make()


def find(rgb: np.ndarray) -> Optional[Face]:
    """The biggest confident face in an RGB picture, or None."""
    h, w = rgb.shape[:2]
    k = min(1.0, DETECT_LONG / float(max(h, w)))
    small = cv2.resize(rgb, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA) \
        if k < 1.0 else rgb
    small = np.ascontiguousarray(small[..., :3])
    sh, sw = small.shape[:2]
    det = _yunet()
    if det is not False:
        det.setInputSize((sw, sh))
        _, rows = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if rows is None or not len(rows):
            return None
        r = max(rows, key=lambda r: float(r[2]) * float(r[3]) * float(r[-1]))
        x, y, fw, fh = (float(v) for v in r[:4])
        pts = [(float(r[4 + 2 * i]) / sw, float(r[5 + 2 * i]) / sh) for i in range(5)]
        # YuNet's points: right eye, left eye (as seen), nose tip, mouth's right and left corners
        mouth = ((pts[3][0] + pts[4][0]) / 2, (pts[3][1] + pts[4][1]) / 2)
        return Face((x + fw / 2) / sw, (y + fh / 2) / sh, fw / sw, fh / sh, (pts[0], pts[1]), mouth)
    from .faces import _detect_haar
    found = _detect_haar(small)
    if not found:
        return None
    cx, cy, fw, fh, _ = max(found, key=lambda f: f[2] * f[3])
    eyes = ((cx - fw * 0.2, cy - fh * 0.12), (cx + fw * 0.2, cy - fh * 0.12))
    return Face(cx, cy, fw, fh, eyes, (cx, cy + fh * 0.25))


# ----------------------------------------------------------------------------- warping
def _warp(img: np.ndarray, cx: float, cy: float, rx: float, ry: float, strength: float,
          axis: str = "both") -> np.ndarray:
    """Bulge (strength > 0) or pinch (< 0) the picture inside the ellipse centred cx, cy (pixels) with radii rx,
    ry, smoothly back to the untouched picture at its edge. axis: "x" / "y" stretch one way only."""
    h, w = img.shape[:2]
    if rx < 2 or ry < 2 or abs(strength) < 1e-3:
        return img
    x0, x1 = max(0, int(cx - rx)), min(w, int(np.ceil(cx + rx)) + 1)
    y0, y1 = max(0, int(cy - ry)), min(h, int(np.ceil(cy + ry)) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return img
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    dx, dy = (gx - cx) / rx, (gy - cy) / ry
    r = np.sqrt(dx * dx + dy * dy)
    inside = r < 1.0
    # Where a pixel at distance r takes its picture from: r * (1 - s (1 - r)^2). At the centre the picture is
    # magnified 1 / (1 - s); at the edge nothing moves, and the change is smooth in between.
    s = float(np.clip(strength, -1.5, 0.85))
    f = np.where(inside, 1.0 - s * (1.0 - r) ** 2, 1.0).astype(np.float32)
    fx = f if axis in ("both", "x") else np.ones_like(f)
    fy = f if axis in ("both", "y") else np.ones_like(f)
    mx = (cx + (gx - cx) * fx).astype(np.float32)
    my = (cy + (gy - cy) * fy).astype(np.float32)
    out = img.copy() if not img.flags.writeable else img
    roi = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[y0:y1, x0:x1] = roi
    return out


def apply(img: np.ndarray, kind: str, face: Optional[Face], amount: float = DEFAULT_AMOUNT) -> np.ndarray:
    """The picture with the face filter `kind` on `face` (amount 0..1)."""
    if not kind or face is None or amount <= 0:
        return img
    h, w = img.shape[:2]
    a = float(np.clip(amount, 0.0, 1.0))
    fw, fh = face.w * w, face.h * h
    cx, cy = face.cx * w, face.cy * h
    img = np.array(img, copy=True)
    if kind == "big_head":       # the whole head (hair and chin too), a little above the face's middle
        return _warp(img, cx, cy - fh * 0.12, fw * 1.25, fh * 1.25, 0.75 * a)
    if kind == "tiny_head":
        return _warp(img, cx, cy - fh * 0.08, fw * 1.2, fh * 1.2, -1.2 * a)
    if kind == "wide":
        return _warp(img, cx, cy, fw * 1.1, fh * 0.9, 0.7 * a, "x")
    if kind == "long":
        return _warp(img, cx, cy, fw * 0.8, fh * 1.15, 0.7 * a, "y")
    eyes = face.eyes or ((face.cx - face.w * 0.2, face.cy - face.h * 0.12),
                         (face.cx + face.w * 0.2, face.cy - face.h * 0.12))
    mouth = face.mouth or (face.cx, face.cy + face.h * 0.25)
    er = max(fw, fh) * 0.24
    if kind in ("big_eyes", "alien"):
        for ex, ey in eyes:
            img = _warp(img, ex * w, ey * h, er, er, 0.8 * a)
    if kind == "big_mouth":
        img = _warp(img, mouth[0] * w, mouth[1] * h, fw * 0.42, fh * 0.28, 0.8 * a)
    if kind == "alien":
        img = _warp(img, mouth[0] * w, mouth[1] * h, fw * 0.35, fh * 0.22, -1.0 * a)
        img = _warp(img, cx, cy - fh * 0.2, fw * 1.0, fh * 0.9, 0.35 * a, "y")
    return img


# ----------------------------------------------------------------------------- preview
_cache: "OrderedDict[tuple, Optional[Face]]" = OrderedDict()
_cache_lock = threading.Lock()
CACHE_SIZE = 2000


def face_at(key: str, t: float, img: np.ndarray) -> Optional[Face]:
    """The face in `img`, the frame of footage `key` at source time t (cached: the viewer shows the same frames
    again and again)."""
    k = (key, round(t, 2), img.shape[0])
    with _cache_lock:
        if k in _cache:
            _cache.move_to_end(k)
            return _cache[k]
    face = find(img)
    with _cache_lock:
        _cache[k] = face
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return face


def apply_frame(img: np.ndarray, kind: str, amount: float, key: str, t: float) -> np.ndarray:
    """The viewer's frame with the clip's face filter."""
    if not kind:
        return img
    try:
        return apply(img, kind, face_at(key, t, img), amount)
    except cv2.error:
        return img


# ----------------------------------------------------------------------------- export
class Tracker:
    """Follows the face from frame to frame: a smoothed position, kept for HOLD seconds when it is lost (its
    effect fading out), so the warp doesn't jump or flicker."""

    def __init__(self, fps: float, every: int = 2):
        self.fps = max(1.0, fps)
        self.every = max(1, every)
        self.face: Optional[Face] = None
        self.lost = 0.0
        self.n = 0

    def step(self, img: np.ndarray) -> tuple[Optional[Face], float]:
        """(face, how much of the effect shows 0..1) for the next frame."""
        if self.n % self.every == 0:
            found = find(img)
            if found is not None:
                self.face = found if self.face is None else self.face.mixed(found, 0.6)
                self.lost = 0.0
            else:
                self.lost += self.every / self.fps
        self.n += 1
        if self.face is None:
            return None, 0.0
        if self.lost > HOLD:
            self.face = None
            return None, 0.0
        return self.face, 1.0 - max(0.0, self.lost / HOLD)


def render_still(path: str, out: str, kind: str, amount: float) -> str:
    """A photo with the face filter, written as a PNG (its transparency kept)."""
    from .images import load
    frames, _, _ = load(path)
    rgba = frames[0]
    face = find(rgba[..., :3])
    img = apply(rgba, kind, face, amount)
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    return out


def render_clip(src: str, out: str, ss: float, dur: float, width: int, height: int, fps: float, matrix: str,
                quarters: int, kind: str, amount: float, cancel: Optional[threading.Event] = None,
                progress: Optional[Callable[[float], None]] = None) -> str:
    """Write `dur` seconds of the video `src` from `ss` with the face filter on every frame, as a near-lossless
    10-bit BT.709 file the export reads in its place. quarters: the clip's quarter turns (faces are found the
    right way up)."""
    from .ffmpeg import ffmpeg_bin, popen
    from .reframe import rotate_quarters
    fps = max(1.0, fps)
    exe = ffmpeg_bin()
    dec = popen([exe, "-v", "error", "-ss", f"{ss:.4f}", "-t", f"{dur:.4f}", "-i", src, "-map", "0:v:0",
                 "-vf", f"scale=in_color_matrix={matrix}:in_range=auto,format=rgb48le", "-r", f"{fps:.4f}",
                 "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = popen([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{width}x{height}",
                 "-r", f"{fps:.4f}", "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "6", "-pix_fmt", "yuv444p10le",
                 "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                 out], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    tracker = Tracker(fps)
    size = width * height * 3 * 2
    total = max(1, int(round(dur * fps)))
    n = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cancelled")
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint16).reshape(height, width, 3)
            up = rotate_quarters(frame, quarters) if quarters else frame
            look = (up >> 8).astype(np.uint8)
            face, k = tracker.step(look)
            if face is not None and k > 0:
                up = apply(up, kind, face, amount * k)
            frame = rotate_quarters(up, -quarters % 4) if quarters else up
            enc.stdin.write(np.ascontiguousarray(frame).tobytes())
            n += 1
            if progress is not None and n % 10 == 0:
                progress(min(1.0, n / total))
    finally:
        try:
            enc.stdin.close()
        except OSError:
            pass
        dec.stdout.close()
        dec.wait()
        enc.wait()
    if enc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("Could not apply the face filter to " + os.path.basename(src))
    return out
