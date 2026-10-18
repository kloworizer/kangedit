"""Stabilization: how the camera shook in a clip, measured once from the footage, and the steadier path its
picture follows instead.

The camera's movement is tracked frame to frame (corners of the picture followed with optical flow: a shift
and a turn). Smoothing that path (more with a higher strength) gives the movement that was meant, a pan or a
walk; the difference is the shake. Each frame then shows a window of the picture that follows the shake, so
what it shows holds still, zoomed in just enough that the window never runs off the picture (see motion.py).
The measured movement is kept per file in the cache (stabilize/), for the ranges measured so far."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .config import cache_dir
from .ffmpeg import Cancelled
from .models import Footage

TRACK_HEIGHT = 270          # frames are tracked this small: plenty for the camera's movement, and fast
HANDLE = 2.0                # seconds measured past each end of a clip (transitions, a little slipping)
MAX_SHIFT = (0.03, 0.10)    # most the window follows the shake, as a part of the picture: strength 0..1
MAX_TURN = (1.0, 4.0)       # and turns with it, degrees
SIGMA = (0.25, 3.0)         # seconds of smoothing of the camera's path: strength 0..1


def _key(f: Footage) -> str:
    try:
        st = os.stat(f.path)
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        stamp = ""
    # (v2: measured from every frame; v1 missed every other one in a clip that doesn't start the file)
    return hashlib.sha1(f"v2|{os.path.abspath(f.path)}|{stamp}".encode()).hexdigest()[:20]


def _cache_file(f: Footage):
    return cache_dir("stabilize") / f"{_key(f)}.npz"


@dataclass
class Track:
    """A file's measured camera movement: per frame (source time t) the move from the frame before, as a part
    of the picture's width (dx) and height (dy), and the turn in radians (clockwise), for the `ranges` measured."""
    t: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    da: np.ndarray
    ranges: list = field(default_factory=list)      # [(t0, t1)] measured, merged

    def covers(self, t0: float, t1: float) -> bool:
        return any(a <= t0 + 0.05 and t1 - 0.05 <= b for a, b in self.ranges)


_lock = threading.Lock()
_tracks: dict[str, Track] = {}


def load(f: Footage) -> Optional[Track]:
    k = _key(f)
    with _lock:
        if k in _tracks:
            return _tracks[k]
    path = _cache_file(f)
    if not path.exists():
        return None
    try:
        z = np.load(path)
        tr = Track(z["t"], z["dx"], z["dy"], z["da"], [tuple(map(float, r)) for r in z["ranges"]])
    except Exception:  # noqa: BLE001 - a broken cache file is measured again
        return None
    with _lock:
        _tracks[k] = tr
    return tr


def _save(f: Footage, tr: Track) -> None:
    with _lock:
        _tracks[_key(f)] = tr
        _corrections.clear()
    try:
        np.savez(_cache_file(f), t=tr.t, dx=tr.dx, dy=tr.dy, da=tr.da,
                 ranges=np.asarray(tr.ranges, np.float64).reshape(-1, 2))
    except OSError:
        pass


def _merge(ranges: list) -> list:
    out: list = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 0.1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def measured(f: Footage, t0: float, t1: float) -> bool:
    tr = load(f)
    return tr is not None and tr.covers(max(0.0, t0), min(f.duration, t1))


def missing(project) -> list[tuple[Footage, float, float]]:
    """(footage, from, to) of the stabilized clips (main track and video layers) whose camera movement isn't
    measured yet."""
    from . import spherical
    todo = []
    for c in list(project.clips) + list(project.overlays):
        f = project.footage.get(c.footage_id)
        if c.stabilize > 0 and f is not None and not f.is_image and not spherical.is_360(f):
            t0, t1 = c.src_in - HANDLE, c.src_out + HANDLE
            if not measured(f, t0, t1) and (f, t0, t1) not in todo:
                todo.append((f, t0, t1))
    return todo


def frame_motion(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float, float]:
    """How the picture moved from grey frame `prev` to `cur`: (dx, dy) in pixels and the turn (radians,
    clockwise), about the picture's centre. Zero when nothing could be followed."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=12, blockSize=3)
    if pts is None or len(pts) < 8:
        return 0.0, 0.0, 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1) == 1
    if ok.sum() < 8:
        return 0.0, 0.0, 0.0
    h, w = prev.shape[:2]
    c = np.array([w / 2.0, h / 2.0], np.float32)
    m, _ = cv2.estimateAffinePartial2D(pts[ok].reshape(-1, 2) - c, nxt[ok].reshape(-1, 2) - c,
                                       method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if m is None:
        return 0.0, 0.0, 0.0
    return float(m[0, 2]), float(m[1, 2]), float(np.arctan2(m[1, 0], m[0, 0]))


def analyze(f: Footage, t0: float, t1: float, use_proxy: bool = True,
            progress: Optional[Callable[[float], None]] = None,
            cancel: Optional[threading.Event] = None) -> Track:
    """Measure the camera's movement in source time t0..t1 of footage f (added to what is measured)."""
    from .decoder import FrameDecoder
    t0, t1 = max(0.0, t0), min(f.duration, t1)
    old = load(f)
    if old is not None and old.covers(t0, t1):
        return old
    proxy = use_proxy and f.has_proxy()
    dec = FrameDecoder(f.proxy_path if proxy else f.path, 0 if proxy else f.rotation,
                       "bt709" if proxy else f.color_space, TRACK_HEIGHT)
    ts, dxs, dys, das = [], [], [], []
    try:
        dec._seek(t0)
        prev = None
        for frame in dec._iter:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            t = float(frame.time or 0.0) - dec.start
            if dec._skip_until is not None and t >= dec._skip_until:
                dec._skip_nonref(None)      # caught up with the seek: every frame from here (each one's shake)
            if t < t0 - dec.frame_dur:
                continue
            if t > t1 + dec.frame_dur:
                break
            grey = cv2.cvtColor(dec.to_rgb(frame), cv2.COLOR_RGB2GRAY)
            if prev is not None:
                x, y, a = frame_motion(prev, grey)
                h, w = grey.shape[:2]
                ts.append(t)
                dxs.append(x / w)
                dys.append(y / h)
                das.append(a)
            prev = grey
            if progress is not None and t1 > t0:
                progress(min(1.0, (t - t0) / (t1 - t0)))
    finally:
        dec.close()
    t = np.asarray(ts, np.float64)
    new = (t, np.asarray(dxs, np.float32), np.asarray(dys, np.float32), np.asarray(das, np.float32))
    if old is not None:                         # merged with what was measured before (this range replaces it)
        keep = ~((old.t >= t0) & (old.t <= t1))
        new = tuple(np.concatenate([o[keep], n]) for o, n in zip((old.t, old.dx, old.dy, old.da), new))
        order = np.argsort(new[0], kind="stable")
        new = tuple(a[order] for a in new)
    tr = Track(*new, ranges=_merge((old.ranges if old else []) + [(t0, t1)]))
    _save(f, tr)
    return tr


@dataclass(frozen=True)
class Correction:
    """The shake to follow in a clip's source range: at source time t the window moves (sx, sy) parts of the
    picture and turns sa degrees (clockwise) with it; `zoom` keeps the window inside the picture."""
    t: np.ndarray = field(compare=False, hash=False)
    sx: np.ndarray = field(compare=False, hash=False)
    sy: np.ndarray = field(compare=False, hash=False)
    sa: np.ndarray = field(compare=False, hash=False)
    zoom: float = 1.0
    key: tuple = ()

    def at(self, t: float) -> tuple[float, float, float]:
        if len(self.t) == 0:
            return 0.0, 0.0, 0.0
        return (float(np.interp(t, self.t, self.sx)), float(np.interp(t, self.t, self.sy)),
                float(np.interp(t, self.t, self.sa)))


_corrections: dict[tuple, Correction] = {}


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if len(x) < 3 or sigma <= 0:
        return x.copy()
    r = int(min(len(x) - 1, np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="reflect"), k, mode="valid")


def lerp(pair: tuple[float, float], s: float) -> float:
    return pair[0] + (pair[1] - pair[0]) * float(np.clip(s, 0.0, 1.0))


def correction(f: Footage, t0: float, t1: float, strength: float, aspect: float = 16 / 9) -> Optional[Correction]:
    """The shake to follow for source range t0..t1 of footage f at `strength` (0..1), or None while that range
    isn't measured. aspect: the shown picture's width / height (a turn needs more zoom on a wide picture)."""
    if strength <= 0:
        return None
    t0, t1 = max(0.0, t0), min(f.duration, t1)
    tr = load(f)
    if tr is None or not tr.covers(t0, t1):
        return None
    key = (_key(f), round(t0, 3), round(t1, 3), round(strength, 3), round(aspect, 3))
    with _lock:
        if key in _corrections:
            return _corrections[key]
    sel = (tr.t >= t0 - 0.05) & (tr.t <= t1 + 0.05)
    t = tr.t[sel]
    if len(t) < 3:
        return None
    fps = 1.0 / max(1e-3, float(np.median(np.diff(t))))
    sigma = lerp(SIGMA, strength) * fps
    paths = [np.cumsum(a[sel].astype(np.float64)) for a in (tr.dx, tr.dy, tr.da)]
    shake = [p - _smooth(p, sigma) for p in paths]
    lim_s, lim_a = lerp(MAX_SHIFT, strength), np.radians(lerp(MAX_TURN, strength))
    # The window follows most of the shake (the odd jolt bigger than the rest is followed only partly).
    sx, sy = (np.clip(s, -lim_s, lim_s) for s in shake[:2])
    sa = np.clip(shake[2], -lim_a, lim_a)
    reach = max(float(np.percentile(np.abs(sx), 99)), float(np.percentile(np.abs(sy), 99)), 1e-4)
    turn = float(np.percentile(np.abs(sa), 99))
    sx, sy = np.clip(sx, -reach, reach), np.clip(sy, -reach, reach)
    sa = np.clip(sa, -turn, turn)
    # Zoom so a window moved `reach` and turned `turn` stays inside the picture.
    wide = max(aspect, 1.0 / max(1e-3, aspect))
    zoom = (1.0 / max(0.5, 1.0 - 2.0 * reach)) * (abs(np.cos(turn)) + wide * abs(np.sin(turn)))
    c = Correction(t, sx.astype(np.float32), sy.astype(np.float32), np.degrees(sa).astype(np.float32),
                   float(min(zoom, 1.5)), key)
    with _lock:
        _corrections[key] = c
    return c
