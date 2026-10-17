"""360° footage (Insta360, DJI Osmo 360, GoPro Max, Ricoh Theta...): each clip shows a flat view of the sphere,
looking where its yaw (left / right), pitch (up / down) and roll (tilt) say, with its own field of view, or the
whole sphere as a "tiny planet".

Two kinds of 360 file are read:
* equirectangular (2:1, what the cameras' apps export as a "360° video", and what the cameras record when set to
  do so; YouTube / Facebook 360 too);
* dual fisheye: both lenses side by side, as some cameras record raw (each lens about 190-200°); stitched here
  without the camera's calibration, so seams can show (export a 360° video from the camera's app for the best
  result).

The export uses FFmpeg's v360 filter; the preview remaps the picture with OpenCV using the same projection
arithmetic (checked against v360: the same pixels for flat views, tiny planets and dual fisheye).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

PROJECTIONS = {
    "": "Normal (not 360°)",
    "equirect": "360° equirectangular (2:1)",
    "dfisheye": "360° dual fisheye (both lenses side by side)",
}
LENS_FOV = 195.0          # each lens of a dual-fisheye camera (degrees)
DEFAULT_FOV = 100.0       # horizontal field of view of a flat view
TINY_FOV = 290.0          # how much of the sphere a tiny planet shows
OVERLAY_SIZE = (1920, 1080)     # a 360 clip on a video layer is a view this big (a tiny planet: square)
RAW_360_EXT = {".insv", ".osv", ".360", ".lrv"}


@dataclass(frozen=True)
class View:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov: float = DEFAULT_FOV
    tiny: bool = False


def is_360(footage) -> bool:
    return getattr(footage, "projection", "") in ("equirect", "dfisheye")


def view_of(clip) -> View:
    return View(clip.view_yaw, clip.view_pitch, clip.view_roll, clip.view_fov, clip.view_tiny)


def overlay_dims(clip) -> tuple[int, int]:
    """Size of the picture a 360 clip on a video layer shows (a flat 16:9 view, or a square tiny planet)."""
    w, h = OVERLAY_SIZE
    return (h, h) if clip.view_tiny else (w, h)


def dims(footage, clip=None) -> tuple[int, int]:
    """The picture size of footage as a clip shows it: a 360 file's view, else the file's own size."""
    if is_360(footage) and clip is not None and hasattr(clip, "view_tiny"):
        return overlay_dims(clip)
    return footage.width, footage.height


def detect(path: str, side_projection: str, width: int, height: int) -> str:
    """The projection of a newly imported file: from its spherical metadata (Google's spatial media tags, as the
    cameras and their apps write), else from a 360 camera's raw file type with a 2:1 picture."""
    p = (side_projection or "").lower()
    if "equirect" in p:
        return "equirect"
    if "fisheye" in p:
        return "dfisheye"
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_360_EXT and width and abs(width / max(1, height) - 2.0) < 0.05:
        return "dfisheye"
    return ""


def _vfov(view: View, w: int, h: int) -> float:
    if view.tiny:
        return float(np.degrees(4 * np.arctan(np.tan(np.radians(min(view.fov, 359.0)) / 4) * h / w)))
    return float(np.degrees(2 * np.arctan(np.tan(np.radians(view.fov) / 2) * h / w)))


def _turn(view: View) -> tuple[float, float, float]:
    """(yaw, pitch, roll) given to v360: a tiny planet looks straight down."""
    return view.yaw, (view.pitch - 90.0) if view.tiny else view.pitch, view.roll


def ffmpeg_filter(projection: str, view: View, w: int, h: int, lens_fov: float = LENS_FOV) -> str:
    """The v360 filter turning a 360 picture into this view, w x h."""
    yaw, pitch, roll = _turn(view)
    inp = "e" if projection == "equirect" else f"dfisheye:ih_fov={lens_fov:.2f}:iv_fov={lens_fov:.2f}"
    out = "sg" if view.tiny else "flat"
    hf = min(view.fov, 359.0) if view.tiny else view.fov
    return (f"v360=input={inp}:output={out}:yaw={yaw:.3f}:pitch={pitch:.3f}:roll={roll:.3f}:h_fov={hf:.3f}:"
            f"v_fov={_vfov(view, w, h):.3f}:w={w}:h={h}:interp=cubic")


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz                      # v360's order: yaw, then pitch, then roll


def directions(view: View, w: int, h: int) -> np.ndarray:
    """(h, w, 3) unit directions on the sphere each output pixel looks at (x right, y down, z ahead)."""
    i, j = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    if view.tiny:                            # stereographic: the whole sphere around the nadir
        rx = np.tan(np.radians(min(view.fov, 359.0)) / 4)
        ry = np.tan(np.radians(_vfov(view, w, h)) / 4)
        x, y = ((2 * i + 1) / w - 1) * rx, ((2 * j + 1) / h - 1) * ry
        r = np.hypot(x, y)
        r = np.where(r == 0, 1e-12, r)
        th = np.arctan(r) * 2
        v = np.stack([x / r * np.sin(th), y / r * np.sin(th), np.cos(th)], -1)
    else:
        fx = np.tan(np.radians(view.fov) / 2)
        fy = np.tan(np.radians(_vfov(view, w, h)) / 2)
        v = np.stack([fx * ((2 * i + 1) / w - 1), fy * ((2 * j + 1) / h - 1), np.ones_like(i)], -1)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v @ _rotation(*_turn(view)).T


def source_maps(projection: str, view: View, w: int, h: int, sw: int, sh: int,
                lens_fov: float = LENS_FOV) -> tuple[np.ndarray, np.ndarray]:
    """Where in a sw x sh 360 picture each pixel of the w x h view comes from (cv2.remap maps)."""
    v = directions(view, w, h)
    if projection == "equirect":
        phi = np.arctan2(v[..., 0], v[..., 2])
        th = np.arcsin(np.clip(v[..., 1], -1.0, 1.0))
        u = (0.5 * phi / np.pi + 0.5) * (sw - 1)
        vv = (0.5 * th / (np.pi / 2) + 0.5) * (sh - 1)
    else:                                    # dual fisheye: the front lens on the right half, the back mirrored
        ew, eh = sw / 2.0, float(sh)
        hh = np.hypot(v[..., 0], v[..., 1])
        lh = np.where(hh > 0, hh, 1.0)
        th = np.arccos(np.clip(np.abs(v[..., 2]), 0.0, 1.0)) / np.pi
        rng = lens_fov / 360.0
        u = (0.5 * th * (v[..., 0] / lh) / rng + 0.5) * (ew - 1)
        vv = (0.5 * th * (v[..., 1] / lh) / rng + 0.5) * (eh - 1)
        back = v[..., 2] < 0
        u = np.where(back, ew - u - 1, u) + np.where(back, 0.0, np.ceil(ew))
    return u.astype(np.float32), vv.astype(np.float32)


INTERP = cv2.INTER_LINEAR
_maps: OrderedDict = OrderedDict()
_maps_lock = threading.Lock()


def reproject(img: np.ndarray, projection: str, view: View, w: int, h: int, lens_fov: float = LENS_FOV) -> np.ndarray:
    """The view of a 360 picture (any channel count), w x h (the preview; the maps are kept for the next frame)."""
    sh, sw = img.shape[:2]
    key = (projection, view, w, h, sw, sh, lens_fov)
    with _maps_lock:
        m = _maps.get(key)
        if m is not None:
            _maps.move_to_end(key)
    if m is None:
        m = source_maps(projection, view, w, h, sw, sh, lens_fov)
        with _maps_lock:
            _maps[key] = m
            while len(_maps) > 12:
                _maps.popitem(last=False)
    border = cv2.BORDER_WRAP if projection == "equirect" else cv2.BORDER_CONSTANT
    return cv2.remap(img, m[0], m[1], INTERP, borderMode=border)
