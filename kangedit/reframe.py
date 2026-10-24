"""Reframing geometry shared by the viewer and the exporter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class Placement:
    scaled_w: int      # size of the scaled source
    scaled_h: int
    crop_x: int        # crop window inside the scaled source
    crop_y: int
    crop_w: int
    crop_h: int
    pad_x: int         # where the crop lands on the output canvas
    pad_y: int


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def place(sw: int, sh: int, tw: int, th: int, mode: str, zoom: float = 1.0,
          ox: float = 0.0, oy: float = 0.0) -> Placement:
    """Fit a sw x sh source into a tw x th frame. mode: fill (crop) | fit/blur (letterbox)."""
    if mode == "fill":
        s = max(tw / sw, th / sh)
    else:
        s = min(tw / sw, th / sh)
    s *= max(1.0, zoom)
    w, h = _even(sw * s), _even(sh * s)
    cw, ch = min(w, tw), min(h, th)
    cx = int(round((w - cw) / 2.0 * (1.0 + max(-1.0, min(1.0, ox)))))
    cy = int(round((h - ch) / 2.0 * (1.0 + max(-1.0, min(1.0, oy)))))
    cx = max(0, min(w - cw, cx))
    cy = max(0, min(h - ch, cy))
    return Placement(w, h, cx, cy, cw, ch, (tw - cw) // 2, (th - ch) // 2)


def window_matrix(sw: int, sh: int, p: Placement, window: tuple) -> np.ndarray:
    """The 2 x 3 affine map from a sw x sh source to its framed picture (p.crop_w x p.crop_h) showing a motion
    window (centre x, centre y as parts of the framed picture, zoom, turn in degrees clockwise; see motion.py)."""
    cx, cy, z, turn = window
    a = np.radians(turn)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * z
    lin = r @ np.diag([p.scaled_w / sw, p.scaled_h / sh])
    centre = np.array([p.crop_x + cx * p.crop_w, p.crop_y + cy * p.crop_h])
    t = np.array([p.crop_w / 2.0, p.crop_h / 2.0]) - r @ centre
    return np.hstack([lin, t[:, None]]).astype(np.float64)


def is_still(window: Optional[tuple]) -> bool:
    return window is None or (abs(window[2] - 1.0) < 1e-6 and abs(window[3]) < 1e-6
                              and abs(window[0] - 0.5) < 1e-6 and abs(window[1] - 0.5) < 1e-6)


def render(img: np.ndarray, tw: int, th: int, mode: str, zoom: float = 1.0,
           ox: float = 0.0, oy: float = 0.0,
           pixel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
           window: Optional[tuple] = None) -> np.ndarray:
    """Reframe an RGB image onto a tw x th canvas (viewer path). `window`: the part of the framed picture
    shown, enlarged to it (a dynamic zoom or stabilization, see motion.py).

    `pixel_fn` (a per-pixel color transform) is applied only to the visible pixels, before padding.
    """
    fn = pixel_fn or (lambda a: a)
    sh, sw = img.shape[:2]
    p = place(sw, sh, tw, th, mode, zoom, ox, oy)
    if not is_still(window):
        crop = fn(cv2.warpAffine(img, window_matrix(sw, sh, p, window), (p.crop_w, p.crop_h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    else:
        scaled = cv2.resize(img, (p.scaled_w, p.scaled_h), interpolation=cv2.INTER_AREA
                            if p.scaled_w < sw else cv2.INTER_LINEAR)
        crop = fn(np.ascontiguousarray(scaled[p.crop_y:p.crop_y + p.crop_h, p.crop_x:p.crop_x + p.crop_w]))
    if p.crop_w == tw and p.crop_h == th:
        return np.ascontiguousarray(crop)
    if mode == "blur":
        bp = place(sw, sh, tw, th, "fill")
        bg = cv2.resize(img, (max(2, bp.scaled_w // 8), max(2, bp.scaled_h // 8)), interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(fn(bg), (0, 0), 3)
        bg = cv2.resize(bg, (bp.scaled_w, bp.scaled_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.ascontiguousarray(bg[bp.crop_y:bp.crop_y + th, bp.crop_x:bp.crop_x + tw])
        canvas = (canvas * 0.7).astype(np.uint8)
    else:
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[p.pad_y:p.pad_y + p.crop_h, p.pad_x:p.pad_x + p.crop_w] = crop
    return canvas


def _moving(p: Placement, tag: str, opts: str, to: str, ss: int, rotates: bool, fill: str = "black",
            cut: tuple = (0, 0)) -> str:
    """Scale and crop whose size and position follow a motion command file per frame (see motion.export_plan):
    the crop is cut ss times larger and scaled down to the framed picture; turned when `rotates`, from a window
    `cut` (supersampled pixels) larger than the picture, whose middle is kept (the corners stay picture)."""
    cw, ch = cut if cut[0] and cut[1] else (p.crop_w * ss, p.crop_h * ss)
    fg = (f"scale@{tag}s={max(p.scaled_w * ss, cw)}:{max(p.scaled_h * ss, ch)}:flags=lanczos{opts}{to},"
          f"crop@{tag}c={cw}:{ch}:{p.crop_x * ss}:{p.crop_y * ss}")
    if ss > 1:
        fg += f",scale={cw // ss}:{ch // ss}:flags=lanczos"
    if rotates:
        fg += f",rotate@{tag}r=0:ow={p.crop_w}:oh={p.crop_h}:c={fill}"
    return fg


def _cut(motion: tuple) -> tuple:
    return motion[3] if len(motion) > 3 else (0, 0)


def ffmpeg_moving_layer(p: Placement, tag: str, motion: tuple) -> str:
    """A layer's picture scaled and cropped as Placement p, moving inside it by a command file per frame:
    motion (command file, supersampling, turns), see motion.export_plan. Ends with a comma."""
    return (f"sendcmd=f={motion[0]}," + _moving(p, tag, "", "", motion[1], motion[2], "none", _cut(motion))
            + ",")


def ffmpeg_filter(sw: int, sh: int, tw: int, th: int, mode: str, zoom: float, ox: float, oy: float,
                  inp: str, out: str, tag: str, convert: str = "", motion: Optional[tuple] = None) -> str:
    """Filtergraph snippet reframing label `inp` into label `out` (export path). convert: scale options and the
    pixel format to convert to in the same pass (e.g. ":in_color_matrix=bt709:in_range=auto", "gbrp16le" as
    "opts|fmt"): one swscale pass resizes and converts (not with the blurred background, which mixes two).
    motion: (command file, supersampling, turns) of a clip moving inside its picture (see motion.export_plan)."""
    p = place(sw, sh, tw, th, mode, zoom, ox, oy)
    opts, fmt = convert.split("|") if convert else ("", "")
    to = f",format={fmt}" if fmt else ""
    fg = (f"scale={p.scaled_w}:{p.scaled_h}:flags=lanczos{opts}{to},"
          f"crop={p.crop_w}:{p.crop_h}:{p.crop_x}:{p.crop_y}")
    cmds = ""
    if motion:
        cmds = f"sendcmd=f={motion[0]},"
        fg = _moving(p, tag, opts, to, motion[1], motion[2], cut=_cut(motion))
    if p.crop_w == tw and p.crop_h == th:
        return f"[{inp}]{cmds}{fg},setsar=1[{out}]"
    if mode == "blur":
        fg = (_moving(p, tag, "", "", motion[1], motion[2], cut=_cut(motion)) if motion else
              f"scale={p.scaled_w}:{p.scaled_h}:flags=lanczos,"
              f"crop={p.crop_w}:{p.crop_h}:{p.crop_x}:{p.crop_y}")
        bp = place(sw, sh, tw, th, "fill")
        return (
            f"[{inp}]{cmds}split=2[{tag}bgi][{tag}fgi];"
            f"[{tag}bgi]scale={_even(bp.scaled_w / 8)}:{_even(bp.scaled_h / 8)}:flags=area,"
            f"gblur=sigma=3,scale={bp.scaled_w}:{bp.scaled_h}:flags=bilinear,"
            f"crop={tw}:{th}:{bp.crop_x}:{bp.crop_y},colorlevels=romax=0.7:gomax=0.7:bomax=0.7[{tag}bg];"
            f"[{tag}fgi]{fg}[{tag}fg];"
            f"[{tag}bg][{tag}fg]overlay={p.pad_x}:{p.pad_y},setsar=1[{out}]"
        )
    return f"[{inp}]{cmds}{fg},pad={tw}:{th}:{p.pad_x}:{p.pad_y}:black,setsar=1[{out}]"


# ----------------------------------------------------------------------------- rotation
def split_rotation(deg: float) -> tuple[int, float]:
    """A clockwise rotation as (quarter turns 0..3, remaining angle within ±45°).

    Quarter turns rotate the source before it is fitted into the frame (a sideways phone clip fills the
    frame upright); the remaining angle tilts the framed picture about the frame's centre."""
    q = int(round(deg / 90.0))
    return q % 4, float(deg - q * 90.0)


def rotate_quarters(img: np.ndarray, q: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, -q)) if q % 4 else img


def quarter_dims(sw: int, sh: int, q: int) -> tuple[int, int]:
    return (sh, sw) if q % 2 else (sw, sh)


def tilt(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a framed picture clockwise about its centre, keeping its size (black corners)."""
    if abs(deg) < 0.01:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def ffmpeg_quarters(q: int) -> str:
    """Filter turning the source q quarter turns clockwise ("" for none)."""
    return {1: "transpose=1", 2: "hflip,vflip", 3: "transpose=2"}.get(q % 4, "")


def ffmpeg_tilt(deg: float, fill: str = "black") -> str:
    if abs(deg) < 0.01:
        return ""
    return f"rotate={np.radians(deg):.6f}:ow=iw:oh=ih:c={fill}"


# ----------------------------------------------------------------------------- overlay layers
def overlay_place(sw: int, sh: int, tw: int, th: int, scale: float, box_w: float = 0.0, box_h: float = 0.0,
                  pan_x: float = 0.0, pan_y: float = 0.0, zoom: float = 1.0) -> Placement:
    """How an overlay picture is scaled and cropped for a tw x th frame: fitted inside the frame and scaled,
    or (with a box) scaled to fill a box_w x box_h part of the frame (times scale) and cropped to it (zoom > 1:
    to a smaller part of the picture, such as a facecam corner)."""
    if box_w > 0 and box_h > 0:
        bw, bh = _even(box_w * scale * tw), _even(box_h * scale * th)
        return place(sw, sh, bw, bh, "fill", zoom, pan_x, pan_y)
    s = min(tw / max(1, sw), th / max(1, sh)) * max(0.01, scale)
    w, h = _even(sw * s), _even(sh * s)
    return Placement(w, h, 0, 0, w, h, 0, 0)


def overlay_size(sw: int, sh: int, tw: int, th: int, scale: float, box_w: float = 0.0,
                 box_h: float = 0.0) -> tuple[int, int]:
    """Visible size of an overlay in a tw x th frame (see overlay_place)."""
    p = overlay_place(sw, sh, tw, th, scale, box_w, box_h)
    return p.crop_w, p.crop_h


def overlay_layer(rgba: np.ndarray, tw: int, th: int, scale: float, rotation: float, box_w: float = 0.0,
                  box_h: float = 0.0, pan_x: float = 0.0, pan_y: float = 0.0,
                  window: Optional[tuple] = None, zoom: float = 1.0) -> np.ndarray:
    """The overlay picture (RGBA) scaled (and cropped to its box) for a tw x th frame and rotated clockwise on
    a transparent canvas just big enough for it. window: the part of its picture shown (see render)."""
    sh, sw = rgba.shape[:2]
    p = overlay_place(sw, sh, tw, th, scale, box_w, box_h, pan_x, pan_y, zoom)
    if not is_still(window):
        img = cv2.warpAffine(rgba, window_matrix(sw, sh, p, window), (p.crop_w, p.crop_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        img = cv2.resize(rgba, (p.scaled_w, p.scaled_h), interpolation=cv2.INTER_AREA if p.scaled_w < sw
                         else cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img[p.crop_y:p.crop_y + p.crop_h, p.crop_x:p.crop_x + p.crop_w])
    w, h = p.crop_w, p.crop_h
    if abs(rotation) < 0.01:
        return img
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -rotation, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    rw, rh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    m[0, 2] += rw / 2.0 - w / 2.0
    m[1, 2] += rh / 2.0 - h / 2.0
    return cv2.warpAffine(img, m, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def composite(canvas: np.ndarray, layer: np.ndarray, cx: float, cy: float, alpha: float) -> None:
    """Blend an RGBA (or BGRA) layer onto a 3- or 4-channel canvas of the same channel order, centred at
    (cx, cy) in pixels, in place. alpha: extra opacity 0..1."""
    th, tw = canvas.shape[:2]
    lh, lw = layer.shape[:2]
    x0, y0 = int(round(cx - lw / 2.0)), int(round(cy - lh / 2.0))
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(tw, x0 + lw), min(th, y0 + lh)
    if ax1 <= ax0 or ay1 <= ay0 or alpha <= 0:
        return
    src = layer[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    a = src[..., 3:4].astype(np.float32) * (alpha / 255.0)
    dst = canvas[ay0:ay1, ax0:ax1, :3]
    dst[:] = (src[..., :3].astype(np.float32) * a + dst.astype(np.float32) * (1.0 - a) + 0.5).astype(np.uint8)


def box_rect(tw: int, th: int, box: tuple) -> tuple[int, int, int, int]:
    """(x, y, width, height) in pixels of a (centre x, centre y, width, height) box given in parts of a tw x th
    frame; even sizes, kept inside the frame."""
    cx, cy, bw, bh = box
    w, h = min(tw, _even(bw * tw)), min(th, _even(bh * th))
    x = int(round(min(max(0.0, cx * tw - w / 2), tw - w)))
    y = int(round(min(max(0.0, cy * th - h / 2), th - h)))
    return x, y, w, h


def pan_for_point(cx: float, cy: float, sw: int, sh: int, tw: int, th: int, box_w: float, box_h: float,
                  scale: float = 1.0, at_y: float = 0.42) -> tuple[float, float]:
    """pan_x / pan_y that put the point (cx, cy, parts of the picture) at the middle of a box crop (a face a
    bit above the middle)."""
    p = overlay_place(sw, sh, tw, th, scale, box_w, box_h)

    def offset(centre: float, full: int, win: int, at: float) -> float:
        if full <= win:
            return 0.0
        return float(np.clip(2.0 * (centre * full - at * win) / (full - win) - 1.0, -1.0, 1.0))

    return offset(cx, p.scaled_w, p.crop_w, 0.5), offset(cy, p.scaled_h, p.crop_h, at_y)


# ----------------------------------------------------------------------------- zoom to a face
FACE_SIZES = {"medium": 0.22, "close": 0.34, "extreme": 0.5}   # face height as a part of the frame height


def find_face(frames: list[np.ndarray]) -> Optional[tuple[float, float, float]]:
    """The main face over a few RGB frames of a clip: (centre x, centre y, height), as parts of the frame
    (see faces.main_face)."""
    from .faces import main_face
    return main_face(frames)


def face_framing(face: tuple[float, float, float], sw: int, sh: int, tw: int, th: int, mode: str,
                 size: float) -> tuple[float, float, float]:
    """(zoom, offset x, offset y) that make the face `size` of the frame height, eyes a bit above centre."""
    fx, fy, fh = face
    base = max(tw / sw, th / sh) if mode == "fill" else min(tw / sw, th / sh)
    zoom = float(np.clip(size * th / max(1e-6, fh * sh * base), 1.0, 4.0))
    p = place(sw, sh, tw, th, mode, zoom)

    def offset(centre: float, full: int, win: int, at: float) -> float:
        if full <= win:
            return 0.0
        start = centre * full - at * win             # crop start that puts the face at `at` of the window
        return float(np.clip(2.0 * start / (full - win) - 1.0, -1.0, 1.0))

    return (zoom, offset(fx, p.scaled_w, p.crop_w, 0.5), offset(fy, p.scaled_h, p.crop_h, 0.42))
