"""Chroma key (green / blue screen) for clips on a video layer: the key colour becomes see-through, so the main
track (or a lower layer) shows behind the subject, with a soft edge and the colour's spill taken off the subject.

The export uses FFmpeg's chromakey and despill filters; the preview does the same arithmetic in NumPy (YUV
distance from the key colour, averaged over each pixel's 3×3 neighbours, BT.601 like FFmpeg's conversion), so
what the preview shows is what the export makes.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

PRESETS = {"green": "#00b140", "blue": "#0047bb"}       # chroma green / blue as painted on screens
SIMILARITY = 0.12          # how far from the key colour a pixel still goes (0..1); FFmpeg's chromakey scale
BLEND = 0.08               # the soft edge beyond that


def _uv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BT.601 limited-range U, V (0..255) of RGB 0..255, as FFmpeg's chromakey works with them."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128.0
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128.0
    return u, v


def parse(color: str) -> Optional[tuple[int, int, int]]:
    """(r, g, b) of "#rrggbb", or None."""
    c = (color or "").strip().lstrip("#")
    if len(c) != 6:
        return None
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return None


def screen(color: str) -> str:
    """"green" or "blue": which spill to take off (the stronger of the two in the key colour)."""
    rgb = parse(color) or (0, 255, 0)
    return "blue" if rgb[2] > rgb[1] else "green"


def ffmpeg_filter(color: str, similarity: float, blend: float, despill: bool) -> str:
    """The filters (ending in a comma) that key an RGBA stream: transparent where it is the key colour."""
    rgb = parse(color)
    if rgb is None:
        return ""
    hexc = "0x%02X%02X%02X" % rgb
    out = (f"format=yuva444p,chromakey=color={hexc}:similarity={max(1e-5, similarity):.4f}:"
           f"blend={max(0.0, blend):.4f},format=rgba,")
    if despill:
        out += f"despill=type={screen(color)},"
    return out


def apply(rgba: np.ndarray, color: str, similarity: float, blend: float, despill: bool) -> np.ndarray:
    """Key an RGBA uint8 picture (the preview): alpha 0 where it is the key colour, a soft edge beyond it, and
    the spill taken off the rest."""
    rgb = parse(color)
    if rgb is None:
        return rgba
    u, v = _uv(rgba[..., :3])
    ku, kv = _uv(np.array([[rgb]], np.uint8))
    diff = np.sqrt(((u - ku[0, 0]) ** 2 + (v - kv[0, 0]) ** 2) / (255.0 * 255.0 * 2.0))
    diff = cv2.blur(diff, (3, 3), borderType=cv2.BORDER_REPLICATE)     # FFmpeg averages the 3x3 neighbours
    if blend > 1e-4:
        a = np.clip((diff - similarity) / blend, 0.0, 1.0)
    else:
        a = (diff > similarity).astype(np.float32)
    out = rgba.copy()
    out[..., 3] = (rgba[..., 3].astype(np.float32) * a + 0.5).astype(np.uint8)
    if despill:
        f = out[..., :3].astype(np.float32)
        if screen(color) == "green":
            f[..., 1] = np.minimum(f[..., 1], 0.5 * (f[..., 0] + f[..., 2]))
        else:
            f[..., 2] = np.minimum(f[..., 2], 0.5 * (f[..., 0] + f[..., 1]))
        out[..., :3] = (f + 0.5).astype(np.uint8)
    return out


def guess(rgb: np.ndarray) -> Optional[str]:
    """The screen colour of a picture (RGB uint8), read from its edges, where a green or blue screen shows: the
    median of the strongly green / blue edge pixels, or None when the edges aren't a screen."""
    h, w = rgb.shape[:2]
    m = max(2, int(min(h, w) * 0.08))
    edge = np.concatenate([rgb[:m].reshape(-1, 3), rgb[-m:].reshape(-1, 3),
                           rgb[:, :m].reshape(-1, 3), rgb[:, -m:].reshape(-1, 3)]).astype(np.int32)
    r, g, b = edge[:, 0], edge[:, 1], edge[:, 2]
    for sel in ((g > r + 30) & (g > b + 30), (b > r + 30) & (b > g + 20)):
        if sel.mean() > 0.4:
            med = np.median(edge[sel], axis=0).astype(int)
            return "#%02x%02x%02x" % tuple(int(x) for x in med)
    return None
