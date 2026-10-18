"""Video thumbnails: the best stills of the edit (lively, sharp, a face, the hook's kind of moment), framed on the
face for the thumbnail's size, with big outlined text, saved as a JPG (YouTube's 1280 x 720, or vertical)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import Footage, Project

SIZES = {
    "16:9": ("YouTube 1280 × 720", 1280, 720),
    "9:16": ("Vertical 1080 × 1920 (Shorts / Reels cover)", 1080, 1920),
    "1:1": ("Square 1080 × 1080", 1080, 1080),
}
COLORS = {"#FFD400": "Yellow", "#FFFFFF": "White", "#FF3B30": "Red", "#30D158": "Green"}
PLACES = {"bottom": "Bottom", "top": "Top", "middle": "Middle"}


@dataclass
class Still:
    footage_id: str
    t: float             # source time
    score: float
    clip_id: str = ""    # the timeline clip it comes from (its grade colours the still)


def candidates(project: Project, analyses: dict, count: int = 6) -> list[Still]:
    """The best stills of the edit, strongest first: in each main-track clip the moment that would make the
    best hook (movement, faces, people enjoying it), sharp and well exposed; one per clip, other files first."""
    from .autoedit import score_samples
    ids = [c.footage_id for c in project.clips if c.footage_id in analyses
           and not project.footage[c.footage_id].is_image]
    scores = score_samples({i: analyses[i] for i in dict.fromkeys(ids)}, project.footage, "hook") if ids else {}
    found: list[Still] = []
    for c in project.clips:
        f = project.footage.get(c.footage_id)
        if f is None:
            continue
        if f.is_image:
            found.append(Still(f.id, 0.0, 0.5, c.id))
            continue
        a, sc = analyses.get(c.footage_id), scores.get(c.footage_id)
        if a is None or sc is None:
            continue
        t = np.asarray(a.times)
        faces = np.clip(np.asarray(a.faces) * 25.0, 0, 1)
        sharp = np.asarray(a.sharp, np.float32)
        sharp = (sharp - sharp.min()) / max(1e-6, float(sharp.max() - sharp.min()))
        val = sc + 0.3 * faces + 0.2 * sharp            # a still: a face and sharpness count for more
        inside = (t >= c.src_in + 0.2) & (t <= c.src_out - 0.2)
        if not inside.any():
            continue
        k = int(np.argmax(np.where(inside, val, -np.inf)))
        found.append(Still(c.footage_id, float(t[k]), float(val[k]), c.id))
    found.sort(key=lambda s: -s.score)
    out: list[Still] = []
    for s in found:                                    # other files first, then the rest
        if all(o.footage_id != s.footage_id for o in out):
            out.append(s)
    out += [s for s in found if s not in out]
    return out[:count]


def frame(f: Footage, t: float, grade=None, use_proxy: bool = False, height: int = 1080) -> Optional[np.ndarray]:
    """The still at source time t as RGB uint8, its colours as the edit shows them (log conversion, grade)."""
    from .color import Grade, apply_grade, needs_color
    if f.is_image:
        from .images import load
        img = np.ascontiguousarray(load(f.path)[0][0][..., :3])
    else:
        from .decoder import FrameDecoder
        proxy = use_proxy and f.has_proxy()
        dec = FrameDecoder(f.proxy_path if proxy else f.path, 0 if proxy else f.rotation,
                           "bt709" if proxy else f.color_space, height)
        try:
            img = dec.frame_at(t)
        finally:
            dec.close()
    if img is None:
        return None
    g = grade or Grade()
    if needs_color(g, f.log_profile):
        img = (np.clip(apply_grade(img.astype(np.float32) / 255.0, g, f.log_profile, f.tonemap,
                                   f.color_range == "pc"), 0, 1) * 255 + 0.5).astype(np.uint8)
    return img


def framed(img: np.ndarray, w: int, h: int, face: Optional[tuple] = None) -> np.ndarray:
    """The still cropped to fill w x h, the face (centre x, y as parts of the picture) a bit above the middle."""
    from .reframe import place, render
    sh, sw = img.shape[:2]
    ox = oy = 0.0
    if face is not None:
        p = place(sw, sh, w, h, "fill")

        def offset(centre: float, full: int, win: int, at: float) -> float:
            if full <= win:
                return 0.0
            return float(np.clip(2.0 * (centre * full - at * win) / (full - win) - 1.0, -1.0, 1.0))

        ox, oy = offset(face[0], p.scaled_w, p.crop_w, 0.5), offset(face[1], p.scaled_h, p.crop_h, 0.45)
    return render(img, w, h, "fill", 1.0, ox, oy)


def pop(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Livelier colours and contrast, as thumbnails usually have."""
    import cv2
    if amount <= 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.35 * amount), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = (out - 128.0) * (1 + 0.15 * amount) + 128.0 + 6 * amount
    return np.clip(out, 0, 255).astype(np.uint8)
