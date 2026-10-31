"""Movement inside a clip's framed picture: a dynamic zoom (a slow zoom in or out, or a pan across the picture,
as DaVinci Resolve's Dynamic Zoom) and stabilization (stabilize.py). Both choose, frame by frame, a window of
the clip's picture as it is framed for the export ratio (after its zoom, pan and crop), shown enlarged to the
picture's size: a centre and a zoom (1 = the whole picture), and a turn. The viewer warps the picture to it
(reframe.render) and the export drives FFmpeg's scale / crop / rotate with a command per frame (sendcmd)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MOTIONS = {
    "": "Off",
    "in": "Zoom in",
    "out": "Zoom out",
    "left": "Pan left",
    "right": "Pan right",
    "up": "Pan up",
    "down": "Pan down",
    "in_left": "Zoom in, pan left",
    "in_right": "Zoom in, pan right",
    "out_left": "Zoom out, pan left",
    "out_right": "Zoom out, pan right",
    "face": "Zoom in on the face",
    "face_out": "Zoom out from the face",
    # Quick effects: timed in seconds (not spread over the clip), at least FAST_AMOUNT of zoom.
    "punch_in": "Fast zoom in (punch in)",
    "punch_out": "Fast zoom out",
    "pulse": "Zoom pulse (in and back out)",
    "crash": "Crash zoom at the end",
    "heartbeat": "Heartbeat (two quick pulses, repeating)",
    "shake": "Camera shake",
    "wobble": "Drunk wobble",
}
FACE_KINDS = ("face", "face_out")
FAST_KINDS = ("punch_in", "punch_out", "pulse", "crash", "heartbeat", "shake", "wobble")
FAST_TIME = 0.25           # seconds a fast zoom takes
FAST_AMOUNT = 0.3          # least zoom of the fast zooms (30 %): they are meant to be seen
EASES = {
    "both": "Ease in and out",
    "in": "Ease in",
    "out": "Ease out",
    "linear": "Linear",
}
DEFAULT_AMOUNT = 0.12      # how much a dynamic zoom zooms (12 %): slight, as most edits use it
MAX_AMOUNT = 0.6


def ease(u: float, kind: str) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    if kind == "linear":
        return u
    if kind == "in":           # starts slowly
        return u * u
    if kind == "out":          # ends slowly
        return 1.0 - (1.0 - u) ** 2
    return u * u * (3.0 - 2.0 * u)


def _ends(kind: str, amount: float, target: tuple = (0.5, 0.5)) -> tuple[tuple[float, float, float],
                                                                         tuple[float, float, float]]:
    """Start and end windows (centre x, centre y, zoom) of a dynamic zoom. A centre of 0 / 1 means against the
    left / top or right / bottom edge (as far as the zoom leaves room). target: where "face" zooms to."""
    z = 1.0 + max(0.0, amount)
    mid = (0.5, 0.5, 1.0)
    table = {
        "in": (mid, (0.5, 0.5, z)),
        "face": (mid, (target[0], target[1], z)),
        "left": ((1.0, 0.5, z), (0.0, 0.5, z)),
        "right": ((0.0, 0.5, z), (1.0, 0.5, z)),
        "up": ((0.5, 1.0, z), (0.5, 0.0, z)),
        "down": ((0.5, 0.0, z), (0.5, 1.0, z)),
        "in_left": (mid, (0.0, 0.5, z)),
        "in_right": (mid, (1.0, 0.5, z)),
    }
    if kind in table:
        return table[kind]
    if kind == "face_out":
        a, b = _ends("face", amount, target)
        return b, a
    if kind.startswith("out"):         # a zoom in played backwards (panning the other way)
        a, b = _ends("in" + {"_left": "_right", "_right": "_left"}.get(kind[3:], kind[3:]), amount)
        return b, a
    return mid, mid


def _fit(c: float, z: float) -> float:
    """A window centre kept so a window 1/z wide stays inside the picture."""
    half = 0.5 / max(1.0, z)
    return float(np.clip(c, half, 1.0 - half))


def _bump(t: float, start: float, length: float) -> float:
    """0 -> 1 -> 0 over `length` seconds from `start`: up fast, down a little slower (a punch)."""
    u = (t - start) / max(1e-6, length)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return ease(u / 0.35, "out") if u < 0.35 else 1.0 - ease((u - 0.35) / 0.65, "both")


def fast(kind: str, amount: float, t: float, dur: float) -> tuple[float, float, float]:
    """A quick effect's window (centre x, centre y, zoom) `t` seconds into a clip `dur` seconds long."""
    amount = max(FAST_AMOUNT, amount)
    z1 = 1.0 + amount
    ft = min(FAST_TIME, dur / 2)
    if kind == "punch_in":
        return 0.5, 0.5, 1.0 + amount * ease(t / max(1e-6, ft), "out")
    if kind == "punch_out":
        return 0.5, 0.5, z1 - amount * ease(t / max(1e-6, ft), "out")
    if kind == "pulse":
        return 0.5, 0.5, 1.0 + amount * _bump(t, 0.0, min(0.6, dur))
    if kind == "crash":
        return 0.5, 0.5, 1.0 + amount * ease((t - (dur - ft)) / max(1e-6, ft), "in")
    if kind == "heartbeat":
        beat = 1.0                                  # one heartbeat a second: lub-dub
        ph = t % beat
        return 0.5, 0.5, 1.0 + amount * max(_bump(ph, 0.0, 0.22), 0.7 * _bump(ph, 0.26, 0.22))
    z = 1.0 + amount * 0.5
    room = 0.5 - 0.5 / z                            # how far the window can move off centre
    if kind == "shake":                             # quick, uneven: a few sines of unrelated rates
        dx = np.sin(t * 37.0) * 0.6 + np.sin(t * 23.0 + 1.3) * 0.4
        dy = np.sin(t * 31.0 + 0.7) * 0.6 + np.sin(t * 19.0 + 2.1) * 0.4
        return 0.5 + room * 0.9 * dx, 0.5 + room * 0.9 * dy, z
    if kind == "wobble":                            # slow and swaying
        return 0.5 + room * 0.9 * np.sin(t * 2.2), 0.5 + room * 0.6 * np.sin(t * 3.1 + 0.8), \
            z * (1.0 + 0.04 * np.sin(t * 1.7))
    return 0.5, 0.5, 1.0


def dynamic(kind: str, amount: float, ease_kind: str, u: float,
            target: tuple = (0.5, 0.5), dur: float = 1.0) -> tuple[float, float, float]:
    """The dynamic zoom's window (centre x, centre y, zoom) at `u` (0..1) of the clip. target: the face, as parts
    of the framed picture, for the "face" zooms; dur: the clip's length (seconds), for the quick effects."""
    if not kind:
        return 0.5, 0.5, 1.0
    if kind in FAST_KINDS:
        cx, cy, z = fast(kind, amount, u * dur, dur)
        z = max(1.0, z)
        return _fit(cx, z), _fit(cy, z), z
    (x0, y0, z0), (x1, y1, z1) = _ends(kind, amount, target)
    e = ease(u, ease_kind)
    z = z0 * (z1 / z0) ** e                     # zooms at an even pace (not faster at the wide end)
    x0, y0, x1, y1 = _fit(x0, z0), _fit(y0, z0), _fit(x1, z1), _fit(y1, z1)
    return _fit(x0 + (x1 - x0) * e, z), _fit(y0 + (y1 - y0) * e, z), z


def _quarter(dx: float, dy: float, q: int) -> tuple[float, float]:
    """A shift (parts of the picture) as it shows once the picture is turned q quarter turns clockwise."""
    q %= 4
    if q == 1:
        return -dy, dx
    if q == 2:
        return -dx, -dy
    if q == 3:
        return dy, -dx
    return dx, dy


@dataclass(frozen=True)
class Motion:
    """A clip's movement (a private snapshot). Equal when it would show the same."""
    kind: str = ""
    amount: float = DEFAULT_AMOUNT
    ease: str = "both"
    dur: float = 1.0                       # the clip's length on the timeline
    src_in: float = 0.0
    quarters: int = 0                      # quarter turns the picture gets before it is framed
    stab: Optional[object] = None          # stabilize.Correction, or None
    stab_key: tuple = ()
    face: tuple = ()                       # where a "face" zoom goes: (x, y) parts of the source picture

    @property
    def active(self) -> bool:
        return bool(self.kind) or self.stab is not None

    def max_zoom(self) -> float:
        z = 1.0 + max(0.0, self.amount) if self.kind else 1.0
        if self.kind in FAST_KINDS:
            z = 1.0 + max(FAST_AMOUNT, self.amount)
        return z * (self.stab.zoom if self.stab is not None else 1.0)

    def window(self, off: float, src_t: float, kx: float = 1.0, ky: float = 1.0, x0: float = 0.0,
               y0: float = 0.0) -> tuple[float, float, float, float]:
        """The window at `off` seconds into the clip (timeline), showing source time src_t: (centre x, centre y)
        as parts of the framed picture, zoom, and turn (degrees clockwise) to show it with. kx / ky: the source
        picture's size as multiples of the framed picture's (a crop shows part of it), so a shake measured on
        the source moves the window as much on the picture; x0 / y0: where the crop starts, as parts of the
        framed picture (see at_placement)."""
        target = (self.face[0] * kx - x0, self.face[1] * ky - y0) if self.face else (0.5, 0.5)
        cx, cy, z = dynamic(self.kind, self.amount, self.ease, off / max(1e-6, self.dur), target, self.dur)
        turn = 0.0
        if self.stab is not None:
            sx, sy, sa = self.stab.at(src_t)
            sx, sy = _quarter(sx, sy, self.quarters)
            z *= self.stab.zoom
            cx, cy = _fit(cx + sx * kx, z), _fit(cy + sy * ky, z)
            turn = -sa
        return cx, cy, z, turn

    def at_placement(self, off: float, src_t: float, p) -> tuple[float, float, float, float]:
        """window() for a picture framed as reframe.Placement p."""
        cw, ch = max(1, p.crop_w), max(1, p.crop_h)
        return self.window(off, src_t, p.scaled_w / cw, p.scaled_h / ch, p.crop_x / cw, p.crop_y / ch)


def of_clip(clip, footage, aspect: float = 16 / 9) -> Optional[Motion]:
    """A clip's Motion, on the main track or a video layer (None when it has neither a dynamic zoom nor
    stabilization). aspect: the shown picture's width / height."""
    from . import reframe, spherical, stabilize
    kind = getattr(clip, "motion", "") or ""
    strength = float(getattr(clip, "stabilize", 0.0) or 0.0)
    stab = None
    if strength > 0 and footage is not None and not footage.is_image and not spherical.is_360(footage):
        stab = stabilize.correction(footage, clip.src_in - stabilize.HANDLE, clip.src_out + stabilize.HANDLE,
                                    strength, aspect)
    if not kind and stab is None:
        return None
    # A layer's rotation turns the whole layer: its picture isn't turned before it is framed.
    quarters = 0 if hasattr(clip, "track") else reframe.split_rotation(clip.rotation)[0]
    face = tuple(clip.motion_face[:2]) if kind in FACE_KINDS and clip.motion_face else ()
    return Motion(kind, float(clip.motion_amount), clip.motion_ease or "both", max(1e-3, clip.duration),
                  clip.src_in, quarters, stab, stab.key if stab is not None else (), face)


# ----------------------------------------------------------------------------- export
@dataclass
class Plan:
    """The export's FFmpeg commands for one clip's movement."""
    tag: str
    commands: list = field(default_factory=list)   # sendcmd lines
    rotates: bool = False
    # A turning clip's window, cut larger than the picture (supersampled pixels) so that turned, its corners
    # are still picture (as the preview's), not black; (0, 0): as large as the picture.
    cut: tuple = (0, 0)


def export_plan(m: Motion, clip, p, sw: int, sh: int, frames: int, pre_f: int, fps: float, tag: str,
                ss: int = 1) -> Plan:
    """Per output frame (0..frames-1, the first pre_f before the clip's start) the size the source is scaled to,
    where the crop window sits in it, and the turn: commands to scale@{tag}s, crop@{tag}c and rotate@{tag}r.
    p: the clip's static reframe.Placement of the sw x sh source; ss: supersampling of the crop (the window is
    cut ss times larger, then scaled down, so it moves in steps of 1/ss pixel)."""
    plan = Plan(tag)
    cw, ch = p.crop_w * ss, p.crop_h * ss
    at = [m.at_placement((n - pre_f) / fps, clip.src_at((n - pre_f) / fps), p) for n in range(frames)]
    most = max((abs(a[3]) for a in at), default=0.0)
    if most > 1e-4:                        # the window turned by up to `most` degrees still covers the picture
        c, s = abs(np.cos(np.radians(most))), abs(np.sin(np.radians(most)))
        step = 2 * ss                      # (even, and whole pixels once scaled down)
        cw = int(np.ceil((p.crop_w * c + p.crop_h * s) * ss / step)) * step
        ch = int(np.ceil((p.crop_w * s + p.crop_h * c) * ss / step)) * step
        plan.cut = (cw, ch)
    for n, (cx, cy, z, turn) in enumerate(at):
        w = max(cw, int(round(p.scaled_w * z * ss / 2.0)) * 2)
        h = max(ch, int(round(p.scaled_h * z * ss / 2.0)) * 2)
        zx, zy = w / (p.scaled_w * ss), h / (p.scaled_h * ss)
        # The window's centre in the scaled source, then its top-left corner in the source scaled by z.
        x = int(round(((p.crop_x + cx * p.crop_w) * ss) * zx - cw / 2.0))
        y = int(round(((p.crop_y + cy * p.crop_h) * ss) * zy - ch / 2.0))
        x, y = min(max(0, x), w - cw), min(max(0, y), h - ch)
        t = max(0.0, (n - 0.25) / fps)
        line = f"{t:.4f} scale@{tag}s w {w}, scale@{tag}s h {h}, crop@{tag}c x {x}, crop@{tag}c y {y}"
        if abs(turn) > 1e-4:
            plan.rotates = True
        line += f", rotate@{tag}r a {np.radians(turn):.6f};"
        plan.commands.append(line)
    if not plan.rotates:
        plan.commands = [c.rsplit(", rotate@", 1)[0] + ";" for c in plan.commands]
    return plan
