"""Multicam: several camera angles of the same moments, synced by their sound and shown together.

The main angle's footage is edited as usual on the main track. Every other angle is synced to it (sync.py)
and laid on the video layers above, cut at exactly the same moments, so each picture shows the same instant.
A layout template places the angles in the frame: picture in picture, a talking-head box, stacked, side by
side or a grid. Where the main angle sits in part of the frame, its clips on the main track are shown in that
box, so N angles always take the main track and N - 1 layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import reframe
from .config import frame_size
from .models import Footage, OverlayClip, Project, TimelineClip
from .sync import SyncResult

TEMPLATES = {
    "pip": "Picture in picture (main full frame, other angles in a corner)",
    "talking": "Main video + talking cam (a framed box for the speaker)",
    "stack": "Stacked, top / bottom",
    "side": "Side by side",
    "grid": "Grid (2 × 2)",
}
TEMPLATE_HINTS = {
    "pip": "The main angle fills the frame; the others are small, whole pictures in the corners.",
    "talking": "Landscape: the main angle fills the frame and the talking cam sits in a square box in the corner, "
               "cropped around the face. Vertical / square: the main angle on top, the speaker below.",
    "stack": "Each angle gets a full-width band, cropped around its subject. Made for vertical 9:16.",
    "side": "Each angle gets a full-height column, cropped around its subject. Made for landscape 16:9.",
    "grid": "Up to four angles in a 2 × 2 grid, each cropped around its subject.",
}
AUDIO_CHOICES = {"main": "Main angle's sound"}   # + "angle:<n>": that angle's sound (see audio_choices)
MARGIN = 0.03          # corner boxes keep this part of the frame from the edges
MIN_PIECE = 0.3        # seconds: shorter overlaps of an angle are left out


def audio_choices(n_angles: int) -> dict[str, str]:
    out = dict(AUDIO_CHOICES)
    for k in range(2, n_angles + 1):
        out[f"angle:{k}"] = f"Angle {k}'s sound"
    return out


@dataclass
class Slot:
    """Where one angle goes: centre, size and whether it is cropped to a box."""
    x: float
    y: float
    scale: float = 1.0
    box_w: float = 0.0       # 0: the whole picture, fitted
    box_h: float = 0.0


def main_in_box(template: str, aspect: tuple[int, int]) -> bool:
    """Whether the template puts the main angle in a box on a layer (instead of full frame on the main track)."""
    if template == "talking":
        return aspect[0] <= aspect[1]
    return template in ("stack", "side", "grid")


def slots(template: str, n: int, tw: int, th: int) -> list[Slot]:
    """Places for angles 1..n (index 0 is the main angle: full frame when it stays on the main track)."""
    landscape = tw > th
    corners = [(1 - MARGIN, 1 - MARGIN), (MARGIN, 1 - MARGIN), (1 - MARGIN, MARGIN), (MARGIN, MARGIN)]
    if template == "pip":
        out = [Slot(0.5, 0.5)]
        for k in range(1, n):
            cx, cy = corners[(k - 1) % 4]
            sc = 0.32 if landscape else 0.45
            out.append(Slot(cx, cy, sc))
        return out
    if template == "talking":
        if not landscape:   # main on top, speaker below
            out = [Slot(0.5, 0.3, 1.0, 1.0, 0.6)]
            rows = max(1, n - 1)
            for k in range(1, n):
                w = 1.0 / rows
                out.append(Slot(w * (k - 0.5), 0.8, 1.0, w, 0.4))
            return out
        box_h = 0.32                    # a square box, a third of the frame's height
        box_w = box_h * th / tw
        out = [Slot(0.5, 0.5)]
        for k in range(1, n):
            cx, cy = corners[(k - 1) % 4]
            out.append(Slot(cx, cy, 1.0, box_w, box_h))
        return out
    if template == "stack":
        return [Slot(0.5, (k + 0.5) / n, 1.0, 1.0, 1.0 / n) for k in range(n)]
    if template == "side":
        return [Slot((k + 0.5) / n, 0.5, 1.0, 1.0 / n, 1.0) for k in range(n)]
    # grid
    cols = 2 if n > 1 else 1
    rows = (min(n, 4) + cols - 1) // cols
    out = []
    for k in range(n):
        r, c = divmod(k % 4, cols)
        out.append(Slot((c + 0.5) / cols, (r + 0.5) / rows, 1.0, 1.0 / cols, 1.0 / rows))
    return out


def _corner_centre(slot: Slot, w: float, h: float) -> tuple[float, float]:
    """A corner slot's (x, y) are the corner the picture hugs: move the centre in by half its size."""
    x = slot.x - w / 2 if slot.x > 0.5 else slot.x + w / 2 if slot.x < 0.5 else slot.x
    y = slot.y - h / 2 if slot.y > 0.5 else slot.y + h / 2 if slot.y < 0.5 else slot.y
    return x, y


def place(c: OverlayClip, f: Footage, slot: Slot, template: str, tw: int, th: int) -> None:
    """Put overlay c (showing footage f) in a template slot, cropping a box around the footage's face."""
    c.scale, c.box_w, c.box_h, c.rotation = slot.scale, slot.box_w, slot.box_h, 0.0
    c.pan_x = c.pan_y = 0.0
    w, h = reframe.overlay_size(f.width, f.height, tw, th, c.scale, c.box_w, c.box_h)
    if template in ("pip", "talking") and (slot.x, slot.y) != (0.5, 0.5) and slot.box_w < 0.9:
        c.x, c.y = _corner_centre(slot, w / tw, h / th)
    else:
        c.x, c.y = slot.x, slot.y
    if c.box_w > 0 and f.face:
        c.pan_x, c.pan_y = reframe.pan_for_point(f.face[0], f.face[1], f.width, f.height, tw, th, c.box_w, c.box_h,
                                                 c.scale)


def place_main(c: TimelineClip, f: Footage, slot: Slot, boxed: bool, tw: int, th: int) -> None:
    """Put a main-angle clip in its slot: a box of the frame cropped around the face (boxed layouts), or the
    whole frame (the box and the crop it set are cleared)."""
    if boxed:
        c.box_x, c.box_y, c.box_w, c.box_h = slot.x, slot.y, slot.box_w, slot.box_h
        c.zoom, c.offset_x, c.offset_y = 1.0, 0.0, 0.0
        if f.face:
            c.offset_x, c.offset_y = reframe.pan_for_point(f.face[0], f.face[1], f.width, f.height, tw, th,
                                                           slot.box_w, slot.box_h)
    elif c.box_w > 0:
        c.box_x, c.box_y, c.box_w, c.box_h = 0.5, 0.5, 0.0, 0.0
        c.zoom, c.offset_x, c.offset_y = 1.0, 0.0, 0.0


def _copy_box(dst: TimelineClip, src: TimelineClip) -> None:
    for k in ("box_x", "box_y", "box_w", "box_h", "zoom", "offset_x", "offset_y"):
        setattr(dst, k, getattr(src, k))


def _piece(r: SyncResult, a0: float, a1: float, dur: float) -> Optional[tuple[float, float, float]]:
    """The part of a main-angle source range [a0, a1] another angle covers: (main time where it starts,
    the angle's in point, its out point)."""
    b0, b1 = r.b_time(a0), r.b_time(a1)
    c0, c1 = max(0.0, b0), min(dur, b1)
    if c1 - c0 < MIN_PIECE:
        return None
    return r.a_time(c0), c0, c1


def build(project: Project, clips: list[TimelineClip], angles: dict[str, int],
          syncs: dict[str, tuple[str, SyncResult]], template: str, audio: str = "main",
          set_volume: bool = True, layout_main: bool = True) -> list[OverlayClip]:
    """The overlays showing the other angles over `clips` (main-angle clips already on the main track), in
    sync: angle k on layer V(k), one layer per extra angle. angles: {footage id: angle number (1 = main)};
    syncs: {footage id: (main footage id, SyncResult)}. Unless told not to, sets the main clips' volume for
    the chosen sound and puts them in the layout's place (a box of the frame for the boxed layouts)."""
    st = project.settings
    tw, th = frame_size(st.aspect, st.resolution)
    n = max(angles.values(), default=1)
    lay = slots(template, n, tw, th)
    boxed_main = main_in_box(template, (tw, th))
    starts = project.clip_starts()
    out: list[OverlayClip] = []
    sound_angle = int(audio.split(":")[1]) if audio.startswith("angle:") else 1
    for c in clips:
        if angles.get(c.footage_id) != 1:
            continue
        i = project.index_of(c.id)
        if i < 0:
            continue
        t0 = starts[i]
        if set_volume:
            c.volume = 1.0 if sound_angle == 1 else 0.0
        if layout_main:
            place_main(c, project.footage[c.footage_id], lay[0], boxed_main, tw, th)
        for k in range(2, n + 1):
            best = None
            for fid, (ref, r) in syncs.items():
                if angles.get(fid) != k or ref != c.footage_id or fid not in project.footage:
                    continue
                piece = _piece(r, c.src_in, c.src_out, project.footage[fid].duration)
                if piece is not None and (best is None or piece[2] - piece[1] > best[1][2] - best[1][1]):
                    best = (fid, piece)
            if best is None:
                continue
            fid, (a_start, b_in, b_out) = best
            oc = OverlayClip(fid, round(t0 + (a_start - c.src_in), 4), round(b_in, 4), round(b_out, 4),
                             k - 2, angle=k, volume=1.0 if sound_angle == k else 0.0, link=c.id)
            place(oc, project.footage[fid], lay[k - 1], template, tw, th)
            out.append(oc)
    return out


def stored_syncs(project: Project) -> dict[str, tuple[str, SyncResult]]:
    out = {}
    for fid, (ref, o, d, conf, ov) in ((project.settings.multicam or {}).get("syncs") or {}).items():
        out[fid] = (ref, SyncResult(o, d, conf, ov))
    return out


# What a user may have changed on an angle's layer clip, kept when the clip is re-timed.
_LOOK = ("x", "y", "scale", "box_w", "box_h", "pan_x", "pan_y", "rotation", "opacity", "volume", "track", "grade")


def follow(project: Project) -> bool:
    """Keep the angles in step with the main track after an edit (trim, slip, split, Other moment, a new hook,
    reorder...): every angle clip is re-timed from the main clip it follows; main-angle clips that are new
    get their angles (and the layout's box); angle clips whose main clip is gone go too. An angle clip the
    user deleted stays deleted, and placements changed by hand are kept. Returns whether anything changed."""
    mc = project.settings.multicam or {}
    angles = {fid: k for fid, k in (mc.get("angles") or {}).items() if fid in project.footage}
    if not angles or not mc.get("syncs"):
        return False
    syncs = stored_syncs(project)
    handled = set(mc.get("clips") or [])
    before = [c.to_dict() for c in project.overlays] + [c.to_dict() for c in project.clips]
    # Earlier versions showed the main angle's box as a layer clip (angle 1): the main track does it now.
    project.overlays = [c for c in project.overlays if c.angle != 1]
    linked: dict[tuple[str, int], OverlayClip] = {(c.link, c.angle): c for c in project.overlays
                                                  if c.link and c.linked}
    unlinked = {(c.link, c.angle) for c in project.overlays if c.link and not c.linked}   # left as they are
    like = {c.angle: c for c in project.overlays if c.link}      # an angle's current placement, for new clips
    st = project.settings
    tw, th = frame_size(st.aspect, st.resolution)
    template = mc.get("template", "pip")
    boxed = main_in_box(template, (tw, th))
    boxed_like = next((c for c in project.clips if c.id in handled and c.box_w > 0), None)
    out = [c for c in project.overlays if not c.link or not c.linked]
    current = []
    for clip in project.clips:
        if angles.get(clip.footage_id) != 1:
            continue
        current.append(clip.id)
        if boxed and clip.id not in handled and clip.box_w <= 0:     # a new clip joins the layout
            if boxed_like is not None:
                _copy_box(clip, boxed_like)
            else:
                place_main(clip, project.footage[clip.footage_id],
                           slots(template, max(angles.values()), tw, th)[0], True, tw, th)
        for oc in build(project, [clip], angles, syncs, template, mc.get("audio", "main"),
                        set_volume=False, layout_main=False):
            if (clip.id, oc.angle) in unlinked:
                continue
            old = linked.get((clip.id, oc.angle))
            if old is None and clip.id in handled:
                continue                       # deleted by the user
            src = old if old is not None else like.get(oc.angle)
            if src is not None:
                for k in _LOOK:
                    setattr(oc, k, getattr(src, k))
            if old is not None:
                oc.id, oc.fade_in, oc.fade_out = old.id, old.fade_in, old.fade_out
            out.append(oc)
    project.overlays = out
    project.settings.multicam = {**mc, "clips": current}
    return [c.to_dict() for c in project.overlays] + [c.to_dict() for c in project.clips] != before


def angles_of(project: Project, clip_id: str) -> list[OverlayClip]:
    """The camera-angle clips (linked or not) that belong to a main-track clip."""
    return [c for c in project.overlays if c.link == clip_id and c.angle >= 2]


def set_linked(project: Project, overlays: list[OverlayClip], on: bool) -> int:
    """Link (back in sync, re-timed from its main clip) or unlink (edited on its own) camera-angle clips.
    Returns how many changed."""
    n = 0
    for c in overlays:
        if c.link and c.linked != on:
            c.linked = on
            n += 1
    if on and n:
        follow(project)
    return n


def relayout(project: Project, template: str) -> int:
    """Move the angles to another template: the main angle's clips take its place on the main track (a box
    of the frame, or the whole frame) and each other angle its place on its layer. Returns how many angle
    clips (on the layers) were placed."""
    st = project.settings
    tw, th = frame_size(st.aspect, st.resolution)
    multi = [c for c in project.overlays if c.angle >= 2]
    if not multi:
        return 0
    n = max(c.angle for c in multi)
    lay = slots(template, n, tw, th)
    boxed_main = main_in_box(template, (tw, th))
    main_ids = {fid for fid, k in (st.multicam.get("angles") or {}).items() if k == 1}
    mains = [c for c in project.clips if c.footage_id in main_ids]
    for c in mains:
        place_main(c, project.footage[c.footage_id], lay[0], boxed_main, tw, th)
    for c in multi:
        c.track = c.angle - 2
        place(c, project.footage[c.footage_id], lay[c.angle - 1], template, tw, th)
    project.overlays = [c for c in project.overlays if c.angle == 0] + multi
    st.multicam = {**st.multicam, "template": template}
    top = max((c.track for c in project.overlays), default=0)
    st.overlay_tracks = max(1, top + 1)
    return len(multi)


def coverage_blockers(project_footage: dict[str, Footage], angles: dict[str, int],
                      syncs: dict[str, tuple[str, SyncResult]]) -> list[TimelineClip]:
    """Stand-in clips over the parts of the main-angle footage no other angle recorded, so an auto edit
    (which avoids clips already used) only picks moments every angle shows."""
    out = []
    n = max(angles.values(), default=1)
    for fid, k in angles.items():
        if k != 1 or fid not in project_footage:
            continue
        dur = project_footage[fid].duration
        covered = [(0.0, dur)]
        for other in range(2, n + 1):
            spans = []
            for sid, (ref, r) in syncs.items():
                if angles.get(sid) == other and ref == fid and sid in project_footage:
                    spans.append((max(0.0, r.a_time(0.0)), min(dur, r.a_time(project_footage[sid].duration))))
            if not spans:          # this angle didn't sync at all: it doesn't limit the edit
                continue
            # intersect what is covered so far with this angle's spans
            covered = [(max(a, s0), min(b, s1)) for a, b in covered for s0, s1 in spans if min(b, s1) > max(a, s0)]
        t = 0.0
        for a, b in sorted(covered):
            if a - t > 0.05:
                out.append(TimelineClip(fid, t, a))
            t = max(t, b)
        if dur - t > 0.05:
            out.append(TimelineClip(fid, t, dur))
    return out
