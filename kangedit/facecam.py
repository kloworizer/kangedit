"""Facecam layout for game recordings: many streamers record the game with their webcam in a corner of the
same picture. The layout shows that webcam corner big and the game beside it:

* vertical (9:16) and square projects: the classic stacked layout, the facecam band on top (a third of the
  frame) and the game below, cropped to the game area (beside the cam corner when there is room);
* landscape projects: picture in picture, the cam enlarged in its own corner (covering the small original).

The webcam is found by its face: the face seen in the same place in most of a few frames, away from the middle
of the picture (a face in the middle is the video's subject or an in-game face, not a facecam corner), grown to
a typical webcam box around it. When no face is found the user picks the corner.

Both parts are the same recording: the main-track clip framed on the game (its box / zoom / offsets) and a
video-layer clip of the same footage and source range, cropped to the cam (OverlayClip.box / zoom / pan, role
"facecam", linked to its main clip). follow() keeps every facecam clip on its main clip's range after trims,
slips, splits and reorders (called with multicam.follow, after every edit), so they never drift apart. A
separate webcam file is laid out with the multicam angles instead (synced by sound; see main_window).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import reframe
from .models import Footage, OverlayClip, Project, TimelineClip, new_id

FACECAM = "facecam"    # OverlayClip.role of a facecam corner
CORNERS = {"bl": "Bottom left", "tl": "Top left", "br": "Bottom right", "tr": "Top right"}
CAM_RATIO = 16 / 9     # a webcam box's shape (width / height, in pixels)
HEAD = 2.6             # a webcam box is about this many face heights tall (head and shoulders)
FACE_AT = 0.45         # the face's centre this far down the box
MAX_CAM = 0.5          # a webcam box at most this part of the picture's width and height
EDGE = 0.2             # a facecam's face sits at least this far from the picture's centre (either way)
SNAP = 0.06            # a box this close to the picture's edge sits against it
CORNER_W = 0.25        # a corner picked by hand: the cam box is this part of the picture's width
BAND = 1 / 3           # stacked layout: the facecam's part of the frame's height, on top
PIP_GROW = 1.6         # picture in picture: the cam this much bigger than in the recording
PIP_MAX = 0.45         # ... and at most this part of the frame's width / height

Rect = tuple[float, float, float, float]   # x, y (top left), width, height: parts of the picture
_BOX = ("box_x", "box_y", "box_w", "box_h", "zoom", "offset_x", "offset_y")   # a main clip's game framing


# ----------------------------------------------------------------------------- finding the webcam
def stable_face(per_frame: list[list[tuple]]) -> Optional[tuple]:
    """The facecam's face from the faces found in a few frames (faces.detect: centre x, centre y, width,
    height, score as parts of the picture): the one seen in the same place in the most frames, at least EDGE
    from the picture's centre. Its median place, or None when no face is steady enough."""
    n = len(per_frame)
    every = [f for fs in per_frame for f in fs if max(abs(f[0] - 0.5), abs(f[1] - 0.5)) >= EDGE]
    if not every:
        return None

    def same(a, b) -> bool:
        return abs(a[0] - b[0]) < max(a[2], b[2]) * 0.8 and abs(a[1] - b[1]) < max(a[3], b[3]) * 0.8

    best, best_key = None, None
    for cand in every:
        hits = [m for fs in per_frame if (m := next((f for f in fs if same(cand, f)), None)) is not None]
        key = (len(hits), sum(h[4] for h in hits))
        if best_key is None or key > best_key:
            best, best_key = hits, key
    if len(best) < max(1, int(np.ceil(0.4 * n))) or (n >= 3 and len(best) < 2):
        return None              # a face in one frame of many is the game's, not a webcam
    return tuple(float(v) for v in np.median(np.asarray(best), axis=0))


def cam_rect(face: tuple, sw: int, sh: int, ratio: float = CAM_RATIO) -> Rect:
    """The webcam box around a facecam face (centre x, centre y, width, height, parts of a sw x sh picture):
    a ratio-shaped box HEAD face heights tall, the face a bit above its middle, against the picture's edge
    when it is close to it, kept inside the picture."""
    cx, cy, fh = face[0], face[1], face[3]
    h = fh * HEAD
    w = h * sh * ratio / sw
    k = min(1.0, MAX_CAM / max(w, 1e-6), MAX_CAM / max(h, 1e-6))
    w, h = w * k, h * k
    x, y = cx - w / 2, cy - FACE_AT * h
    x = 0.0 if x < SNAP else 1.0 - w if x + w > 1.0 - SNAP else x
    y = 0.0 if y < SNAP else 1.0 - h if y + h > 1.0 - SNAP else y
    return (float(np.clip(x, 0.0, 1.0 - w)), float(np.clip(y, 0.0, 1.0 - h)), float(w), float(h))


def corner_rect(corner: str, sw: int, sh: int, ratio: float = CAM_RATIO) -> Rect:
    """A webcam box in a corner picked by hand (CORNERS key): CORNER_W of the picture wide, ratio-shaped."""
    w = CORNER_W
    h = min(MAX_CAM, w * sw / (sh * ratio))
    return (0.0 if "l" in corner else 1.0 - w, 0.0 if "t" in corner else 1.0 - h, w, h)


def corner_of(rect: Rect) -> str:
    """The corner (CORNERS key) a webcam box is nearest."""
    x, y, w, h = rect
    return ("t" if y + h / 2 < 0.5 else "b") + ("l" if x + w / 2 < 0.5 else "r")


def find(frames: list[np.ndarray]) -> Optional[Rect]:
    """The webcam box of a game recording from a few of its RGB frames, or None when no facecam is found."""
    from .faces import detect
    if not frames:
        return None
    face = stable_face([detect(img) for img in frames])
    if face is None:
        return None
    h, w = frames[0].shape[:2]
    return cam_rect(face, w, h)


# ----------------------------------------------------------------------------- the layout
def fill_crop(sw: int, sh: int, bw: float, bh: float, rect: Rect) -> tuple[float, float, float]:
    """(zoom, offset x, offset y) that make a bw x bh box (filled: reframe.place "fill") show the part `rect`
    of a sw x sh picture: as much of it as the box's shape allows, centred on it."""
    bw, bh = max(2.0, bw), max(2.0, bh)
    s0 = max(bw / sw, bh / sh)
    rw, rh = max(1e-6, rect[2] * sw), max(1e-6, rect[3] * sh)
    region_w = min(rw, rh * bw / bh)               # the widest box-shaped part of the rect
    zoom = float(np.clip(bw / (s0 * region_w), 1.0, 8.0))
    sc = s0 * zoom

    def offset(centre: float, full: float, win: float) -> float:
        if full <= win + 1e-6:
            return 0.0
        return float(np.clip(2.0 * (centre * full - win / 2) / (full - win) - 1.0, -1.0, 1.0))

    return zoom, offset(rect[0] + rect[2] / 2, sw * sc, bw), offset(rect[1] + rect[3] / 2, sh * sc, bh)


def game_rect(cam: Optional[Rect], sw: int, sh: int, aspect: float) -> Rect:
    """The game's part of a sw x sh picture for a box `aspect` (width / height) wide: the whole height (or
    width), centred, moved beside the cam corner when it would show it and there is room."""
    w = sh * aspect / sw
    if w >= 1.0:
        h = sw / (aspect * sh)
        return (0.0, (1.0 - h) / 2, 1.0, h)
    x = (1.0 - w) / 2
    if cam is not None and x < cam[0] + cam[2] and cam[0] < x + w:
        if cam[0] + cam[2] / 2 < 0.5 and cam[0] + cam[2] + w <= 1.0:
            x = cam[0] + cam[2]
        elif cam[0] + cam[2] / 2 >= 0.5 and cam[0] - w >= 0.0:
            x = cam[0] - w
    return (x, 0.0, w, 1.0)


def stacked(tw: int, th: int) -> bool:
    """Whether a frame gets the stacked layout (vertical and square) rather than picture in picture."""
    return tw <= th


def frame_box(rect: Rect, sw: int, sh: int, tw: int, th: int, mode: str = "fill") -> Rect:
    """Where part `rect` of a full-frame clip's picture lands in a tw x th frame (parts of the frame)."""
    p = reframe.place(sw, sh, tw, th, mode)
    x = (rect[0] * p.scaled_w - p.crop_x + p.pad_x) / tw
    y = (rect[1] * p.scaled_h - p.crop_y + p.pad_y) / th
    return (x, y, rect[2] * p.scaled_w / tw, rect[3] * p.scaled_h / th)


def lay_out(clip: TimelineClip, f: Footage, cam: Rect, tw: int, th: int, start: float,
            mode: str = "fill") -> OverlayClip:
    """Lay one main-track clip out with its facecam (cam: the webcam box in its picture). The clip is framed
    on the game (stacked: a box under the cam band; picture in picture: the whole frame) and the returned
    layer clip shows the cam, cropped from the same recording over the same range, linked to the clip."""
    sw, sh = f.width, f.height
    oc = OverlayClip(f.id, round(start, 4), clip.src_in, clip.src_out, volume=0.0, link=clip.id, role=FACECAM,
                     speed=clip.speed, ramp=clip.ramp, audio_mode=clip.audio_mode)
    if stacked(tw, th):
        bh = (1.0 - BAND) * th
        clip.box_x, clip.box_y, clip.box_w, clip.box_h = 0.5, BAND + (1.0 - BAND) / 2, 1.0, 1.0 - BAND
        clip.zoom, clip.offset_x, clip.offset_y = fill_crop(sw, sh, tw, bh, game_rect(cam, sw, sh, tw / bh))
        oc.x, oc.y, oc.box_w, oc.box_h = 0.5, BAND / 2, 1.0, BAND
    else:
        clip.box_x, clip.box_y, clip.box_w, clip.box_h = 0.5, 0.5, 0.0, 0.0
        clip.zoom, clip.offset_x, clip.offset_y = 1.0, 0.0, 0.0
        fx, fy, fw, fh = frame_box(cam, sw, sh, tw, th, mode)
        corner = corner_of(cam)
        left, top = "l" in corner, "t" in corner
        # Big enough to cover the recording's own small cam in that corner.
        reach_x = fx + fw if left else 1.0 - fx
        reach_y = fy + fh if top else 1.0 - fy
        k = max(PIP_GROW, reach_x / max(fw, 1e-6), reach_y / max(fh, 1e-6))
        k = min(k, PIP_MAX / max(fw, 1e-6), PIP_MAX / max(fh, 1e-6))
        oc.box_w, oc.box_h = fw * k, fh * k
        oc.x = oc.box_w / 2 if left else 1.0 - oc.box_w / 2
        oc.y = oc.box_h / 2 if top else 1.0 - oc.box_h / 2
    oc.zoom, oc.pan_x, oc.pan_y = fill_crop(sw, sh, oc.box_w * tw, oc.box_h * th, cam)
    return oc


def free_track(project: Project, start: float, end: float, max_tracks: int = 8) -> int:
    """The lowest video layer with nothing on it from start to end (facecam clips of other clips excepted)."""
    for k in range(max_tracks):
        if not any(o.track == k and o.start < end - 1e-3 and o.end > start + 1e-3 and o.role != FACECAM
                   for o in project.overlays):
            return k
    return max_tracks - 1


def apply(project: Project, clips: list[TimelineClip], cam: Rect, tw: int, th: int, mode: str = "fill") -> int:
    """Give main-track clips (all of the same kind of recording: the same webcam box) the facecam layout.
    Their earlier facecam clips are replaced. Returns how many clips were laid out."""
    ids = {c.id for c in clips}
    project.overlays = [o for o in project.overlays if not (o.role == FACECAM and o.link in ids)]
    starts = project.clip_starts()
    track = None
    n = 0
    for c in clips:
        i = project.index_of(c.id)
        f = project.footage.get(c.footage_id)
        if i < 0 or f is None or f.is_image:
            continue
        if track is None:          # one layer for the whole layout, free over the edit's range
            track = free_track(project, starts[i], starts[i] + c.duration)
        oc = lay_out(c, f, cam, tw, th, starts[i], mode)
        oc.track = track
        project.overlays.append(oc)
        n += 1
    st = project.settings
    st.overlay_tracks = max(st.overlay_tracks, (track or 0) + 1)
    # Every clip of these recordings now counts as laid out: only new ones (a split, a clip added later) join.
    # The game framing of each recording is kept for its clips to come (even after the laid-out ones are gone).
    fids = {c.footage_id for c in clips}
    done = set((st.facecam or {}).get("clips") or []) | {c.id for c in project.clips if c.footage_id in fids}
    frames = dict((st.facecam or {}).get("frames") or {})
    for c in clips:
        frames[c.footage_id] = {k: getattr(c, k) for k in _BOX}
    st.facecam = {**(st.facecam or {}), "clips": sorted(done), "frames": frames}
    return n


def follow(project: Project) -> bool:
    """Keep every linked facecam clip on its main clip: the same place on the timeline, source range and speed.
    A facecam whose main clip is gone goes too; a new clip of a laid-out recording (a split, a clip added later)
    gets the facecam (and the game framing) of the others; one the user deleted stays deleted. Returns whether
    anything changed."""
    st = project.settings
    cams = [o for o in project.overlays if o.role == FACECAM]
    if not cams:
        return False
    before = [o.to_dict() for o in project.overlays] + [c.to_dict() for c in project.clips]
    handled = set((st.facecam or {}).get("clips") or [])
    linked = {o.link: o for o in cams if o.linked}
    by_id = {c.id: c for c in project.clips}
    like = {}                                  # footage id -> a facecam clip of it (its look, for new clips)
    for o in sorted(cams, key=lambda o: o.link not in by_id):
        like.setdefault(o.footage_id, o)
    frames = (st.facecam or {}).get("frames") or {}
    starts = dict(zip(by_id, project.clip_starts()))
    out = [o for o in project.overlays if o.role != FACECAM or not o.linked]
    current = []
    for c in project.clips:
        if c.footage_id not in like:
            continue
        current.append(c.id)
        o = linked.get(c.id)
        if o is None:
            if c.id in handled:
                continue                       # its facecam was deleted (or unlinked) by the user
            src = like[c.footage_id]
            o = OverlayClip.from_dict(src.to_dict())
            o.id, o.link, o.linked, o.fade_in, o.fade_out = new_id(), c.id, True, 0.0, 0.0
            frame = frames.get(c.footage_id)
            if frame is not None and c.box_w <= 0 < frame.get("box_w", 0):
                for k in _BOX:
                    setattr(c, k, frame.get(k, getattr(c, k)))
        if o.footage_id == c.footage_id:
            o.src_in, o.src_out = c.src_in, c.src_out
        o.start = round(starts[c.id], 4)
        o.speed, o.ramp, o.audio_mode = c.speed, c.ramp, c.audio_mode
        out.append(o)
    project.overlays = out
    st.facecam = {**(st.facecam or {}), "clips": sorted(set(current) | (handled & set(by_id)))}
    return [o.to_dict() for o in project.overlays] + [c.to_dict() for c in project.clips] != before


def cam_overlay_place(oc: OverlayClip, f: Footage, tw: int, th: int, corner: str) -> None:
    """Place a separate webcam file's layer clip (a multicam angle) for the facecam layout: the band on top
    (stacked, cropped around its face) or a corner (picture in picture, the whole cam picture)."""
    oc.rotation, oc.zoom = 0.0, 1.0
    if stacked(tw, th):
        oc.scale, oc.x, oc.y, oc.box_w, oc.box_h = 1.0, 0.5, BAND / 2, 1.0, BAND
        oc.pan_x = oc.pan_y = 0.0
        if f.face:
            oc.pan_x, oc.pan_y = reframe.pan_for_point(f.face[0], f.face[1], f.width, f.height, tw, th,
                                                       oc.box_w, oc.box_h)
        return
    oc.scale, oc.box_w, oc.box_h, oc.pan_x, oc.pan_y = 0.34, 0.0, 0.0, 0.0, 0.0
    w, h = reframe.overlay_size(f.width, f.height, tw, th, oc.scale)
    oc.x = w / 2 / tw if "l" in corner else 1.0 - w / 2 / tw
    oc.y = h / 2 / th if "t" in corner else 1.0 - h / 2 / th


def game_box(clip: TimelineClip, f: Footage, tw: int, th: int) -> None:
    """Frame a main clip on the game under the cam band (stacked) or on the whole frame (picture in picture),
    when the webcam is a separate file."""
    if stacked(tw, th):
        bh = (1.0 - BAND) * th
        clip.box_x, clip.box_y, clip.box_w, clip.box_h = 0.5, BAND + (1.0 - BAND) / 2, 1.0, 1.0 - BAND
        clip.zoom, clip.offset_x, clip.offset_y = fill_crop(f.width, f.height, tw, bh,
                                                            game_rect(None, f.width, f.height, tw / bh))
    else:
        clip.box_x, clip.box_y, clip.box_w, clip.box_h = 0.5, 0.5, 0.0, 0.0
        clip.zoom, clip.offset_x, clip.offset_y = 1.0, 0.0, 0.0
