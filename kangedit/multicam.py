"""Multicam: several camera angles of the same moments, synced by their sound and shown together.

The main angle's footage is edited as usual on the main track. Every other angle is synced to it (sync.py)
and laid on the video layers above, cut at exactly the same moments, so each picture shows the same instant.
A layout template places the angles in the frame: picture in picture, a talking-head box, stacked, side by
side or a grid. Where the main angle sits in part of the frame, its clips on the main track are shown in that
box, so N angles always take the main track and N - 1 layers.

Switching is the other way to use several cameras: one angle at a time, full frame, cut to the best angle as
a director would in a live multicam edit. switch_plan() scores every angle along the main angle's recording
(who is speaking, face size, activity, sharpness, shake) and picks the shots; the angles other than the main
one then sit on a single layer, one piece per shot.

At a transition between two main-track clips the angles take part too: a dissolve cross-fades them (the
outgoing angle runs on past the cut, the incoming one fades in over it), a dip fades them out and in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import reframe
from .edits import edits
from .config import frame_size
from .models import Footage, OverlayClip, Project, TimelineClip
from .speed import clamp
from .sync import SyncResult

TEMPLATES = {
    "pip": "Picture in picture (main full frame, other angles in a corner)",
    "talking": "Main video + talking cam (a framed box for the speaker)",
    "stack": "Stacked, top / bottom",
    "side": "Side by side",
    "grid": "Grid (2 × 2)",
    "switch": "Switching: one angle at a time, cut to the best angle",
}
TEMPLATE_HINTS = {
    "pip": "The main angle fills the frame; the others are small, whole pictures in the corners.",
    "talking": "Landscape: the main angle fills the frame and the talking cam sits in a square box in the corner, "
               "cropped around the face. Vertical / square: the main angle on top, the speaker below.",
    "stack": "Each angle gets a full-width band, cropped around its subject. Made for vertical 9:16.",
    "side": "Each angle gets a full-height column, cropped around its subject. Made for landscape 16:9.",
    "grid": "Up to four angles in a 2 × 2 grid, each cropped around its subject.",
    "switch": "Like a live multicam edit: one camera fills the frame at a time, and the edit cuts between them at "
              "the best moments (who is speaking, the closer face, the steadier and sharper picture). The sound "
              "stays on the main angle.",
}
AUDIO_CHOICES = {"main": "Main angle's sound"}   # + "angle:<n>": that angle's sound (see audio_choices)
MARGIN = 0.03          # corner boxes keep this part of the frame from the edges
MIN_PIECE = 0.3        # seconds: shorter overlaps of an angle are left out
DIPS = ("dip_black", "dip_white")
# Switching: scores every SWITCH_STEP seconds; shots last at least MIN_SHOT, an angle held longer than MAX_SHOT
# gives way to the next best one for a while, and SWITCH_COST (score x seconds) keeps noise from cutting.
SWITCH_STEP = 0.1
MIN_SHOT = 2.0
MAX_SHOT = 8.0
SWITCH_COST = 1.2
MAIN_BIAS = 0.25       # the main angle (usually the wide shot) wins a tie


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
    if template == "switch":     # every other angle fills the frame, cropped around its subject
        return [Slot(0.5, 0.5)] + [Slot(0.5, 0.5, 1.0, 1.0, 1.0) for _ in range(1, n)]
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


def _best_file(project: Project, c: TimelineClip, k: int, a0: float, a1: float, angles: dict[str, int],
               syncs: dict[str, tuple[str, SyncResult]]) -> Optional[tuple[str, tuple[float, float, float]]]:
    """The file of angle k covering the most of main clip c's source range [a0, a1]: (footage id, piece)."""
    best = None
    for fid, (ref, r) in syncs.items():
        if angles.get(fid) != k or ref != c.footage_id or fid not in project.footage:
            continue
        piece = _piece(r, a0, a1, project.footage[fid].duration)
        if piece is not None and (best is None or piece[2] - piece[1] > best[1][2] - best[1][1]):
            best = (fid, piece)
    return best


def _shots(project: Project, c: TimelineClip, angles: dict[str, int], syncs: dict, template: str,
           switch: Optional[dict]) -> list[tuple[int, int, str, tuple[float, float, float]]]:
    """The angle pieces shown over main clip c: [(angle, piece number, footage id, (main time where it starts,
    in point, out point))]. Every covering angle for the layouts; with switching, the shots of the plan."""
    n = max(angles.values(), default=1)
    out = []
    if template != "switch":
        for k in range(2, n + 1):
            best = _best_file(project, c, k, c.src_in, c.src_out, angles, syncs)
            if best is not None:
                out.append((k, 0, best[0], best[1]))
        return out
    for s0, s1, k in (switch or {}).get(c.footage_id, []):
        a0, a1 = max(s0, c.src_in), min(s1, c.src_out)
        if k < 2 or a1 - a0 < MIN_PIECE:
            continue
        best = _best_file(project, c, k, a0, a1, angles, syncs)
        if best is not None:
            out.append((k, len(out), best[0], best[1]))
    return out


def _transitions(oc: OverlayClip, f: Footage, e_in, e_out, cross_out: bool) -> None:
    """Let an angle clip take part in the main track's transitions into / out of its clip (edits.Edit, or None
    when it doesn't start / end with the clip). A dissolve-like transition runs the angle into its handles and
    fades the incoming one in over the outgoing one (cross_out: the next clip doesn't start on this angle, so
    this one fades out itself); a dip fades it out / in at the cut. A ramped clip only fades (its handles would
    change the ramp)."""
    tm = oc.timemap()
    if e_in is not None and e_in.half > 0:
        h = e_in.half
        if e_in.trans in DIPS or oc.ramp:
            oc.mix_in = h
        else:
            ext = max(0.0, min(h, oc.src_in / tm.v_in))
            oc.src_in = round(oc.src_in - ext * tm.v_in, 4)
            oc.start = round(oc.start - ext, 4)
            oc.mix_in = ext + h
    if e_out is not None and e_out.half > 0:
        h = e_out.half
        if e_out.trans in DIPS or oc.ramp:
            oc.mix_out = h
        else:
            ext = max(0.0, min(h, (f.duration - oc.src_out) / tm.v_out))
            oc.src_out = round(oc.src_out + ext * tm.v_out, 4)
            if cross_out:
                oc.mix_out = ext + h


def build(project: Project, clips: list[TimelineClip], angles: dict[str, int],
          syncs: dict[str, tuple[str, SyncResult]], template: str, audio: str = "main",
          set_volume: bool = True, layout_main: bool = True, switch: Optional[dict] = None) -> list[OverlayClip]:
    """The overlays showing the other angles over `clips` (main-angle clips already on the main track), in
    sync: angle k on layer V(k), one layer per extra angle; with switching, the shots of the plan (`switch`,
    else the stored one) on one layer. angles: {footage id: angle number (1 = main)}; syncs: {footage id:
    (main footage id, SyncResult)}. Unless told not to, sets the main clips' volume for the chosen sound and
    puts them in the layout's place (a box of the frame for the boxed layouts)."""
    st = project.settings
    tw, th = frame_size(st.aspect, st.resolution)
    n = max(angles.values(), default=1)
    lay = slots(template, n, tw, th)
    boxed_main = main_in_box(template, (tw, th))
    if template == "switch":
        audio = "main"                    # one picture at a time: the sound stays on the main angle
        if switch is None:
            switch = (st.multicam or {}).get("switch") or {}
    starts = project.clip_starts()
    es = edits(project) + [None]
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
        nxt = project.clips[i + 1] if i + 1 < len(project.clips) else None
        next_first = set()                # angles the next clip starts on (they fade in over this clip's)
        if nxt is not None and angles.get(nxt.footage_id) == 1:
            next_first = {k for k, _, _, (a, _, _) in _shots(project, nxt, angles, syncs, template, switch)
                          if abs(a - nxt.src_in) < 0.02}
        for k, piece, fid, (a_start, b_in, b_out) in _shots(project, c, angles, syncs, template, switch):
            # The angle plays at the main clip's speed. A ramp needs the same source range to line up all the
            # way through; an angle covering only part of the clip gets the ramp's average speed over that part
            # instead (in step at both ends).
            speed, ramp = c.speed, c.ramp
            t_in = c.time_at(a_start)
            a_end = a_start + (b_out - b_in)
            at_start, at_end = abs(a_start - c.src_in) < 0.02, abs(a_end - c.src_out) < 0.02
            if ramp and not (at_start and at_end):
                speed, ramp = clamp((b_out - b_in) / max(1e-3, c.time_at(a_end) - t_in)), ""
            oc = OverlayClip(fid, round(t0 + t_in, 4), round(b_in, 4), round(b_out, 4),
                             0 if template == "switch" else k - 2, angle=k,
                             volume=1.0 if sound_angle == k else 0.0, link=c.id, speed=speed, ramp=ramp,
                             audio_mode=c.audio_mode, piece=piece)
            place(oc, project.footage[fid], lay[k - 1], template, tw, th)
            _transitions(oc, project.footage[fid], es[i] if at_start and i > 0 else None,
                         es[i + 1] if at_end else None, k not in next_first)
            out.append(oc)
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    """A feature as a robust z-score against the recording's own level (so each camera is judged by itself)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    med = np.median(x)
    spread = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return np.clip((x - med) / spread, -3.0, 3.0)


def angle_scores(an, times: np.ndarray) -> np.ndarray:
    """How good a shot angle footage (its Analysis) is at its own times `times`: louder than usual (this camera's
    microphone is nearest whoever is speaking), a bigger face, the subject moving, sharp, steady and well exposed.
    NaN where the recording has no picture."""
    t = np.asarray(an.times, dtype=np.float64)
    if t.size < 2:
        return np.full(len(times), np.nan)

    def at(x):
        return np.interp(times, t, np.asarray(x, dtype=np.float64))
    score = (1.0 * at(_norm(an.audio)) + 0.7 * at(_norm(an.faces)) + 0.3 * at(_norm(an.activity))
             + 0.3 * at(_norm(an.sharp)) - 0.4 * at(_norm(an.jerk)) - 0.5 * at(_norm(an.clipped)))
    score[(times < t[0] - 0.5) | (times > t[-1] + 0.5)] = np.nan
    return score


def _choose(scores: np.ndarray, step: float) -> np.ndarray:
    """Pick one angle (row) per sample: the best score over time, minus SWITCH_COST per cut (Viterbi), then
    shots shorter than MIN_SHOT folded into their neighbours and shots longer than MAX_SHOT broken by the next
    best angle where it comes closest. Rows that are NaN can't be chosen there."""
    k, n = scores.shape
    s = np.where(np.isnan(scores), -1e9, scores * step)
    back = np.zeros((k, n), dtype=np.int32)
    acc = s[:, 0].copy()
    for i in range(1, n):
        stay = acc
        best = int(np.argmax(acc))
        move = acc[best] - SWITCH_COST
        take = move > stay
        back[:, i] = np.where(take, best, np.arange(k))
        acc = np.where(take, move, stay) + s[:, i]
    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmax(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    min_len = max(1, int(round(MIN_SHOT / step)))

    def runs(p):
        edges = np.flatnonzero(np.diff(p)) + 1
        return list(zip(np.r_[0, edges], np.r_[edges, len(p)]))
    for _ in range(3):                       # short shots join the better neighbour
        changed = False
        for a, b in runs(path):
            if b - a >= min_len or (a == 0 and b == n):
                continue
            left = path[a - 1] if a > 0 else None
            right = path[b] if b < n else None
            cands = [x for x in (left, right) if x is not None and not np.isnan(scores[x, a:b]).any()]
            if cands:
                path[a:b] = max(cands, key=lambda x: np.nansum(scores[x, a:b]))
                changed = True
        if not changed:
            break
    max_len = int(round(MAX_SHOT / step))
    for a, b in runs(path):                  # a long hold gives way to the runner-up for a shot
        if b - a <= max_len + 2 * min_len or k < 2:
            continue
        cur = path[a]
        other = np.where(np.arange(k)[:, None] == cur, -np.inf, np.where(np.isnan(scores), -np.inf, scores))
        alt = np.max(other[:, a:b], axis=0)
        lo, hi = a + min_len, b - 2 * min_len
        if hi <= lo or not np.isfinite(alt[lo - a:hi - a]).any():
            continue
        gap = scores[cur, a:b] - alt          # +inf where no other angle is recording
        j = lo + int(np.argmin(gap[lo - a:hi - a]))
        who = int(np.argmax(other[:, j]))
        end = min(b - min_len, j + int(round(1.5 * min_len)))     # a short cutaway (reaction shot)
        if end - j >= min_len and np.isfinite(other[who, j:end]).all():
            path[j:end] = who
    return path


def switch_plan(project: Project, analyses: dict, angles: Optional[dict] = None,
                syncs: Optional[dict] = None) -> dict[str, list[list[float]]]:
    """The switching cut for each main-angle recording: {main footage id: [[start, end, angle], ...]} in its
    source time, from the angles' analyses (footage id -> Analysis). Angles not recording at a moment, or not
    analysed, are skipped there."""
    mc = project.settings.multicam or {}
    angles = angles if angles is not None else {f: k for f, k in (mc.get("angles") or {}).items()}
    syncs = syncs if syncs is not None else stored_syncs(project)
    plan = {}
    for fid, k in angles.items():
        if k != 1 or fid not in project.footage:
            continue
        dur = project.footage[fid].duration
        t = np.arange(0.0, max(SWITCH_STEP, dur), SWITCH_STEP)
        n_ang = max(angles.values(), default=1)
        rows = np.full((n_ang, len(t)), np.nan)
        main_an = analyses.get(fid)
        rows[0] = (angle_scores(main_an, t) if main_an is not None else np.zeros(len(t))) + MAIN_BIAS
        rows[0] = np.where(np.isnan(rows[0]), MAIN_BIAS, rows[0])          # the main angle is always there
        for sid, (ref, r) in syncs.items():
            a = angles.get(sid, 0)
            if ref != fid or a < 2 or sid not in project.footage or analyses.get(sid) is None:
                continue
            tb = r.b_time(t)
            sc = angle_scores(analyses[sid], tb)
            sc[(tb < 0) | (tb > project.footage[sid].duration)] = np.nan
            rows[a - 1] = np.where(np.isnan(rows[a - 1]), sc, np.fmax(rows[a - 1], sc))
        path = _choose(rows, SWITCH_STEP)
        edges = np.flatnonzero(np.diff(path)) + 1
        plan[fid] = [[round(float(t[a]), 3), round(float(t[b - 1] + SWITCH_STEP), 3), int(path[a]) + 1]
                     for a, b in zip(np.r_[0, edges], np.r_[edges, len(path)])]
    return plan


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
    linked: dict[tuple[str, int, int], OverlayClip] = {(c.link, c.angle, c.piece): c for c in project.overlays
                                                       if c.link and c.linked}
    unlinked = {(c.link, c.angle, c.piece) for c in project.overlays if c.link and not c.linked}   # left alone
    like = {c.angle: c for c in project.overlays if c.link}      # an angle's current placement, for new clips
    st = project.settings
    tw, th = frame_size(st.aspect, st.resolution)
    template = mc.get("template", "pip")
    switching = template == "switch"   # shots come and go with the plan: none counts as deleted by the user
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
            key = (clip.id, oc.angle, oc.piece)
            if key in unlinked:
                continue
            old = linked.get(key)
            if old is None and clip.id in handled and not switching:
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
    if template == "switch" or (st.multicam or {}).get("template") == "switch":
        # To or from switching the angle clips themselves change (one per shot, or one per angle): rebuild
        # them all from the main clips, in the new layout. Unlinked ones stay as they are.
        n = max(int(k) for k in (st.multicam.get("angles") or {"": 1}).values())
        lay0 = slots(template, n, tw, th)[0]
        main_ids = {fid for fid, k in (st.multicam.get("angles") or {}).items() if k == 1}
        for c in project.clips:
            if c.footage_id in main_ids:
                place_main(c, project.footage[c.footage_id], lay0, main_in_box(template, (tw, th)), tw, th)
                c.volume = 1.0 if template == "switch" else c.volume
        project.overlays = [c for c in project.overlays if not c.link or not c.linked]
        st.multicam = {**st.multicam, "template": template, "clips": []}
        follow(project)
        top = max((c.track for c in project.overlays), default=0)
        st.overlay_tracks = max(1, top + 1)
        return sum(1 for c in project.overlays if c.angle >= 2)
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


def covered_spans(project_footage: dict[str, Footage], angles: dict[str, int],
                  syncs: dict[str, tuple[str, SyncResult]]) -> dict[str, list[tuple[float, float]]]:
    """For each main-angle file, the parts of it every other (synced) angle recorded too: {id: [(a, b)]}."""
    out = {}
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
            covered = [(max(a, s0), min(b, s1)) for a, b in covered for s0, s1 in spans if min(b, s1) > max(a, s0)]
        out[fid] = sorted(covered)
    return out


def whole_clips(project_footage: dict[str, Footage], angles: dict[str, int],
                syncs: dict[str, tuple[str, SyncResult]], order: list[str], covered: bool = True) -> list[TimelineClip]:
    """The whole recording instead of its best moments: each main-angle file from start to end (in `order`),
    or only the parts every angle recorded."""
    spans = covered_spans(project_footage, angles, syncs) if covered else {}
    out = []
    order = list(order) + sorted((f for f, k in angles.items() if k == 1 and f not in order),
                                 key=lambda f: project_footage[f].path if f in project_footage else f)
    for fid in order:
        if angles.get(fid) != 1 or fid not in project_footage:
            continue
        for a, b in spans.get(fid, [(0.0, project_footage[fid].duration)]):
            if b - a >= MIN_PIECE:
                out.append(TimelineClip(fid, round(a, 3), round(b, 3)))
    return out


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
