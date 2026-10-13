"""Transitions and split (J/L) edits between timeline clips.

A transition is centered on the cut and uses source handles: the outgoing clip runs on `half` seconds
past its out point and the incoming one starts `half` seconds before its in point, so clip positions,
the timeline length and the captions don't move. A split edit moves only the audio cut: a J-cut
(split < 0) brings the incoming clip's sound in early, an L-cut (split > 0) lets the outgoing clip's
sound run on under the next picture. The export, the preview audio and the viewer all use plan().
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Project

# key -> (label, ffmpeg xfade name)
TRANSITIONS = {
    "dissolve": ("Cross dissolve", "fade"),
    "dip_black": ("Dip to black", "fadeblack"),
    "dip_white": ("Dip to white", "fadewhite"),
    "wipe": ("Wipe", "wipeleft"),
    "slide": ("Slide", "slideleft"),
    "zoom": ("Zoom in", "zoomin"),
}
MIN_TRANSITION = 0.1     # shorter (for lack of handles) means a plain cut
MAX_SPLIT_SHARE = 0.45   # a split may take at most this share of either clip


@dataclass
class Edit:
    """The edit into clip k from clip k-1 (edits()[0] is the start of the timeline)."""
    trans: str = ""      # TRANSITIONS key or "" for a cut
    half: float = 0.0    # transition half-length, seconds
    split: float = 0.0   # audio cut offset, seconds (< 0 J-cut, > 0 L-cut); 0 with a transition


def edits(project: Project) -> list[Edit]:
    """Effective edits, clamped to the source handles and clip lengths that are available."""
    clips = project.clips
    out = [Edit()]
    for k in range(1, len(clips)):
        a, b = clips[k - 1], clips[k]
        fa, fb = project.footage.get(a.footage_id), project.footage.get(b.footage_id)
        if fa is None or fb is None:
            out.append(Edit())
            continue
        # Source handles are spent at the clips' edge speeds (slow motion uses less of them per second).
        ma, mb = a.timemap(), b.timemap()
        after_a, before_b = (fa.duration - a.src_out) / ma.v_out, b.src_in / mb.v_in
        if b.transition in TRANSITIONS:
            half = min(b.transition_dur / 2, after_a, before_b, a.duration / 2, b.duration / 2)
            if 2 * half >= MIN_TRANSITION:
                out.append(Edit(b.transition, half))
                continue
        s = b.audio_split
        if s < 0:      # J: the incoming sound starts before its in point, under the outgoing picture
            s = max(s, -before_b, -a.duration * MAX_SPLIT_SHARE, -b.duration * MAX_SPLIT_SHARE)
        elif s > 0:    # L: the outgoing sound continues past its out point, under the incoming picture
            s = min(s, after_a, a.duration * MAX_SPLIT_SHARE, b.duration * MAX_SPLIT_SHARE)
        out.append(Edit("", 0.0, s if abs(s) >= 0.04 else 0.0))
    return out


@dataclass
class ClipPlan:
    start_f: int      # first timeline frame
    nfr: int          # timeline frames
    pre_f: int        # handle frames before the in point (transition into this clip)
    post_f: int       # handle frames after the out point (transition out of it)
    v_in: float       # source time of the first rendered video frame
    a_in: float       # source time where this clip's audio starts
    a_dur: float      # audio length, seconds
    xfade: str        # ffmpeg xfade name for the transition into this clip ("" = cut)
    v_end: float = 0.0   # source time just after the last rendered video frame
    a_end: float = 0.0   # source time where this clip's audio ends
    a_t0: float = 0.0    # timeline offset (from the clip's start) where its audio starts

    @property
    def v_frames(self) -> int:
        return self.nfr + self.pre_f + self.post_f


def plan(project: Project, fps: float) -> list[ClipPlan]:
    """Frame-exact video and audio ranges for every clip (clip positions rounded to whole frames)."""
    fps = max(1.0, fps)
    es = edits(project) + [Edit()]
    out, t = [], 0.0
    for k, c in enumerate(project.clips):
        start_f, end_f = int(round(t * fps)), int(round((t + c.duration) * fps))
        t += c.duration
        nfr = max(1, end_f - start_f)
        e_in, e_out = es[k], es[k + 1]
        pre_f, post_f = int(round(e_in.half * fps)), int(round(e_out.half * fps))
        # Audio extends over the transition handles, or shifts with the split edits.
        a_pre = pre_f / fps if pre_f else -e_in.split
        a_post = post_f / fps if post_f else e_out.split
        a_dur = max(0.02, nfr / fps + a_pre + a_post)
        out.append(ClipPlan(start_f, nfr, pre_f, post_f, c.src_at(-pre_f / fps), c.sound_src_at(-a_pre), a_dur,
                            TRANSITIONS[e_in.trans][1] if pre_f else "",
                            c.src_at((nfr + post_f) / fps), c.sound_src_at(a_dur - a_pre), -a_pre))
    return out


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def blend(kind: str, a: np.ndarray, b: np.ndarray, p: float) -> np.ndarray:
    """Preview of the ffmpeg xfade `kind` at progress p (0 = all outgoing, 1 = all incoming)."""
    p = min(1.0, max(0.0, p))
    h, w = a.shape[:2]
    if kind == "wipeleft":            # the incoming clip is revealed from the right edge
        out = a.copy()
        x = int(round(w * (1 - p)))
        out[:, x:] = b[:, x:]
        return out
    if kind == "slideleft":           # both move left; the incoming clip enters from the right
        off = int(round(w * p))
        out = np.empty_like(a)
        out[:, :w - off] = a[:, off:]
        out[:, w - off:] = b[:, :off]
        return out
    if kind == "zoomin":              # zoom into the outgoing clip, then dissolve
        z = 1.0 + smoothstep(p * 2) * 0.6
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, z)
        az = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(az, 1 - smoothstep((p - 0.5) * 2), b, smoothstep((p - 0.5) * 2), 0)
    mix = cv2.addWeighted(a, 1 - p, b, p, 0)
    if kind in ("fadeblack", "fadewhite"):
        k = 0.25 + 0.75 * abs(2 * p - 1)        # darkest (brightest) at the middle
        if kind == "fadeblack":
            return cv2.convertScaleAbs(mix, alpha=k)
        return cv2.addWeighted(mix, k, np.full_like(mix, 255), 1 - k, 0)
    return mix
