"""Gaming moments, for Auto Edit's *Prioritize: Gaming* (see autoedit.score_samples): a kill in a shooter, a
double / triple kill or a teamfight in a MOBA, the reveal of a gacha pull, a goal, a boss going down.

Nothing here knows a game: every game marks its big moments the same few ways, measured against what is usual
in that recording (a loud game is loud all the time; only what stands out of it counts):

* the sound bursts: gunshots, an explosion, the announcer ("Double kill!", "Savage!"), a reveal sting;
* the picture flashes or floods with colour: hit flashes, a kill banner, a golden / rainbow gacha reveal;
* the whole screen changes at once: an explosion, a teamfight, the reveal animation;
* the player reacts: shouting, laughing, cheering (with Smarter analysis);
* bursts come in clusters: several kills close together (a multi-kill) count more than one;
* a burst after a calm stretch: the gacha spin, then the reveal; the wait before a play, then the play.

The score is spread from a little before the burst (the lead-up) to a few seconds after it (the reaction, the
announcer, the result), so the clips Auto Edit picks start before the moment and show how it ends.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import smart
from .analysis import SAMPLE_RATE, Analysis

BASELINE = 12.0        # seconds of a recording a moment is compared with (its usual level)
CLUSTER = 6.0          # seconds in which several bursts make a multi-kill
CALM = 6.0             # seconds of calm before a burst that make it a payoff (a gacha reveal)
LEAD, TAIL = 1.5, 3.0  # seconds a moment's score reaches before and after its burst
LOUDER_DB = 8.0        # this much louder than usual: a full burst
NOISE = 2.5            # rises within this many times the recording's ordinary ups and downs don't count


def _local(x: np.ndarray, seconds: float) -> np.ndarray:
    """x's usual level around each sample: the median of the `seconds` around it."""
    w = max(3, int(seconds * SAMPLE_RATE) | 1)
    pad = np.pad(x.astype(np.float32), w // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)


def _stand_out(x: np.ndarray, scale: float) -> np.ndarray:
    """How far x rises above its usual level and the recording's ordinary ups and downs, 0..1 (1: `scale`
    above them)."""
    rise = x - _local(x, BASELINE)
    noise = 1.4826 * float(np.median(np.abs(rise)))              # (a robust standard deviation)
    return np.clip((rise - NOISE * noise) / max(scale, 1e-6), 0.0, 1.0)


def _spread(x: np.ndarray) -> float:
    """A recording's own range of x (between its quiet and its busy moments)."""
    lo, hi = np.percentile(x, [10, 90])
    return float(hi - lo)


def _window(x: np.ndarray, before: float, after: float, how=np.max) -> np.ndarray:
    """how() of x over each sample's window from `before` seconds before it to `after` seconds after."""
    b, a = int(round(before * SAMPLE_RATE)), int(round(after * SAMPLE_RATE))
    pad = np.pad(x, (b, a), mode="constant")
    return how(np.lib.stride_tricks.sliding_window_view(pad, b + a + 1), axis=1)


def bursts(a: Analysis, has_audio: bool = True) -> np.ndarray:
    """How strongly each sample is a burst (0..1): what stands out of the recording's usual sound and picture,
    and the player's reaction."""
    n = len(a.times)
    if n < 8:
        return np.zeros(n, np.float32)
    loud = (_stand_out(np.maximum(np.asarray(a.audio, np.float32), -90.0), LOUDER_DB)
            if has_audio else np.zeros(n, np.float32))
    act = np.asarray(a.activity, np.float32)
    change = _stand_out(act, 1.5 * _spread(act) + 1e-3)
    flash = _stand_out(np.asarray(a.brightness, np.float32), 0.15)
    color = np.asarray(a.colorful, np.float32)
    flood = _stand_out(color, 1.5 * _spread(color) + 1.0)
    picture = np.maximum(flash, flood)
    react = np.zeros(n, np.float32)
    sound = getattr(a, "sound", None)
    if sound is not None and len(sound):
        get = lambda name: smart.at_samples(sound, name, a.times)  # noqa: E731
        react = np.clip(np.maximum.reduce([get("excited"), get("laughter"), get("applause")]) * 1.5, 0.0, 1.0)
    # Sound and picture together make the strongest burst; either alone still counts.
    burst = 0.45 * loud + 0.25 * change + 0.30 * picture + 0.20 * loud * np.maximum(change, picture)
    return np.clip(np.maximum(burst, 0.8 * react) + 0.25 * react * burst, 0.0, 1.0).astype(np.float32)


def moments(a: Analysis, has_audio: bool = True) -> np.ndarray:
    """The gaming-moment score of each sample (0..1): bursts, more when several come close together (a
    multi-kill) or after a calm stretch (a reveal), spread over the moment's lead-up and its aftermath."""
    b = bursts(a, has_audio)
    if not b.size:
        return b
    up = b > 0.35
    starts = (up & ~np.r_[False, up[:-1]]).astype(np.float32)          # where each burst begins
    # separate bursts within CLUSTER seconds: 0 for one, 0.5 for two, 1 for three or more (a triple kill)
    many = np.clip((_window(starts, CLUSTER / 2, CLUSTER / 2, np.sum) - 1.0) / 2.0, 0.0, 1.0)
    calm_before = 1.0 - np.clip(_window(b, CALM, 0.0, np.mean) / 0.25, 0.0, 1.0)
    calm_before = np.r_[np.zeros(1, np.float32), calm_before[:-1]]        # (the burst itself not counted)
    m = b * (0.65 + 0.35 * many) + 0.3 * b * calm_before
    # spread over the moment: a sample scores for a burst LEAD s after it ... TAIL s before it
    return np.clip(_window(m, TAIL, LEAD), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------- labelling the plays
# Tools → Label the gaming plays: a title at each multi-kill of the edit ("DOUBLE KILL", "TRIPLE KILL"...).
LABEL_PRESET = "game_label"   # TextClip.preset of the labels (so labelling again replaces them)
LABEL_SECONDS = 1.5           # how long a label shows
BURST_ON = 0.35               # a burst: bursts() above this (as moments() counts them)
BURST_JOIN = 0.3              # seconds: dips shorter than this don't split a burst in two
MULTI_GAP = 4.0               # seconds: bursts each within this of the one before are one multi-kill
# The words for 1, 2, 3 and 4 or more bursts in a row ({n}: how many). Generic, no game's own announcer words.
LABEL_WORDS = {1: "KILL", 2: "DOUBLE KILL", 3: "TRIPLE KILL", 4: "MULTI KILL ×{n}"}


def burst_times(a: Analysis, has_audio: bool = True, t0: float = 0.0, t1: float = float("inf")) -> list[float]:
    """Where each separate burst begins (seconds of the recording), between t0 and t1."""
    b = bursts(a, has_audio)
    if not b.size:
        return []
    up = b > BURST_ON
    join = max(1, int(round(BURST_JOIN * SAMPLE_RATE)))
    times = np.asarray(a.times, np.float64)
    out: list[float] = []
    last_end = -10 ** 9
    i, n = 0, len(up)
    while i < n:
        if not up[i]:
            i += 1
            continue
        j = i
        while j < n and up[j]:
            j += 1
        if i - last_end > join and t0 - 1e-6 <= times[i] <= t1 + 1e-6:
            out.append(float(times[i]))          # a new burst (not a flicker of the one before)
        last_end = j
        i = j
    return out


def clusters(times: list[float], gap: float = MULTI_GAP) -> list[list[float]]:
    """Burst times grouped into plays: each burst within `gap` seconds of the one before joins its play."""
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def label_text(n: int, words: Optional[dict] = None) -> str:
    """The label for a play of n bursts ("" for none). words: {1..4: text} over LABEL_WORDS."""
    words = {**LABEL_WORDS, **{int(k): v for k, v in (words or {}).items()}}
    text = words.get(min(n, 4), "") or ""
    return text.replace("{n}", str(n)).strip()


def play_labels(project, analyses: dict, words: Optional[dict] = None, singles: bool = False) -> list:
    """Titles (TextClips, preset LABEL_PRESET) at the edit's multi-kills: the bursts of each clip's footage
    within the clip's range, grouped into plays; a label at the play's last burst on the timeline (where the
    count is reached), LABEL_SECONDS long. singles: also label single bursts. analyses: footage id ->
    Analysis (clips without one are skipped)."""
    from .titles import make_text
    out = []
    total = project.duration
    last = -10.0
    for c, start in zip(project.clips, project.clip_starts()):
        f = project.footage.get(c.footage_id)
        a = analyses.get(c.footage_id)
        if f is None or a is None or f.is_image:
            continue
        for play in clusters(burst_times(a, f.has_audio, c.src_in, c.src_out)):
            if len(play) < (1 if singles else 2):
                continue
            text = label_text(len(play), words)
            at = start + c.time_at(play[-1])
            if not text or at - last < 0.5 or at >= total - 0.1:
                continue
            tc = make_text("callout", at, total)
            tc.text, tc.preset = text, LABEL_PRESET
            tc.duration = min(LABEL_SECONDS, max(0.2, total - tc.start))
            tc.anim_dur = 0.25
            out.append(tc)
            last = at
    return out


def apply_labels(project, analyses: dict, words: Optional[dict] = None, singles: bool = False) -> int:
    """Replace the edit's play labels with new ones (see play_labels). Returns how many were added."""
    labels = play_labels(project, analyses, words, singles)
    project.texts = [x for x in project.texts if x.preset != LABEL_PRESET] + labels
    project.dirty = True
    return len(labels)
