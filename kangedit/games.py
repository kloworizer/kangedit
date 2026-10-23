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
