"""Voice clean-up: noise and wind taken out of a clip's speech, in the preview sound and the export alike (the
same FFmpeg filters in both): a high-pass for the rumble of wind and handling, then adaptive FFT noise
reduction that follows the background noise as it changes. Clips where the smarter analysis heard wind can
have it switched on at once."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import smart

DEFAULT = 0.6                     # strength switched on by hand or for wind
WIND_HEARD = 0.2                  # the smarter analysis's wind score (mean over the clip) that asks for it
UNKNOWN_FLOOR = -40.0             # dBFS: the background noise level taken when it isn't measured


def ffmpeg_filter(strength: float, floor: float = 0.0) -> str:
    """Filters (ending with a comma) cleaning up the voice at `strength` (0..1); "" when off. floor: the clip's
    background noise level (dBFS, see noise_floor; 0 = unknown): the noise reduction works on noise near it."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return ""
    cut = 70 + 80 * s                 # Hz: wind rumble and handling noise are below the voice
    nr = 6 + 18 * s                   # dB of noise taken off
    nf = float(np.clip(floor if floor < 0 else UNKNOWN_FLOOR, -80.0, -20.0))
    return f"highpass=f={cut:.0f}:p=2,afftdn=nr={nr:.1f}:nf={nf:.1f}:tn=1,"


def noise_floor(analysis, a: float, b: float) -> float:
    """The background noise level in source time a..b (dBFS): the quiet moments' loudness. 0 when unknown."""
    if analysis is None or not len(getattr(analysis, "audio", ())):
        return 0.0
    t = np.asarray(analysis.times)
    db = np.asarray(analysis.audio)[(t >= a) & (t < b)]
    db = db[db > -90]
    if len(db) < 4:
        return 0.0
    return float(np.clip(np.percentile(db, 10), -80.0, -20.0))


def wind_in(analysis, a: float, b: float) -> float:
    """How much wind the smarter analysis heard in source time a..b (0..1; 0 without it)."""
    sound = getattr(analysis, "sound", None)
    if analysis is None or sound is None or not len(sound):
        return 0.0
    t = np.asarray(analysis.times)
    sel = t[(t >= a) & (t < b)]
    if not len(sel):
        return 0.0
    w = smart.at_samples(sound, "wind", sel)
    return 0.0 if w is None else float(np.mean(w))


def suggest(clip, analysis) -> Optional[float]:
    """The strength to clean a clip's voice with when wind was heard in it, else None."""
    return DEFAULT if wind_in(analysis, clip.src_in, clip.src_out) >= WIND_HEARD else None
