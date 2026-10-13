"""Clip speed: slow / fast motion and speed ramps.

A clip keeps its source range (src_in..src_out); its speed decides how long that range lasts on the
timeline. `speed` is a constant factor (0.5 = half speed slow motion, 2 = double speed); a ramp template
multiplies it along the clip, so a moment can rush in, slow right down and speed away again. Ramps are
defined over the clip's source (u = 0 at its in point, 1 at its out point), as speed keyframes joined by
straight lines.

TimeMap converts between timeline time (seconds from the clip's start) and source time (seconds from its in
point). The viewer, the export, the preview sound, captions and the timeline all use it.
"""

from __future__ import annotations

import functools
from typing import Union

import numpy as np

MIN_SPEED, MAX_SPEED = 0.1, 8.0
SPEED_PRESETS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

# key -> (label, description, [(u, speed factor), ...])
RAMPS: dict[str, tuple[str, str, list[tuple[float, float]]]] = {
    "hero": ("Hero moment", "Fast in, slow motion in the middle, fast out",
             [(0.0, 2.0), (0.3, 2.0), (0.42, 0.3), (0.62, 0.3), (0.74, 2.0), (1.0, 2.0)]),
    "bullet": ("Bullet time", "Normal speed, a near freeze in the middle, normal again",
               [(0.0, 1.0), (0.38, 1.0), (0.44, 0.12), (0.58, 0.12), (0.64, 1.0), (1.0, 1.0)]),
    "slow_to_fast": ("Slow → fast", "Starts in slow motion and speeds up",
                     [(0.0, 0.35), (0.3, 0.35), (0.65, 2.2), (1.0, 2.2)]),
    "fast_to_slow": ("Fast → slow", "Rushes in and lands in slow motion",
                     [(0.0, 2.5), (0.45, 2.5), (0.7, 0.4), (1.0, 0.4)]),
    "slow_finish": ("Slow finish", "Normal speed, easing into slow motion at the end",
                    [(0.0, 1.0), (0.55, 1.0), (0.85, 0.35), (1.0, 0.35)]),
    "speed_up": ("Speed burst", "Normal speed with a quick burst in the middle",
                 [(0.0, 1.0), (0.35, 1.0), (0.45, 3.0), (0.6, 3.0), (0.7, 1.0), (1.0, 1.0)]),
    "pulse": ("Pulse", "Fast and slow in turn, for cuts on the beat",
              [(0.0, 2.2), (0.14, 2.2), (0.2, 0.45), (0.3, 0.45), (0.36, 2.2), (0.5, 2.2), (0.56, 0.45),
               (0.66, 0.45), (0.72, 2.2), (0.86, 2.2), (0.92, 0.45), (1.0, 0.45)]),
    "montage": ("Montage", "A whip of speed into normal speed", [(0.0, 3.0), (0.25, 3.0), (0.4, 1.0), (1.0, 1.0)]),
}
SAMPLES = 256          # points of the time map along the clip

# What a clip's sound does when its speed changes: key -> (label, description)
SOUND_MODES: dict[str, tuple[str, str]] = {
    "sync": ("Synced, pitch kept", "The sound follows the picture's speed and keeps its pitch"),
    "pitch": ("Synced, pitch follows speed", "The sound follows the picture's speed like a tape: slow motion "
                                             "sounds deeper, fast motion higher"),
    "normal": ("Not synced (normal speed)", "The sound plays at normal speed from the clip's in point, "
                                            "whatever the picture's speed"),
    "mute": ("Muted", "No sound from this clip"),
}


def clamp(speed: float) -> float:
    return float(min(MAX_SPEED, max(MIN_SPEED, speed)))


def is_normal(speed: float, ramp: str) -> bool:
    return abs(speed - 1.0) < 1e-6 and ramp not in RAMPS


def label(speed: float, ramp: str) -> str:
    """Short text for the timeline ("2×", "0.5×", "Hero moment"), "" at normal speed."""
    if ramp in RAMPS:
        name = RAMPS[ramp][0]
        return name if abs(speed - 1.0) < 1e-6 else f"{name} {speed:g}×"
    return "" if abs(speed - 1.0) < 1e-6 else f"{speed:g}×"


class TimeMap:
    """Timeline time <-> source time for one clip (both measured from the clip's start / in point)."""

    def __init__(self, src_len: float, speed: float = 1.0, ramp: str = ""):
        self.src_len = max(0.0, float(src_len))
        self.speed = clamp(speed)
        self.ramp = ramp if ramp in RAMPS else ""
        if not self.ramp:
            self.duration = self.src_len / self.speed
            self._u = self._t = None
            self.v_in = self.v_out = self.speed
            return
        keys = RAMPS[self.ramp][2]
        u = np.linspace(0.0, 1.0, SAMPLES)
        v = np.interp(u, [k[0] for k in keys], [k[1] for k in keys]) * self.speed
        v = np.clip(v, MIN_SPEED, MAX_SPEED)
        # timeline time to reach u: integral of d(source) / speed (trapezoids)
        ds = self.src_len / (SAMPLES - 1)
        t = np.concatenate([[0.0], np.cumsum(ds * 0.5 * (1.0 / v[1:] + 1.0 / v[:-1]))])
        self._u, self._t, self._v = u * self.src_len, t, v
        self.duration = float(t[-1])
        self.v_in, self.v_out = float(v[0]), float(v[-1])

    @property
    def normal(self) -> bool:
        return self._u is None and abs(self.speed - 1.0) < 1e-9

    def src(self, t: Union[float, np.ndarray]):
        """Source offset shown at timeline offset t. Before the start / after the end (transition handles,
        split edits) it carries on at the edge speed."""
        if self._u is None:
            return t * self.speed
        arr = np.asarray(t, dtype=np.float64)
        out = np.interp(arr, self._t, self._u)
        out = np.where(arr < 0, arr * self.v_in, out)
        out = np.where(arr > self.duration, self.src_len + (arr - self.duration) * self.v_out, out)
        return float(out) if np.ndim(out) == 0 else out

    def time(self, s: Union[float, np.ndarray]):
        """Timeline offset where source offset s is shown (the inverse of src)."""
        if self._u is None:
            return s / self.speed
        arr = np.asarray(s, dtype=np.float64)
        out = np.interp(arr, self._u, self._t)
        out = np.where(arr < 0, arr / self.v_in, out)
        out = np.where(arr > self.src_len, self.duration + (arr - self.src_len) / self.v_out, out)
        return float(out) if np.ndim(out) == 0 else out

    def segments(self, t0: float, t1: float, step: float = 0.25) -> list[tuple[float, float, float]]:
        """The timeline span t0..t1 as pieces of (nearly) constant speed: [(source start, source end,
        speed)], for audio. One piece at a constant speed."""
        if self._u is None:
            return [(self.src(t0), self.src(t1), self.speed)]
        n = max(1, int(np.ceil((t1 - t0) / step)))
        ts = np.linspace(t0, t1, n + 1)
        ss = np.asarray(self.src(ts), dtype=np.float64)
        return [(float(a), float(b), max(MIN_SPEED, (b - a) / max(1e-9, t_b - t_a)))
                for a, b, t_a, t_b in zip(ss[:-1], ss[1:], ts[:-1], ts[1:]) if t_b > t_a]

    def setpts(self, s0: float, knots: int = 48) -> str:
        """ffmpeg setpts expression turning source time T (seconds from source offset s0) into timeline time
        (seconds from the timeline time where s0 shows): straight lines between `knots` points."""
        if self._u is None:
            return f"(PTS-STARTPTS)/{self.speed:.6f}"
        # Knots over the clip; past its out point (transition handles) the last term runs on at the end speed.
        s = np.linspace(s0, max(self.src_len, s0 + 1e-3), knots + 1)
        t = np.asarray(self.time(s), dtype=np.float64) - float(self.time(s0))
        rel = s - s0
        expr = f"{t[-1]:.6f}+(T-{rel[-1]:.6f})/{self.v_out:.6f}"
        for k in range(knots - 1, -1, -1):           # nested ifs, last piece innermost
            slope = (t[k + 1] - t[k]) / max(1e-9, rel[k + 1] - rel[k])
            expr = f"if(lt(T,{rel[k + 1]:.6f}),{t[k]:.6f}+(T-{rel[k]:.6f})*{slope:.6f},{expr})"
        return f"({expr})/TB"


@functools.lru_cache(maxsize=512)
def timemap(src_len: float, speed: float, ramp: str) -> TimeMap:
    return TimeMap(round(src_len, 6), speed, ramp)


def sound_mode(mode: str) -> str:
    return mode if mode in SOUND_MODES else "sync"


def stretch(speed: float, keep_pitch: bool = True) -> str:
    """Audio filters playing 48 kHz sound at `speed`: time-stretched with its pitch kept (atempo), or resampled
    so the pitch moves with the speed; "" at normal speed."""
    if abs(speed - 1.0) < 1e-4:
        return ""
    if keep_pitch:
        return atempo(speed)
    return f"asetrate={int(round(48000 * speed))},aresample=48000"


def atempo(speed: float) -> str:
    """atempo filters (each 0.5..2) for a speed, pitch kept; "" at normal speed."""
    if abs(speed - 1.0) < 1e-4:
        return ""
    parts, s = [], speed
    while s > 2.0:
        parts.append(2.0)
        s /= 2.0
    while s < 0.5:
        parts.append(0.5)
        s /= 0.5
    parts.append(s)
    return ",".join(f"atempo={p:.6f}" for p in parts)
