"""Syncing camera angles (or any recordings of the same moment) by their sound.

Two passes, as dedicated sync tools do:

1. coarse: an onset envelope of each recording (how fast its loudness rises, 100 values a second) is
   cross-correlated over the whole length. Onsets (words, claps, steps) line up whatever each microphone's
   level and tone, so this finds where the recordings overlap to about 10 ms;
2. fine: short windows spread over the overlap are matched sample by sample (GCC-PHAT on 8 kHz audio,
   sub-sample peak), and a straight line is fitted through their offsets. Its slope is the clock drift
   between the cameras (tens of parts per million, a few frames over an hour); windows that disagree
   (silence, a sound only one camera heard) are left out.

The result maps a time in recording B to the same moment in recording A (see SyncResult).
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .ffmpeg import Cancelled, ffmpeg_bin, popen

RATE = 8000              # audio samples per second used for syncing
ENV_RATE = 100           # onset-envelope values per second (coarse pass)
WINDOW = 6.0             # seconds of each fine-pass window
SEARCH = 0.25            # seconds searched either side of the coarse offset in the fine pass
MAX_WINDOWS = 40
MIN_Z = 6.0              # coarse peak this many standard deviations above the rest: a real match
ProgressCb = Callable[[float, str], None]


@dataclass
class SyncResult:
    """Recording B against recording A: A's time = offset + B's time * (1 + drift)."""
    offset: float            # where B's start falls on A's clock, seconds (negative: B started first)
    drift: float = 0.0       # B's clock runs (1 + drift) times A's
    confidence: float = 0.0  # 0..1: how strongly and consistently the sound matched
    overlap: float = 0.0     # seconds both recordings cover
    windows: int = 0         # fine-pass windows that agreed

    def a_time(self, b_time: float) -> float:
        return self.offset + b_time * (1.0 + self.drift)

    def b_time(self, a_time: float) -> float:
        return (a_time - self.offset) / (1.0 + self.drift)

    @property
    def reliable(self) -> bool:
        return self.confidence >= 0.35 and self.overlap > 1.0


def load_audio(path: str, cancel: Optional[threading.Event] = None) -> np.ndarray:
    """The file's sound as mono float32 at RATE (empty when it has none)."""
    exe = ffmpeg_bin()
    if not exe:
        raise OSError("ffmpeg was not found")
    proc = popen([exe, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(RATE),
                  "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunks = []
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            buf = proc.stdout.read(1 << 20)
            if not buf:
                break
            chunks.append(buf)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    raw = b"".join(chunks)
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def envelope(audio: np.ndarray, rate: int = RATE) -> np.ndarray:
    """Onset strength at ENV_RATE: the rise of the log energy, normalized (mean 0, deviation 1)."""
    hop = rate // ENV_RATE
    n = len(audio) // hop
    if n < 4:
        return np.zeros(max(0, n), np.float32)
    frames = audio[: n * hop].reshape(n, hop).astype(np.float64)
    loud = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-9)
    onset = np.maximum(0.0, np.diff(loud, prepend=loud[0]))
    onset = onset - np.convolve(onset, np.ones(50) / 50.0, mode="same")   # local mean off: rhythm, not level
    sd = onset.std()
    return ((onset - onset.mean()) / sd if sd > 0 else onset * 0).astype(np.float32)


def _xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c[k + len(b) - 1] = sum_t a[t + k] * b[t], for k in -(len(b)-1) .. len(a)-1."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    return np.concatenate([c[n - (len(b) - 1):], c[: len(a)]])


def coarse_offset(env_a: np.ndarray, env_b: np.ndarray, min_overlap: float = 2.0) -> tuple[float, float]:
    """(offset of B on A's clock in seconds, z-score of the peak) from the onset envelopes."""
    if len(env_a) < 8 or len(env_b) < 8:
        return 0.0, 0.0
    c = _xcorr(env_a, env_b)
    lags = np.arange(-(len(env_b) - 1), len(env_a))
    overlap = np.minimum(len(env_a), lags + len(env_b)) - np.maximum(0, lags)
    keep = overlap >= min_overlap * ENV_RATE
    if not keep.any():
        return 0.0, 0.0
    # Normalize by the overlap so short overlaps at the ends don't win by chance, but not fully (noisy).
    score = np.where(keep, c / np.sqrt(np.maximum(overlap, 1)), -np.inf)
    i = int(np.argmax(score))
    valid = score[keep]
    z = float((score[i] - np.median(valid)) / (valid.std() + 1e-9))
    return float(lags[i]) / ENV_RATE, z


def gcc_phat(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Where x occurs in the longer y (samples, sub-sample precise) and the peak's sharpness (peak over the
    next highest away from it)."""
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    r = np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n))
    r /= np.abs(r) ** 0.8 + 1e-9                  # PHAT weighting (slightly softened: steadier on noise)
    cc = np.fft.irfft(r, n)[: len(y) - len(x) + 1]
    i = int(np.argmax(cc))
    shift = float(i)
    if 0 < i < len(cc) - 1:                       # parabolic peak: sub-sample position
        a, b, c = cc[i - 1], cc[i], cc[i + 1]
        den = a - 2 * b + c
        if den < 0:
            shift += 0.5 * (a - c) / den
    rest = np.concatenate([cc[: max(0, i - 40)], cc[i + 41:]])
    sharp = float(cc[i] / (np.max(rest) + 1e-12)) if len(rest) and np.max(rest) > 0 else 10.0
    return shift, sharp


def sync_audio(a: np.ndarray, b: np.ndarray, rate: int = RATE,
               cancel: Optional[threading.Event] = None) -> SyncResult:
    """Sync recording b to recording a (mono float arrays at `rate`)."""
    la, lb = len(a) / rate, len(b) / rate
    # A match must share a real stretch of sound (not a couple of seconds at an edge, which noise can fake).
    need = min(10.0, 0.5 * min(la, lb))
    offset, z = coarse_offset(envelope(a, rate), envelope(b, rate), need)
    lo, hi = max(0.0, offset), min(la, offset + lb)
    overlap = max(0.0, hi - lo)
    if overlap < 1.0 or z < MIN_Z * 0.5:
        return SyncResult(offset, 0.0, 0.0, overlap)
    win = min(WINDOW, max(1.0, overlap - 2 * SEARCH))
    span = max(0.0, overlap - win - 2 * SEARCH)
    n = int(max(1, min(MAX_WINDOWS, span // 20 + 1)))
    starts = [lo + SEARCH + span * (k + 0.5) / n for k in range(n)] if span > 0 else [lo + SEARCH * 0]
    points = []                                   # (time in a, offset measured there)
    w = int(win * rate)
    m = int(SEARCH * rate)
    for ta in starts:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        ia = int(ta * rate)
        x = a[ia: ia + w]
        ib = int((ta - offset) * rate) - m
        y = b[max(0, ib): ib + w + 2 * m]
        pad = max(0, -ib)
        if len(x) < w or len(y) < w + 2 * m - pad or float(np.sqrt(np.mean(x * x))) < 1e-3:
            continue
        shift, sharp = gcc_phat(x, y)
        if sharp < 1.3:
            continue
        tb = (max(0, ib) + shift) / rate          # where x starts in b
        points.append((ta, ta - tb, sharp))
    if not points:
        conf = float(np.clip((z - MIN_Z * 0.5) / (MIN_Z * 2), 0.0, 0.5))
        return SyncResult(offset, 0.0, conf, overlap)
    p = np.asarray(points)
    # The windows that agree with each other most (scattered offsets mean no real match there).
    tol = 0.02 + 1e-4 * overlap
    near = np.abs(p[:, 1][:, None] - p[:, 1][None, :]) < tol
    p = p[near[int(np.argmax(near.sum(axis=1)))]]
    o, d = float(np.median(p[:, 1])), 0.0
    if len(p) >= 3 and np.ptp(p[:, 0]) > 30.0:
        for _ in range(2):                        # line fit, then again without the windows that stray
            k, c0 = np.polyfit(p[:, 0], p[:, 1], 1)
            res = p[:, 1] - (k * p[:, 0] + c0)
            keep = np.abs(res) < 0.003
            if keep.sum() < 3 or keep.all():
                break
            p = p[keep]
        k, c0 = np.polyfit(p[:, 0], p[:, 1], 1)
        # offset(ta) = ta - tb = c0 + k*ta  ->  ta = (c0 + tb) / (1 - k)
        o, d = float(c0 / (1.0 - k)), float(1.0 / (1.0 - k) - 1.0)
    agree = len(p) / max(1, len(starts)) * min(1.0, len(p) / min(3, len(starts)))
    conf = float(np.clip(0.5 * agree + 0.5 * min(1.0, z / (MIN_Z * 2)), 0.0, 1.0))
    return SyncResult(o, d, conf, overlap, len(p))


def sync_files(reference: list[str], others: list[str], progress: Optional[ProgressCb] = None,
               cancel: Optional[threading.Event] = None) -> dict[str, tuple[str, SyncResult]]:
    """For each file in `others`, the reference file it overlaps best and how it syncs to it
    ({other path: (reference path, SyncResult)}; files that match no reference are left out)."""
    total = len(reference) + len(others)
    audio: dict[str, np.ndarray] = {}
    for k, path in enumerate(dict.fromkeys(reference + others)):
        if progress:
            progress(0.6 * k / max(1, total), "Reading the sound of each angle…")
        audio[path] = load_audio(path, cancel)
    out = {}
    for k, other in enumerate(others):
        best = None
        for ref in reference:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if progress:
                progress(0.6 + 0.4 * k / max(1, len(others)), "Syncing the angles by their sound…")
            if len(audio[ref]) < RATE or len(audio[other]) < RATE:
                continue
            r = sync_audio(audio[ref], audio[other], RATE, cancel)
            if r.reliable and (best is None or (r.confidence * r.overlap) > (best[1].confidence * best[1].overlap)):
                best = (ref, r)
        if best is not None:
            out[other] = best
    return out


def compose(x_on_hub: SyncResult, m_on_hub: SyncResult) -> SyncResult:
    """Recording X against recording M, from both synced to a third one (the hub): X's time on M's clock."""
    k = 1.0 + m_on_hub.drift
    return SyncResult((x_on_hub.offset - m_on_hub.offset) / k, (1.0 + x_on_hub.drift) / k - 1.0,
                      min(x_on_hub.confidence, m_on_hub.confidence))


def sync_to_longest(mains: list[str], others: list[str], durations: dict[str, float],
                    progress: Optional[ProgressCb] = None,
                    cancel: Optional[threading.Event] = None) -> dict[str, tuple[str, SyncResult]]:
    """Like sync_files, but every recording is synced to the longest one (which usually spans all the others)
    and each other angle's offset to a main-angle file is worked out through it: {other path: (main path,
    SyncResult)}. Files that don't match the longest recording are left out (the caller can sync them
    directly)."""
    paths = list(dict.fromkeys(mains + others))
    if not paths:
        return {}
    hub = max(paths, key=lambda p: durations.get(p, 0.0))
    audio: dict[str, np.ndarray] = {}
    for k, path in enumerate(paths):
        if progress:
            progress(0.6 * k / len(paths), "Reading the sound of each angle…")
        audio[path] = load_audio(path, cancel)
    on_hub: dict[str, SyncResult] = {hub: SyncResult(0.0, 0.0, 1.0, durations.get(hub, 0.0))}
    rest = [p for p in paths if p != hub]
    for k, path in enumerate(rest):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress:
            progress(0.6 + 0.4 * k / max(1, len(rest)), "Syncing the angles to the longest recording…")
        if len(audio[hub]) < RATE or len(audio[path]) < RATE:
            continue
        r = sync_audio(audio[hub], audio[path], RATE, cancel)
        if r.reliable:
            on_hub[path] = r
    out = {}
    for other in others:
        if other not in on_hub:
            continue
        best = None
        for main in mains:
            if main not in on_hub or main == other:
                continue
            r = compose(on_hub[other], on_hub[main])
            lo = max(0.0, r.offset)
            hi = min(durations.get(main, 0.0), r.a_time(durations.get(other, 0.0)))
            r.overlap = max(0.0, hi - lo)
            if r.overlap > 1.0 and (best is None or r.overlap * r.confidence > best[1].overlap * best[1].confidence):
                best = (main, r)
        if best is not None:
            out[other] = best
    return out
