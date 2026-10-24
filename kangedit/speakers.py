"""Who speaks each caption (podcasts, interviews), offline and without a speech model.

Each caption's stretch of sound is described by a few simple voice features: the median pitch of its
voiced moments, the spectral centroid and the share of the energy in a handful of bands (a crude voice
"colour"). The captions are then grouped into speakers with k-means. The speaker is written onto the
transcript words ("p": 0, 1, ...) so it stays through rebuilds, trims, saves and undo, like every other
caption edit. Two clearly different voices separate well; similar voices may need a fix by hand
(right-click a caption → Speaker).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .ffmpeg import Cancelled
from .models import Project

RATE = 16000
FRAME = 640              # 40 ms analysis frames...
HOP = 160                # ...every 10 ms
MIN_F0, MAX_F0 = 60.0, 400.0
BANDS = (0, 150, 300, 500, 800, 1200, 1800, 2700, 4000, 8000)   # Hz, roughly the ear's (mel) spacing
MAX_SPEAKERS = 4
MODES = {"off": "Off", "auto": "Auto", "2": "2 speakers", "3": "3 speakers", "4": "4 speakers"}
AUTO_MIN_SILHOUETTE = 0.25   # Auto: groups less distinct than this are one voice


def voice_features(audio: np.ndarray, rate: int = RATE) -> Optional[np.ndarray]:
    """[log median pitch, log spectral centroid, log band shares...] of the voiced part of 16 kHz audio, or
    None when it holds too little voice. The pitch is nan when no frame is clearly periodic (a whisper,
    noise): filled in by the caller."""
    if audio.size < FRAME * 2:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(audio.astype(np.float32), FRAME)[::HOP]
    energy = np.mean(frames ** 2, axis=1) + 1e-12
    db = 10 * np.log10(energy)
    loud = db >= max(float(np.max(db)) - 25.0, -55.0)      # the voice, not the pauses between words
    frames = frames[loud]
    if len(frames) < 5:
        return None
    frames = frames - frames.mean(axis=1, keepdims=True)
    win = np.hanning(FRAME).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * win, 1024, axis=1)) ** 2
    freqs = np.fft.rfftfreq(1024, 1.0 / rate)
    total = spec.sum(axis=1) + 1e-12
    centroid = float(np.median((spec * freqs).sum(axis=1) / total))
    shares = [np.log(spec[:, (freqs >= a) & (freqs < b)].sum(axis=1) / total + 1e-6).mean()
              for a, b in zip(BANDS, BANDS[1:])]
    # Pitch by autocorrelation: the first lag (shortest period) nearly as alike as the best one, so a
    # voice isn't taken an octave too low.
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames, 2 * FRAME, axis=1)) ** 2, axis=1)[:, :FRAME]
    lo, hi = int(rate / MAX_F0), int(rate / MIN_F0)
    seg = ac[:, lo:hi + 1]
    peak = seg.max(axis=1)
    voiced = peak > 0.45 * ac[:, 0]
    f0 = []
    for row, p in zip(seg[voiced], peak[voiced]):
        k = int(np.flatnonzero(row >= 0.9 * p)[0])
        f0.append(rate / (lo + k))
    pitch = float(np.log(np.median(f0))) if len(f0) >= 3 else float("nan")
    return np.array([pitch, np.log(max(centroid, 1.0)), *shares], np.float64)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for j in range(x.shape[1]):              # a caption without pitch: the others' median
        col = x[:, j]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = np.median(col[~bad])
    sd = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
    x[:, 0] *= 2.0                           # the pitch tells voices apart best
    return x


def kmeans(x: np.ndarray, k: int, seed: int = 0, restarts: int = 8, iters: int = 60) -> np.ndarray:
    """Labels 0..k-1 of the rows of x (k-means, k-means++ starts, the best of `restarts`)."""
    n = len(x)
    if k <= 1 or n <= 1:
        return np.zeros(n, int)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    best, best_cost = np.zeros(n, int), np.inf
    for _ in range(restarts):
        centers = [x[rng.integers(n)]]
        for _ in range(1, k):
            d = np.min([((x - c) ** 2).sum(axis=1) for c in centers], axis=0)
            centers.append(x[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else x[rng.integers(n)])
        c = np.array(centers)
        for _ in range(iters):
            lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(axis=2), axis=1)
            new = np.array([x[lab == j].mean(axis=0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        cost = float(((x - c[lab]) ** 2).sum())
        if cost < best_cost:
            best, best_cost = lab, cost
    return best


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette of a grouping (-1..1: how much nearer each row is to its own group than the next)."""
    groups = np.unique(labels)
    if len(groups) < 2:
        return -1.0
    d = np.sqrt(((x[:, None, :] - x[None]) ** 2).sum(axis=2))
    s = []
    for i in range(len(x)):
        own = labels == labels[i]
        if own.sum() <= 1:
            s.append(0.0)
            continue
        a = d[i, own].sum() / (own.sum() - 1)
        b = min(d[i, labels == g].mean() for g in groups if g != labels[i])
        s.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(s))


def cluster(features: np.ndarray, mode: str) -> np.ndarray:
    """Speaker labels for the rows (captions) of `features`, numbered in order of first appearance. mode:
    "2" / "3" / "4", or "auto": 2 or 3 groups, whichever is more distinct, or one voice when neither is."""
    n = len(features)
    if n == 0:
        return np.zeros(0, int)
    x = _standardize(np.asarray(features, np.float64))
    if mode == "auto":
        best, score = np.zeros(n, int), AUTO_MIN_SILHOUETTE
        for k in (2, 3):
            if n > k:
                lab = kmeans(x, k)
                s = silhouette(x, lab)
                if s > score + (0.05 if k == 3 else 0.0):     # a third voice only when clearly better
                    best, score = lab, s
        labels = best
    else:
        labels = kmeans(x, max(1, min(MAX_SPEAKERS, int(mode))))
    order: dict[int, int] = {}
    for lab in labels:
        order.setdefault(int(lab), len(order))
    return np.array([order[int(lab)] for lab in labels], int)


@dataclass
class SpeakerUnit:
    """A caption's words to tell the speaker of: transcript words `idx` of `footage_id` (their text, to check
    the transcript hasn't changed), heard in `path` from source second s to e."""
    footage_id: str
    path: str
    idx: list
    texts: list
    s: float
    e: float
    group: str = ""                          # units of one clip: their sound is read in one go
    features: Optional[np.ndarray] = field(default=None, repr=False)


def speaker_units(project: Project) -> list[SpeakerUnit]:
    """The captions on the timeline as SpeakerUnits (made on the UI thread: the job only reads these)."""
    from .captions import build_captions
    out = []
    for cap in build_captions(project):
        f = project.transcript_source(cap.footage_id)
        idx = [k for k in cap.word_idx if f is not None and f.transcript and 0 <= k < len(f.transcript)]
        if not idx or not f.path:
            continue
        ws = [f.transcript[k] for k in idx]
        out.append(SpeakerUnit(f.id, f.path, idx, [w["w"] for w in ws], min(w["s"] for w in ws),
                               max(w["e"] for w in ws), f"{cap.clip_id}|{f.id}"))
    return out


def detect(units: list[SpeakerUnit], mode: str, load: Optional[Callable] = None,
           progress: Optional[Callable] = None, cancel: Optional[threading.Event] = None) -> list[int]:
    """The speaker of each unit (-1: too little voice to tell). Reads each clip's sound once."""
    from .captions import _load_audio_16k
    load = load or _load_audio_16k
    groups: dict[str, list[SpeakerUnit]] = {}
    for u in units:
        groups.setdefault(u.group or u.path, []).append(u)
    for k, us in enumerate(groups.values()):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress:
            progress(k / max(1, len(groups)), f"Listening to the voices ({k + 1}/{len(groups)})")
        start = max(0.0, min(u.s for u in us) - 0.05)
        audio = load(us[0].path, start, max(u.e for u in us) + 0.05 - start)
        for u in us:
            a, b = int((u.s - start) * RATE), int((u.e - start) * RATE)
            u.features = voice_features(audio[max(0, a):max(0, b)])
    known = [i for i, u in enumerate(units) if u.features is not None]
    out = [-1] * len(units)
    if known:
        labels = cluster(np.array([units[i].features for i in known]), mode)
        for i, lab in zip(known, labels):
            out[i] = int(lab)
    if progress:
        progress(1.0, "Listening to the voices")
    return out


def apply_speakers(project: Project, units: list[SpeakerUnit], speakers: list[int]) -> int:
    """Write the detected speakers onto the transcript words ("p"; a unit of -1 loses its mark). A unit
    whose words changed since detection started is skipped. Returns how many speakers were found."""
    for u, sp in zip(units, speakers):
        f = project.transcript_source(u.footage_id)
        if f is None or not f.transcript or any(k >= len(f.transcript) for k in u.idx):
            continue
        if [f.transcript[k]["w"] for k in u.idx] != u.texts:
            continue
        for k in u.idx:
            if sp < 0:
                f.transcript[k].pop("p", None)
            else:
                f.transcript[k]["p"] = sp
    return len({sp for sp in speakers if sp >= 0})
