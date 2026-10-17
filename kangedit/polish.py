"""Finishing steps for an automatic edit (and Tools for any timeline): cuts that land between words, on the
music's beat, at a pace that changes along the edit; filler words, long pauses and false starts taken out;
neighbouring shots of different sizes; and every clip's sound at an even loudness.

They work on main-track clips at normal speed (a clip with a speed change or ramp is left as it is) and never
move a cut past the footage's ends, into the next words, or onto a moment another clip of the same file
shows.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

import numpy as np

from .models import Footage, TimelineClip

MIN_LEN = 0.5            # seconds: no clip is made shorter than this
PAUSE_SHIFT = 0.6        # a cut moves at most this far to get between words
WORD_PAD_IN = 0.06       # room kept before the first word / after the last one
WORD_PAD_OUT = 0.12
QUIET_DB = 6.0           # without a transcript: a cut moves to a moment this much quieter
FILLERS = {
    "um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "hmm", "hm", "mm", "mmm", "eh", "ehm", "em",
    "emm", "eee", "ee", "anu", "apa namanya", "gitu", "kayak", "like", "you know",
}
SENTENCE_PAUSE = 1.0     # a silence this long ends a sentence even without punctuation
_SENT_END = re.compile(r"[.!?…。！？][\"'”’)\]]*$")
LONG_PAUSE = 0.9         # a silence longer than this inside a clip is shortened to KEEP_PAUSE
KEEP_PAUSE = 0.35
MIN_PIECE = 0.4          # pieces of a cleaned-up clip shorter than this are dropped
SHOT_CLOSE, SHOT_MEDIUM = 0.05, 0.008     # face area (part of the picture) of a close-up / medium shot
PACING = {
    "steady": "Steady (even clip lengths)",
    "build": "Build-up (long shots first, faster and faster)",
    "trailer": "Trailer (quick, then longer, quick again)",
    "calm": "Calm finish (quick start, longer at the end)",
}


def _normal(clip: TimelineClip) -> bool:
    return abs(clip.speed - 1.0) < 1e-6 and not clip.ramp


def _word(w: dict) -> str:
    t = unicodedata.normalize("NFKD", str(w.get("w", ""))).lower()
    return re.sub(r"[^\w\s]", "", t).strip()


def _free(clips: list[TimelineClip], c: TimelineClip, a: float, b: float, gap: float = 0.05,
          avoid: Sequence[TimelineClip] = ()) -> bool:
    """Whether [a, b] of c's footage is clear, by `gap` seconds, of the other clips showing that footage and of
    the `avoid` clips (moments an edit must not show: the rest of the timeline, parts not every camera
    recorded, what a highlight reel's edit left out)."""
    return not any(o is not c and o.footage_id == c.footage_id and a < o.src_out + gap and b > o.src_in - gap
                   for o in list(clips) + list(avoid))


def _word_at(words: list[dict], t: float) -> Optional[int]:
    for i, w in enumerate(words):
        if w["s"] < t < w["e"]:
            return i
        if w["s"] >= t:
            break
    return None


# ----------------------------------------------------------------------------- cuts between words

def snap_to_pauses(clips: list[TimelineClip], footage: dict[str, Footage], analyses: dict,
                   max_shift: float = PAUSE_SHIFT, gap: float = 0.05, avoid: Sequence[TimelineClip] = ()) -> int:
    """Move cuts that fall inside a word to the pause next to it (the whole word kept when that's close,
    else left out), using the transcript's word timings; without a transcript, to a clearly quieter moment
    within max_shift. In place; returns how many cuts moved."""
    moved = 0
    for c in clips:
        f = footage.get(c.footage_id)
        if f is None or f.is_image or not _normal(c):
            continue
        a, b = c.src_in, c.src_out
        words = f.transcript or []
        if words:
            i = _word_at(words, a)
            if i is not None:
                w = words[i]
                prev_end = words[i - 1]["e"] if i > 0 else 0.0
                if a - w["s"] <= max_shift:                         # take the whole word in
                    a = max(prev_end + 0.02, w["s"] - WORD_PAD_IN, 0.0)
                elif w["e"] - a <= max_shift:                       # or start after it
                    a = min(w["e"] + 0.02, b - MIN_LEN)
            j = _word_at(words, b)
            if j is not None:
                w = words[j]
                nxt = words[j + 1]["s"] if j + 1 < len(words) else f.duration
                if w["e"] - b <= max_shift:                         # finish the word
                    b = min(nxt - 0.02, w["e"] + WORD_PAD_OUT, f.duration)
                elif b - w["s"] <= max_shift:                       # or stop before it
                    b = max(w["s"] - 0.02, a + MIN_LEN)
        else:
            an = analyses.get(c.footage_id)
            if an is not None and f.has_audio:
                a = _quiet_near(an, a, max_shift, lo=0.0, hi=b - MIN_LEN)
                b = _quiet_near(an, b, max_shift, lo=a + MIN_LEN, hi=f.duration)
        if (abs(a - c.src_in) > 1e-3 or abs(b - c.src_out) > 1e-3) and b - a >= MIN_LEN and _free(clips, c, a, b, gap, avoid):
            c.src_in, c.src_out = round(a, 3), round(b, 3)
            moved += 1
    return moved


def sentence_spans(words: list[dict], pause: float = SENTENCE_PAUSE) -> list[tuple[int, int]]:
    """(first, last) word index of each sentence of a transcript: up to a word ending in . ! ? (or …) or
    followed by a silence of `pause` seconds."""
    out: list[tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None or _SENT_END.search(str(w.get("w", "")).strip()) or float(nxt["s"]) - float(w["e"]) > pause:
            out.append((start, i))
            start = i + 1
    return out


def sentences(words: list[dict], pause: float = SENTENCE_PAUSE) -> list[tuple[float, float]]:
    """(start, end) time of each sentence of a transcript (see sentence_spans)."""
    return [(float(words[i]["s"]), float(words[j]["e"])) for i, j in sentence_spans(words, pause)]


def snap_to_sentences(clips: list[TimelineClip], footage: dict[str, Footage], max_len: float,
                      gap: float = 0.05, avoid: Sequence[TimelineClip] = ()) -> int:
    """Make every clip with speech start and end on whole sentences (the transcript's), so each holds at least
    one complete sentence: a sentence cut into is taken in whole when the clip stays within `max_len`
    seconds, else left out when the clip has another whole one. Clips without speech, hooks, and clips that
    would land on a moment another clip shows stay as they are. In place; returns how many changed."""
    changed = 0
    for c in clips:
        f = footage.get(c.footage_id)
        if f is None or f.is_image or not f.transcript or not _normal(c) or c.label == "hook":
            continue
        a, b = c.src_in, c.src_out
        inside = [(s0, s1) for s0, s1 in sentences(f.transcript) if s0 < b and s1 > a]
        if not inside:
            continue
        cut_in, cut_out = inside[0][0] < a - 1e-3, inside[-1][1] > b + 1e-3
        if not cut_in and not cut_out:
            continue
        grow_a = max(0.0, inside[0][0] - WORD_PAD_IN) if cut_in else a
        grow_b = min(f.duration, inside[-1][1] + WORD_PAD_OUT) if cut_out else b
        drop_a = (inside[1][0] - WORD_PAD_IN if len(inside) > 1 else None) if cut_in else a
        drop_b = (inside[-2][1] + WORD_PAD_OUT if len(inside) > 1 else None) if cut_out else b
        for x, y in ((grow_a, grow_b), (grow_a, drop_b), (drop_a, grow_b), (drop_a, drop_b)):
            if x is None or y is None or y - x < max(MIN_LEN, 1.0) or y - x > max_len:
                continue
            if not any(s0 >= x - 1e-3 and s1 <= y + 1e-3 for s0, s1 in inside):
                continue                                              # not one whole sentence in it
            if _free(clips, c, x, y, gap, avoid):
                c.src_in, c.src_out = round(x, 3), round(y, 3)
                changed += 1
                break
    return changed


def drop_repeats(clips: list[TimelineClip], gap: float, avoid: Sequence[TimelineClip] = ()) -> tuple[list[TimelineClip], int]:
    """The clips without those that show a moment (closer than `gap` seconds in the source) that an earlier
    clip, a hook / opening / closing / scene opener or an `avoid` clip already shows. Pieces of one take cut
    straight together (one right after the other, going on in the source) are not repeats. Returns the clips
    kept and how many were dropped."""
    def clash(c: TimelineClip, o: TimelineClip) -> bool:
        if o.footage_id != c.footage_id or o is c:
            return False
        if abs(c.src_in - o.src_out) < 0.02 or abs(o.src_in - c.src_out) < 0.02:
            return False                                               # contiguous pieces
        return c.src_in < o.src_out + gap - 1e-3 and o.src_in < c.src_out + gap - 1e-3

    special = [c for c in clips if c.label in ("hook", "opening", "closing", "scene")]
    kept: list[TimelineClip] = []
    take: list[TimelineClip] = []          # the pieces of the take the last kept clip belongs to
    for c in clips:
        prev = kept[-1] if kept else None
        jump = prev is not None and prev.footage_id == c.footage_id and c.src_in >= prev.src_out - 0.02
        others = list(avoid) + [o for o in special if o is not c] + kept
        if any(o is c for o in special) or not any(clash(c, o) for o in others
                                                   if not (jump and any(o is t for t in take))):
            take = take + [c] if jump else [c]
            kept.append(c)
    return kept, len(clips) - len(kept)


def _quiet_near(an, t: float, shift: float, lo: float, hi: float) -> float:
    times = np.asarray(an.times)
    audio = np.asarray(an.audio, dtype=np.float64)
    if times.size == 0:
        return t
    k = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
    here = audio[k]
    near = np.flatnonzero((np.abs(times - t) <= shift) & (times >= lo) & (times <= hi))
    if near.size == 0:
        return t
    q = near[np.argmin(audio[near])]
    return float(times[q]) if audio[q] <= here - QUIET_DB and here > -45.0 else t


# ----------------------------------------------------------------------------- fillers, pauses, false starts

def _cut_ranges(words: list[dict], a: float, b: float, max_pause: float) -> list[tuple[float, float]]:
    """Parts of [a, b] to take out: filler words, the middle of long pauses, and the first try of a false
    start (words said again right away: "I went, I went to")."""
    inside = [w for w in words if w["s"] >= a - 0.01 and w["e"] <= b + 0.01]
    toks = [_word(w) for w in inside]
    out: list[tuple[float, float]] = []
    k = 0
    while k < len(inside):
        if k + 1 < len(toks) and " ".join(toks[k:k + 2]) in FILLERS:
            out.append((inside[k]["s"], inside[k + 1]["e"]))      # a two-word filler ("you know")
            k += 2
            continue
        if toks[k] in FILLERS or not toks[k]:
            out.append((inside[k]["s"], inside[k]["e"]))
        else:
            for n in (3, 2, 1):                       # false start: n words straight away said again
                if k + 2 * n <= len(toks) and toks[k:k + n] == toks[k + n:k + 2 * n] and all(toks[k:k + n]):
                    out.append((inside[k]["s"], inside[k + n]["s"]))
                    k += n - 1
                    break
        k += 1
    for w0, w1 in zip(inside, inside[1:]):
        if w1["s"] - w0["e"] > max_pause:
            half = KEEP_PAUSE / 2
            out.append((w0["e"] + half, w1["s"] - half))
    if inside and inside[0]["s"] - a > max_pause:         # dead air before the first / after the last word
        out.append((a, inside[0]["s"] - KEEP_PAUSE / 2))
    if inside and b - inside[-1]["e"] > max_pause:
        out.append((inside[-1]["e"] + KEEP_PAUSE / 2, b))
    out.sort()
    merged: list[list[float]] = []
    for s0, s1 in out:
        if s1 - s0 <= 0.02:
            continue
        if merged and s0 <= merged[-1][1] + 0.12:
            merged[-1][1] = max(merged[-1][1], s1)
        else:
            merged.append([s0, s1])
    return [(s0, s1) for s0, s1 in merged]


def remove_fillers(clips: list[TimelineClip], footage: dict[str, Footage],
                   max_pause: float = LONG_PAUSE) -> tuple[list[TimelineClip], float]:
    """The clips with filler words, long pauses and false starts taken out (each clip that had some becomes
    pieces cut straight together), using the transcripts; clips without one stay as they are. Returns the
    new list and the seconds removed."""
    out: list[TimelineClip] = []
    removed = 0.0
    for c in clips:
        f = footage.get(c.footage_id)
        if f is None or not f.transcript or not _normal(c) or c.label in ("hook",):
            out.append(c)
            continue
        cuts = _cut_ranges(f.transcript, c.src_in, c.src_out, max_pause)
        if not cuts:
            out.append(c)
            continue
        keep, t = [], c.src_in
        for s0, s1 in cuts:
            if s0 > t:
                keep.append((t, s0))
            t = max(t, s1)
        if c.src_out > t:
            keep.append((t, c.src_out))
        keep = [(s0, s1) for s0, s1 in keep if s1 - s0 >= MIN_PIECE]
        if not keep:
            out.append(c)
            continue
        removed += (c.src_out - c.src_in) - sum(s1 - s0 for s0, s1 in keep)
        for n, (s0, s1) in enumerate(keep):
            p = TimelineClip.from_dict(c.to_dict())
            p.src_in, p.src_out = round(s0, 3), round(s1, 3)
            if n:
                p.id = TimelineClip(c.footage_id, s0, s1).id
                p.transition, p.audio_split = "", 0.0
                p.label = "" if c.label in ("opening", "closing", "scene") else c.label
            out.append(p)
    return out, removed


# ----------------------------------------------------------------------------- pacing

def pace_factors(n: int, pacing: str) -> np.ndarray:
    """Relative clip lengths (mean 1) along an edit of n clips for a pacing preset."""
    if n <= 1 or pacing not in PACING or pacing == "steady":
        return np.ones(max(n, 0))
    x = np.linspace(0.0, 1.0, n)
    curve = {"build": 1.0 - x, "trailer": np.sin(np.pi * x), "calm": x}[pacing]
    f = 0.55 + 0.9 * curve                       # 0.55x .. 1.45x
    return f / f.mean()


def apply_pacing(clips: list[TimelineClip], footage: dict[str, Footage], pacing: str, clip_min: float,
                 clip_max: float, gap: float = 0.05, avoid: Sequence[TimelineClip] = ()) -> int:
    """Lengthen or shorten the body clips (not hooks, the opening / closing or scene openers) along the edit
    for the pacing, keeping the total and each clip within clip_min..clip_max, around its middle. In place;
    returns how many changed."""
    body = [c for c in clips if c.label not in ("hook", "opening", "closing", "scene") and _normal(c)
            and c.footage_id in footage and not footage[c.footage_id].is_image]
    if len(body) < 3 or pacing == "steady":
        return 0
    total = sum(c.src_out - c.src_in for c in body)
    want = pace_factors(len(body), pacing) * total / len(body)
    want = np.clip(want, clip_min, max(clip_min, clip_max))
    changed = 0
    for c, L in zip(body, want):
        dur = footage[c.footage_id].duration
        mid = (c.src_in + c.src_out) / 2
        a = min(max(0.0, mid - L / 2), max(0.0, dur - L))
        b = min(dur, a + L)
        if abs((b - a) - (c.src_out - c.src_in)) > 0.05 and _free(clips, c, a, b, gap, avoid):
            c.src_in, c.src_out = round(a, 3), round(b, 3)
            changed += 1
    return changed


# ----------------------------------------------------------------------------- shot sizes

def shot_size(an, a: float, b: float) -> str:
    """"close", "medium" or "wide" from the face area over [a, b] of the footage (its analysis)."""
    if an is None:
        return "wide"
    t = np.asarray(an.times)
    m = (t >= a) & (t <= b)
    face = float(np.median(np.asarray(an.faces)[m])) if m.any() else 0.0
    return "close" if face >= SHOT_CLOSE else "medium" if face >= SHOT_MEDIUM else "wide"


def vary_shots(clips: list[TimelineClip], footage: dict[str, Footage], analyses: dict,
               scores: dict[str, np.ndarray], min_gap: float = 0.0, reach: float = 45.0,
               avoid: Sequence[TimelineClip] = ()) -> int:
    """Where three or more shots of the same size follow each other, swap the middle ones for a nearby moment
    of the same file (story order kept) in another size, when one scores nearly as well. In place; returns
    how many were swapped."""
    swapped = 0
    sizes = [shot_size(analyses.get(c.footage_id), c.src_in, c.src_out) for c in clips]
    for k in range(1, len(clips) - 1):
        c = clips[k]
        if not (sizes[k - 1] == sizes[k] == sizes[k + 1]) or c.label or not _normal(c):
            continue
        an, sc, f = analyses.get(c.footage_id), scores.get(c.footage_id), footage.get(c.footage_id)
        if an is None or sc is None or f is None or f.is_image:
            continue
        t = np.asarray(an.times)
        L = c.src_out - c.src_in
        n = max(1, int(round(L * len(t) / max(f.duration, 1e-6))))
        mean = np.convolve(sc, np.ones(n) / n, mode="valid")
        here = float(np.interp(c.src_in, t[:len(mean)], mean)) if len(mean) else -np.inf
        # Between the neighbouring clips of the same file (the story order), or within reach.
        prev, nxt = clips[k - 1], clips[k + 1]
        lo = prev.src_out + min_gap if prev.footage_id == c.footage_id and prev.src_out <= c.src_in else 0.0
        hi = nxt.src_in - min_gap if nxt.footage_id == c.footage_id and nxt.src_in >= c.src_out else f.duration
        best = None
        for i in np.argsort(-mean):
            a = float(t[i])
            if abs(a - c.src_in) > reach or a + L > f.duration - 0.1:
                continue
            if mean[i] < here - 0.15:
                break
            if not (lo <= a and a + L <= hi):          # keeps the story order with the neighbours
                continue
            if shot_size(an, a, a + L) == sizes[k] or not _free(clips, c, a, a + L, min_gap, avoid):
                continue
            best = a
            break
        if best is not None:
            c.src_in, c.src_out = round(best, 3), round(best + L, 3)
            sizes[k] = shot_size(an, best, best + L)
            swapped += 1
    return swapped


# ----------------------------------------------------------------------------- loudness

def even_loudness(clips: list[TimelineClip], footage: dict[str, Footage], analyses: dict,
                  max_gain_db: float = 9.0) -> int:
    """Set each clip's volume so its sound (the louder part of it: speech, action) matches the edit's typical
    level: quiet clips up, loud ones down, at most max_gain_db either way. In place; returns how many
    changed."""
    levels = {}
    for c in clips:
        an, f = analyses.get(c.footage_id), footage.get(c.footage_id)
        if an is None or f is None or not f.has_audio:
            continue
        t = np.asarray(an.times)
        m = (t >= c.src_in) & (t <= c.src_out)
        if m.sum() < 2:
            continue
        lv = float(np.percentile(np.asarray(an.audio)[m], 80))
        if lv > -50.0:                            # silence has no level to match
            levels[c.id] = lv
    if len(levels) < 2:
        return 0
    target = float(np.median(list(levels.values())))
    changed = 0
    for c in clips:
        if c.id not in levels:
            continue
        gain = float(np.clip(target - levels[c.id], -max_gain_db, max_gain_db))
        vol = round(float(np.clip(10 ** (gain / 20.0), 0.3, 2.8)), 3)
        if abs(vol - c.volume) > 0.02:
            c.volume = vol
            changed += 1
    return changed


# ----------------------------------------------------------------------------- beats

ENV_RATE = 100           # onset envelope frames per second


def onset_envelope(audio: np.ndarray, rate: int) -> np.ndarray:
    """Onset strength at ENV_RATE: the rise of the log energy in short frames."""
    hop = max(1, rate // ENV_RATE)
    n = len(audio) // hop
    if n < 4:
        return np.zeros(0)
    frames = audio[:n * hop].reshape(n, hop).astype(np.float64)
    energy = np.log1p(1000.0 * np.mean(frames * frames, axis=1))
    rise = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    rise -= np.convolve(rise, np.ones(15) / 15, mode="same")              # local mean off
    return np.maximum(rise, 0.0)


def detect_beats(audio: np.ndarray, rate: int, lo_bpm: float = 70.0, hi_bpm: float = 180.0) -> np.ndarray:
    """Beat times (seconds from the start of `audio`): the tempo with the strongest periodicity of the onset
    envelope, at the phase whose beats hit the most onsets. Empty when there is no clear beat."""
    env = onset_envelope(audio, rate)
    if len(env) < ENV_RATE * 4 or env.max() <= 0:
        return np.zeros(0)
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lags = np.arange(len(ac))
    ok = (lags >= ENV_RATE * 60 / hi_bpm) & (lags <= ENV_RATE * 60 / lo_bpm)
    if not ok.any() or ac[0] <= 0:
        return np.zeros(0)
    lag = int(lags[ok][np.argmax(ac[ok])])
    if ac[lag] / ac[0] < 0.05:                       # no periodicity worth calling a beat
        return np.zeros(0)
    phases = [env[p::lag].sum() for p in range(lag)]
    p = int(np.argmax(phases))
    # Follow the beat as the tempo drifts a little: each next beat at the strongest onset near where it's due.
    beats, t = [], p
    w = max(1, lag // 8)
    while t < len(env):
        lo_i, hi_i = max(0, t - w), min(len(env), t + w + 1)
        t = lo_i + int(np.argmax(env[lo_i:hi_i])) if env[lo_i:hi_i].max() > 0 else t
        beats.append(t)
        t += lag
    return np.asarray(beats, dtype=np.float64) / ENV_RATE


def snap_to_beats(clips: list[TimelineClip], footage: dict[str, Footage], beats: np.ndarray,
                  start: float = 0.0, max_shift: float = 0.5, gap: float = 0.05,
                  avoid: Sequence[TimelineClip] = ()) -> int:
    """Move each cut (the end of a clip, so the edit's later clips shift with it) to the nearest beat (timeline
    times) within max_shift, by lengthening or shortening the clip before it; a beat that would cut into a word
    or onto a moment another clip shows is passed over for the next nearest. In place; returns how many cuts
    landed on a beat."""
    if len(beats) < 2:
        return 0
    period = float(np.median(np.diff(beats)))
    reach = min(max_shift, 0.5 * period)
    t = start
    hit = 0
    for k, c in enumerate(clips[:-1]):
        f = footage.get(c.footage_id)
        dur = c.duration
        cut = t + dur
        if f is None or f.is_image or not _normal(c):
            t = cut
            continue
        words = f.transcript or []
        near = beats[np.abs(beats - cut) <= reach]
        for bt in sorted(near, key=lambda x: abs(x - cut)):
            d = float(bt - cut)
            out = c.src_out + d
            if out - c.src_in < MIN_LEN or out > f.duration - 0.02:
                continue
            if words and _word_at(words, out) is not None:
                continue
            if d < 0 and words and any(w["s"] < c.src_out and w["e"] > out for w in words):
                continue                                               # would cut off the last words
            if d > 0 and not _free(clips, c, c.src_in, out, gap, avoid):
                continue
            c.src_out = round(out, 3)
            hit += 1
            break
        t += c.duration
    return hit


def music_track(sounds: list):
    """The timeline's music: the longest sound that ducks under the clips (as added music does), else the
    longest sound over 20 s; None when there is none."""
    ducked = [x for x in sounds if x.duck and x.duration > 5.0]
    pool = ducked or [x for x in sounds if x.duration > 20.0]
    return max(pool, key=lambda x: x.duration) if pool else None


def timeline_beats(sound, file_beats: np.ndarray) -> np.ndarray:
    """A sound clip's beats (`file_beats`: seconds in its file) as timeline times, repeats of a looping one
    included."""
    if len(file_beats) == 0:
        return np.zeros(0)
    if not sound.looping:
        b = file_beats[(file_beats >= sound.src_in) & (file_beats <= sound.src_out)]
        return sound.start + (b - sound.src_in)
    one = file_beats[(file_beats >= sound.src_in) & (file_beats < sound.src_in + sound.loop_step())] - sound.src_in
    out = [sound.start + r * sound.loop_step() + one for r in range(sound.repeats())]
    t = np.concatenate(out) if out else np.zeros(0)
    return t[t <= sound.end]


def finish(clips: list[TimelineClip], footage: dict[str, Footage], analyses: dict, s,
           beats: Optional[np.ndarray] = None, start: float = 0.0,
           avoid: Sequence[TimelineClip] = ()) -> tuple[list[TimelineClip], list[str]]:
    """The finishing steps an auto edit's settings `s` (AutoEditSettings) ask for, in order: pacing, varied
    shot sizes, whole sentences (a long-form edit, s.video_format "long"), cuts between words, fillers / pauses
    / false starts out, cuts on the beat (`beats`: timeline times; the clips start at `start`), even loudness.
    No step moves a clip onto the `avoid` clips or nearer than the edit's gap (s.min_gap) to another moment of
    the same footage. Returns the clips (filler removal makes new ones) and notes for the status bar."""
    gap = max(0.05, float(getattr(s, "min_gap", 0.0)))
    from .autoedit import score_samples
    notes: list[str] = []
    if getattr(s, "pacing", "steady") != "steady":
        apply_pacing(clips, footage, s.pacing, s.clip_min, s.clip_max, gap, avoid)
    if getattr(s, "shot_variety", False):
        used = {c.footage_id for c in clips if c.footage_id in analyses}
        scores = score_samples({i: analyses[i] for i in used}, footage, s.style)
        n = vary_shots(clips, footage, analyses, scores, gap, avoid=avoid)
        if n:
            notes.append(f"{n} shot(s) swapped for another size")
    if getattr(s, "whole_sentences", False) and getattr(s, "video_format", "") == "long":
        n = snap_to_sentences(clips, footage, max(2 * s.clip_max, s.clip_max + 10.0), gap, avoid)
        if n:
            notes.append(f"{n} clip(s) set on whole sentences")
    if getattr(s, "cut_on_pauses", False):
        snap_to_pauses(clips, footage, analyses, gap=gap, avoid=avoid)
    if getattr(s, "remove_fillers", False):
        clips, gone = remove_fillers(clips, footage)
        if gone > 0.05:
            notes.append(f"{gone:.1f} s of fillers and pauses taken out")
    if beats is not None and len(beats) and getattr(s, "beat_sync", False):
        n = snap_to_beats(clips, footage, beats, start, gap=gap, avoid=avoid)
        if n:
            notes.append(f"{n} cut(s) on the beat")
    if getattr(s, "even_loudness", False):
        even_loudness(clips, footage, analyses)
    return clips, notes
