"""Best-moment selection.

Scores every analysis sample, slides candidate windows over each footage and greedily
picks the best ones under these rules:
  * a footage may be used several times, but its segments never overlap and keep at
    least `min_gap` seconds of source time between them;
  * visually near-identical segments close together in the same footage are rejected (no repeats);
    further apart they are distinct moments of a long take and only lose a little;
  * each footage gets a share of the clips that grows with its length (a long recording gives several
    moments); uses beyond that share are progressively penalised so all footage gets a chance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .analysis import SAMPLE_RATE, WAVE_RATE, Analysis
from .edits import TRANSITIONS
from .models import Footage, TimelineClip


STYLES = {
    "balanced": dict(sharp=.25, color=.15, act=.15, audio=.15, face=.15, expo=.15, shake=.35),
    "action":   dict(sharp=.15, color=.10, act=.35, audio=.20, face=.05, expo=.15, shake=.20),
    "people":   dict(sharp=.15, color=.05, act=.05, audio=.25, face=.35, expo=.15, shake=.35),
    "scenic":   dict(sharp=.30, color=.30, act=.00, audio=.00, face=.05, expo=.20, shake=.50),
    # Not offered as a style: what makes a strong opening hook (movement, sound, faces, still sharp).
    "hook":     dict(sharp=.15, color=.10, act=.30, audio=.20, face=.25, expo=.10, shake=.30),
}
STYLE_LABELS = {
    "balanced": "Balanced",
    "action": "Action / sports (motion + loud moments)",
    "people": "People / talking (faces + speech)",
    "scenic": "Scenic / travel (colour + stable shots)",
}

TRANSITION_MODES = {
    "off": "Off (straight cuts)",
    "scenes": "Between scenes (different footage)",
    "all": "Every cut",
}
TRANSITION_STYLES = {k: v[0] for k, v in TRANSITIONS.items()} | {"mixed": "Mixed (varies)"}
_MIXED = ["dissolve", "slide", "zoom", "dip_black", "wipe"]

REPEAT_WINDOW = 10.0    # s (at least; 2 x min_gap when larger): look-alike segments closer than this in one
                        # footage count as repeats
FAR_REPEAT_PENALTY = 0.08   # look-alike but further apart: a small penalty only
OVER_SHARE_PENALTY = 0.12   # per use beyond the footage's share
AVOID_PENALTY = 0.35        # per earlier edit that used a moment: a new edit prefers moments not seen yet
SILENCE_DB = (-52.0, -38.0)  # dBFS: dead silent at the first, normal room sound from the second
SILENCE_PENALTY = 0.45      # score taken off dead-silent moments (less for quiet ones)
VARIETY = 0.06              # random spread between moments of about the same score (new edits differ)

ORDERS = {
    "hook": "Hook first (best clip, then chronological)",
    "chronological": "Chronological",
    "best_first": "Best moments first",
}


@dataclass
class AutoEditSettings:
    video_format: str = "short"     # short | long | custom
    target_duration: float = 30.0
    clip_min: float = 1.2
    clip_max: float = 3.0
    min_gap: float = 3.0            # min source-time gap between segments of the same footage
    max_uses: int = 0               # per footage, 0 = unlimited
    use_every_footage: bool = True
    order: str = "hook"
    style: str = "balanced"
    similarity: float = 0.93        # cosine similarity above which segments count as repeats
    replace_timeline: bool = True
    transitions: str = "off"        # TRANSITION_MODES key
    transition_style: str = "dissolve"
    transition_dur: float = 0.5
    jl_cuts: bool = False           # J/L split edits where the sound leads or trails a cut
    bookends: bool = False          # find the opening and closing (file names, recording time, speech)

    def to_dict(self) -> dict:
        return asdict(self)


# Video format presets: (settings overrides, project aspect)
FORMAT_PRESETS = {
    "short": dict(
        label="Short-form (Reels / TikTok / Shorts)",
        aspect="9:16 (Vertical)",
        settings=dict(target_duration=30.0, clip_min=1.2, clip_max=3.0, min_gap=3.0, order="hook",
                      transitions="off", transition_style="slide", transition_dur=0.3, jl_cuts=False,
                      bookends=False),
        hint="Vertical 9:16, 15-60 s, fast 1-3 s cuts, strongest moment first as the hook.",
    ),
    "long": dict(
        label="Long-form (YouTube / film)",
        aspect="16:9 (Landscape)",
        settings=dict(target_duration=180.0, clip_min=4.0, clip_max=10.0, min_gap=8.0, order="chronological",
                      transitions="scenes", transition_style="dissolve", transition_dur=0.8, jl_cuts=True,
                      bookends=True),
        hint="Landscape 16:9, several minutes, longer 4-10 s shots in story order.",
    ),
    "custom": dict(label="Custom", aspect=None, settings={}, hint="Set every option yourself."),
}


def suggest_duration(video_format: str, total: float) -> Optional[float]:
    """Default target length from the total length of the footage used.

    Long-form keeps about a quarter of the footage (1-20 min, and never more than the footage can give
    once segments keep their gaps); short-form stays a short social clip."""
    if total <= 0:
        return None
    if video_format == "long":
        d = min(max(total * 0.25, 60.0), 20 * 60.0, total * 0.7)
        return float(max(10.0, round(d / 10.0) * 10.0))
    if video_format == "short":
        return float(min(30.0, max(10.0, round(total * 0.5))))
    return None


def natural_length(a: Analysis, sc: np.ndarray, dur: float, start: float, lo: float, hi: float,
                   limit: Optional[float] = None) -> float:
    """How long the moment starting at `start` lasts: while the footage stays better than its median score,
    ending at the next scene cut, the footage's end or `limit` (source time), at least `lo` and at most `hi`."""
    t = a.times
    k = int(np.clip(np.searchsorted(t, start), 0, len(t) - 1))
    thr = float(np.median(sc)) if sc.size else 0.0
    j = k
    while j < len(sc) and t[j] - start < hi and (t[j] - start < lo or sc[j] >= thr):
        j += 1
    end = float(t[j]) if j < len(t) else dur
    cuts = [c for c in np.asarray(a.cuts, dtype=float) if c > start + 0.05]
    end = min(end, start + hi, dur, min(cuts, default=dur), dur if limit is None else limit)
    return round(max(0.0, end - start), 3)


def _loud(a: Optional[Analysis], t0: float, t1: float) -> bool:
    """Speech / sound present in [t0, t1] (source time), relative to the footage's own level."""
    if a is None or a.wave.size < 2:
        return False
    i0, i1 = int(max(0.0, t0) * WAVE_RATE), int(max(0.0, t1) * WAVE_RATE) + 1
    seg = a.wave[i0:i1]
    if seg.size == 0:
        return False
    return float(seg.mean()) > max(0.12, 1.3 * float(np.median(a.wave)))


def apply_edit_style(clips: list[TimelineClip], footage: dict[str, Footage], analyses: dict[str, Analysis],
                     s: AutoEditSettings, switches: Optional[set[int]] = None) -> None:
    """Transitions and J/L cuts for an automatic edit (in place). switches: indices of the clips that start
    on another camera angle (multicam switching): those cuts count as scene changes."""
    n_style = 0
    switches = switches or set()
    for k in range(1, len(clips)):
        a, b = clips[k - 1], clips[k]
        b.transition, b.audio_split = "", 0.0
        scene = a.footage_id != b.footage_id or abs(b.src_in - a.src_out) > 30.0 or k in switches
        continuous = a.footage_id == b.footage_id and abs(b.src_in - a.src_out) < 0.01   # one recording, cut
        bookend = a.label == "opening" or b.label == "closing"
        if s.transitions == "all" or (s.transitions == "scenes" and (scene or bookend)):
            style = s.transition_style
            if style == "mixed":
                style = _MIXED[n_style % len(_MIXED)]
                n_style += 1
            b.transition = "dissolve" if bookend and style not in ("dissolve", "dip_black") else style
            b.transition_dur = s.transition_dur
            continue
        if not s.jl_cuts or continuous:    # (the sound runs on through a cut in one recording)
            continue
        fa, fb = footage.get(a.footage_id), footage.get(b.footage_id)
        if not (fa and fb and fa.has_audio and fb.has_audio):
            continue
        lead = min(0.8, a.duration / 3, b.duration / 3, b.src_in)                  # J needs room before b
        lag = min(0.6, a.duration / 3, b.duration / 3, fa.duration - a.src_out)    # L needs room after a
        b_talks = _loud(analyses.get(b.footage_id), b.src_in, b.src_in + 0.6)
        a_talks = _loud(analyses.get(a.footage_id), a.src_out - 0.6, a.src_out)
        # Let the next scene's sound lead in when it starts with sound; let a voice finish over the next
        # picture when the outgoing clip ends mid-sound. Alternate when both apply.
        if b_talks and lead >= 0.3 and not (a_talks and k % 2 and lag >= 0.3):
            b.audio_split = -round(lead, 2)
        elif a_talks and lag >= 0.3:
            b.audio_split = round(lag, 2)


def _rank(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, dtype=np.float32)
    r[order] = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    return r


def score_samples(analyses: dict[str, Analysis], footage: dict[str, Footage], style: str) -> dict[str, np.ndarray]:
    """Per-sample quality score in roughly 0..1, ranked globally across all footage."""
    w = STYLES.get(style, STYLES["balanced"])
    ids = list(analyses)
    if not ids:
        return {}
    cat = lambda name: np.concatenate([getattr(analyses[i], name) for i in ids])  # noqa: E731
    sizes = [len(analyses[i].times) for i in ids]
    sharp, color, act, audio, jerk = (_rank(cat(n)) for n in ("sharp", "colorful", "activity", "audio", "jerk"))
    faces = np.clip(cat("faces") * 25.0, 0, 1)
    bright = cat("brightness")
    clipped = cat("clipped")
    motion = cat("motion")
    has_audio = np.concatenate([np.full(s, footage[i].has_audio, bool) for i, s in zip(ids, sizes)])

    act_s = np.where(act < 0.85, act, np.maximum(0.0, 0.85 - (act - 0.85) * 5.0))
    expo = np.clip(1.0 - np.maximum(0.0, np.abs(bright - 0.5) - 0.35) * 6.0, 0, 1) - np.clip(clipped * 2.0, 0, 1)
    shake = np.clip((jerk - 0.7) / 0.3, 0, 1) + np.clip((motion - 25.0) / 25.0, 0, 1)
    audio_s = np.where(has_audio, audio, 0.5)

    total = (w["sharp"] * sharp + w["color"] * color + w["act"] * act_s + w["audio"] * audio_s
             + w["face"] * faces + w["expo"] * expo - w["shake"] * shake)
    total = total / sum(v for k, v in w.items() if k != "shake")

    out, k = {}, 0
    for i, s in zip(ids, sizes):
        sc = total[k:k + s].astype(np.float32).copy()
        k += s
        t = analyses[i].times
        dur = footage[i].duration
        sc[t < 0.6] -= 0.5                 # camera bump at record start
        sc[t > dur - 0.6] -= 0.5           # and at record stop
        for c in analyses[i].cuts:
            sc[np.abs(t - c) < 0.3] -= 1.0
        if footage[i].has_audio:           # dead / silent sound makes a weak moment
            lo, hi = SILENCE_DB
            sc -= SILENCE_PENALTY * np.clip((hi - analyses[i].audio) / (hi - lo), 0.0, 1.0)
        out[i] = sc
    return out


@dataclass
class _Cands:
    fid: str
    score: np.ndarray      # best window score per start sample
    length: np.ndarray     # chosen window length (seconds) per start sample
    embed: np.ndarray      # normalised embedding per start sample (window middle)
    valid: np.ndarray      # bool mask


def _candidates(fid: str, a: Analysis, sc: np.ndarray, dur: float, s: AutoEditSettings) -> _Cands:
    n = len(sc)
    starts = a.times
    csum = np.r_[0.0, np.cumsum(sc)]
    csq = np.r_[0.0, np.cumsum(sc.astype(np.float64) ** 2)]
    lengths = sorted({round(s.clip_min, 2), round((s.clip_min + s.clip_max) / 2, 2), round(s.clip_max, 2)})
    best = np.full(n, -np.inf, dtype=np.float32)
    best_len = np.full(n, lengths[0], dtype=np.float32)
    cut_times = np.asarray(a.cuts, dtype=np.float32)
    span = max(1e-6, s.clip_max - s.clip_min)
    for L in lengths:
        l = max(1, int(round(L * SAMPLE_RATE)))
        if l > n:
            continue
        m = n - l + 1
        mean = (csum[l:l + m] - csum[:m]) / l
        var = np.maximum(0.0, (csq[l:l + m] - csq[:m]) / l - mean ** 2)
        val = mean - 0.25 * np.sqrt(var) + 0.04 * (L - s.clip_min) / span
        ok = starts[:m] + L <= dur - 0.05
        if cut_times.size:
            st = starts[:m, None]
            ok &= ~np.any((cut_times[None, :] > st + 0.05) & (cut_times[None, :] < st + L - 0.05), axis=1)
        val = np.where(ok, val, -np.inf)
        better = val > best[:m]
        best[:m][better] = val[better]
        best_len[:m][better] = L
    # Very short footage: allow one window covering everything usable.
    if not np.isfinite(best).any() and dur >= min(1.0, s.clip_min):
        best[0] = float(np.mean(sc))
        best_len[0] = min(dur, s.clip_max)
    mid = np.clip((np.arange(n) + (best_len * SAMPLE_RATE / 2).astype(int)), 0, n - 1)
    e = a.embed[mid].astype(np.float32)
    e = e - e.mean(axis=1, keepdims=True)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-6
    return _Cands(fid, best, best_len, e, np.isfinite(best))


def select(project_footage: dict[str, Footage], analyses: dict[str, Analysis],
           s: AutoEditSettings, order_ids: Optional[list[str]] = None,
           existing: Optional[list[TimelineClip]] = None, avoid: Optional[list[TimelineClip]] = None,
           seed: Optional[int] = None) -> list[TimelineClip]:
    """Pick clips. `existing` clips are treated as already used (no overlap / repeats). `avoid` clips
    (moments earlier edits used) are only discouraged, so a new edit of the same footage shows other
    moments while there are good ones left. A `seed` varies the pick between moments of similar score."""
    ids = [i for i in (order_ids or list(project_footage)) if i in analyses]
    if not ids:
        return []
    scores = score_samples({i: analyses[i] for i in ids}, project_footage, s.style)
    cands = {i: _candidates(i, analyses[i], scores[i], project_footage[i].duration, s) for i in ids}
    picked: list[tuple[str, float, float, float]] = []   # fid, in, out, score
    picked_embed: dict[str, list[np.ndarray]] = {i: [] for i in ids}
    picked_mid: dict[str, list[float]] = {i: [] for i in ids}
    uses = {i: 0 for i in ids}
    total = 0.0
    bias = {i: np.zeros(len(cands[i].score), dtype=np.float32) for i in ids}
    for ex in avoid or []:
        if ex.footage_id in bias:
            t = analyses[ex.footage_id].times
            bias[ex.footage_id][(t < ex.src_out) & (t + cands[ex.footage_id].length > ex.src_in)] += AVOID_PENALTY
    if seed is not None:
        rng = np.random.default_rng(seed)
        for i in ids:
            bias[i] += rng.uniform(0.0, VARIETY, len(bias[i])).astype(np.float32)
    # Clips each footage should give, in proportion to its length.
    durs = {i: project_footage[i].duration for i in ids}
    n_clips = s.target_duration / max(0.1, (s.clip_min + s.clip_max) / 2)
    share = {i: n_clips * durs[i] / max(1e-6, sum(durs.values())) for i in ids}

    def block(fid: str, a_in: float, a_out: float, emb: np.ndarray):
        c = cands[fid]
        t = analyses[fid].times
        c.valid &= ~((t < a_out + s.min_gap) & (t + c.length > a_in - s.min_gap))
        picked_embed[fid].append(emb)
        picked_mid[fid].append((a_in + a_out) / 2)

    def effective(fid: str) -> np.ndarray:
        c = cands[fid]
        sc = np.where(c.valid, c.score - bias[fid], -np.inf)
        if picked_embed[fid]:
            sims = c.embed @ np.stack(picked_embed[fid]).T                      # (n, picked)
            mid = analyses[fid].times + c.length / 2
            window = max(REPEAT_WINDOW, 2 * s.min_gap)
            near = np.abs(mid[:, None] - np.asarray(picked_mid[fid])[None, :]) < window
            alike = sims > s.similarity
            sc = np.where((alike & near).any(axis=1), -np.inf, sc)
            sc = sc - FAR_REPEAT_PENALTY * (alike & ~near).any(axis=1)
        return sc - OVER_SHARE_PENALTY * max(0.0, uses[fid] + 1 - share[fid])

    def take(fid: str, eff: np.ndarray) -> bool:
        nonlocal total
        k = int(np.argmax(eff))
        if not np.isfinite(eff[k]):
            return False
        c = cands[fid]
        remaining = s.target_duration - total
        length = float(min(c.length[k], remaining))
        if length < min(s.clip_min, 1.0) - 1e-6:
            return False
        a_in = float(analyses[fid].times[k])
        a_out = min(project_footage[fid].duration, a_in + length)
        picked.append((fid, a_in, a_out, float(c.score[k])))
        block(fid, a_in, a_out, c.embed[k])
        uses[fid] += 1
        total += a_out - a_in
        return True

    for ex in existing or []:
        if ex.footage_id in cands:
            t = analyses[ex.footage_id].times
            k = int(np.clip(np.searchsorted(t, (ex.src_in + ex.src_out) / 2), 0, len(t) - 1))
            block(ex.footage_id, ex.src_in, ex.src_out, cands[ex.footage_id].embed[k])

    if s.use_every_footage:
        firsts = []
        for fid in ids:
            eff = effective(fid)
            if np.isfinite(eff).any():
                firsts.append((float(np.max(eff)), fid))
        for _, fid in sorted(firsts, reverse=True):
            if total >= s.target_duration - 0.05:
                break
            take(fid, effective(fid))

    while total < s.target_duration - 0.05:
        best_fid, best_val, best_eff = None, -np.inf, None
        for fid in ids:
            if s.max_uses and uses[fid] >= s.max_uses:
                continue
            eff = effective(fid)
            v = float(np.max(eff)) if eff.size else -np.inf
            if v > best_val:
                best_fid, best_val, best_eff = fid, v, eff
        if best_fid is None or not np.isfinite(best_val):
            break
        if not take(best_fid, best_eff):
            break

    # Ordering
    n_hook, hook_len = hook_plan(s)
    if s.order == "hook" and n_hook > 1:
        # Long-form: a quick teaser of a few short moments, then the edit in story order.
        hs = copy.copy(s)
        hs.order = "chronological"
        hs.target_duration = max(s.clip_min, s.target_duration - n_hook * hook_len)
        body = select(project_footage, analyses, hs, order_ids, existing, avoid, seed)
        return find_hook_montage(project_footage, analyses, s, (existing or []) + body) + body
    pos = {fid: k for k, fid in enumerate(ids)}
    chrono = sorted(picked, key=lambda p: (pos[p[0]], p[1]))
    if s.order == "best_first":
        ordered = sorted(picked, key=lambda p: -p[3])
    elif s.order == "hook" and picked:
        hook = max(picked, key=lambda p: p[3])
        ordered = [hook] + [p for p in chrono if p is not hook]
    else:
        ordered = chrono
    clips = [TimelineClip(footage_id=f, src_in=a, src_out=b, score=sc) for f, a, b, sc in ordered]
    if s.order == "hook" and clips:
        clips[0].label = "hook"
    return clips


def hook_plan(s: AutoEditSettings) -> tuple[int, float]:
    """(clips, seconds each) of the opening hook: one quick clip in short-form; in long-form, where shots
    are long, a teaser of 3 short moments."""
    if s.video_format == "long" or s.clip_max > 4.0:
        return 3, 2.0
    return 1, s.clip_max


def find_hook_montage(project_footage: dict[str, Footage], analyses: dict[str, Analysis], s: AutoEditSettings,
                      avoid: list[TimelineClip]) -> list[TimelineClip]:
    """The long-form hook: hook_plan(s) short strong moments (none of them overlapping `avoid`), each
    labelled "hook", strongest first."""
    n, length = hook_plan(s)
    hs = copy.copy(s)
    hs.style = "hook"
    hs.target_duration = n * length
    hs.clip_min = hs.clip_max = length
    hs.use_every_footage = False
    hs.max_uses = 0
    hs.order = "best_first"
    clips = select(project_footage, analyses, hs, existing=avoid)[:n]
    for c in clips:
        c.label = "hook"
    return clips


def find_hook(project_footage: dict[str, Footage], analyses: dict[str, Analysis], s: AutoEditSettings,
              length: float, avoid: list[TimelineClip]) -> Optional[TimelineClip]:
    """A strong opening moment (movement, sound, faces) of about `length` seconds that overlaps none of
    the `avoid` clips (the timeline and hooks already tried)."""
    hs = copy.copy(s)
    hs.style = "hook"
    hs.target_duration = length
    hs.clip_min = hs.clip_max = length
    hs.use_every_footage = False
    hs.max_uses = 0
    hs.order = "hook"
    clips = select(project_footage, analyses, hs, existing=avoid)
    return clips[0] if clips else None
