"""When footage was recorded, scenes (time groups of footage), and scenic openers for them.

The recording time of a file is the oldest of the dates it carries: the camera's creation time tag, the file's
creation date, its modification date and a date in its file name. Copying a file renews its creation date but
keeps its modification date, and a camera with its clock unset writes no tag or a wrong one, so the oldest
believable date is the closest to when it was really shot. Bins by date, story order (and so where the opening and closing are looked
for) and scenes all use it.

A scene is a time group: files recorded close together. A new scene starts where the next file begins more
than SCENE_GAP after the previous one ended (a new day), or, when both carry a location (phones and many cameras
record one), more than SCENE_DISTANCE from where the scene's last located recording was made (a new place). In a long-form edit each scene after the
first opens with a scenic / travel shot from that scene (a wide establishing shot: no faces, no speech, sharp,
colourful, steady or gently moving), and "transitions between scenes" means between these time groups.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime
from typing import Optional

import numpy as np

from . import smart
from .models import Footage, TimelineClip

SCENE_GAP = 30 * 60.0          # seconds between two recordings that start a new scene
SCENE_DISTANCE = 1000.0        # metres between two recordings that start a new scene
_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")
LOCATION_TAGS = ("com.apple.quicktime.location.iso6709", "location", "location-eng", "com.android.location")
EARLIEST = datetime(1995, 1, 1).timestamp()   # dates before this are a camera clock that was never set
PREFER_BONUS = 0.12            # scenic score bonus for footage the user put in the B-roll bin
SAMPLE_RATE = 4.0              # analysis samples per second (analysis.py)


def _tag_time(f: Footage) -> Optional[float]:
    if not f.creation_time:
        return None
    try:
        return datetime.fromisoformat(f.creation_time.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _file_times(path: str) -> list[float]:
    try:
        st = os.stat(path)
    except OSError:
        return []
    out = [st.st_mtime]
    birth = getattr(st, "st_birthtime", None)       # macOS, BSD, and Windows (Python 3.12+)
    if birth is None and os.name == "nt":
        birth = st.st_ctime                          # Windows: the creation date
    if birth:
        out.append(birth)
    return out


# A date in a file name, with the time when there is one: VID_20260501_091005, PXL_20260501_091005123,
# DJI_20260501091005_0001, 2026-05-01 09.10.05, IMG-20260501-WA0001, GX010001_2026_05_01...
_NAME_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])"
                        r"(?:(?: at |[ _T.-])?([01]\d|2[0-3])[-_.:h]?([0-5]\d)(?:[-_.:m]?([0-5]\d))?\d{0,3})?(?!\d)")


def name_time(name: str) -> Optional[tuple[float, bool]]:
    """(timestamp, has a time of day) of the date in a file name (local time), or None when it has none (or
    one that can't be a recording date)."""
    stem = os.path.splitext(os.path.basename(name))[0]
    for m in _NAME_DATE.finditer(stem):
        y, mo, d, h, mi, sec = (int(g) if g else None for g in m.groups())
        try:
            dt = datetime(y, mo, d, h if h is not None else 12, mi or 0, sec or 0)
        except ValueError:
            continue
        t = dt.timestamp()
        if EARLIEST <= t <= datetime.now().timestamp() + 86400:
            return t, h is not None
    return None


def recorded_time(f: Footage) -> Optional[float]:
    """When footage was recorded (a timestamp): the oldest believable of its creation tag, its file's creation
    date, its modification date and a date in its file name (phones and cameras often name files by when they
    were recorded); None when it has none of them. A name with a date but no time only counts when it's an
    earlier day than the others."""
    times = [t for t in [_tag_time(f)] + _file_times(f.path) if t is not None and t >= EARLIEST]
    named = name_time(f.path)
    if named is not None:
        t, timed = named
        day = datetime.fromtimestamp(t).date()
        if timed or not any(datetime.fromtimestamp(x).date() <= day for x in times):
            times.append(t)
    return min(times) if times else None


def parse_location(tags: dict) -> list:
    """[latitude, longitude] from a file's metadata tags (ISO 6709, e.g. "+37.3317-122.0302+012.000/", as phones,
    action cameras and drones write it), or [] when it has none."""
    for k in LOCATION_TAGS:
        m = _ISO6709.match(str(tags.get(k, "")).split("|")[0].strip())
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat, lon) != (0.0, 0.0):
                return [lat, lon]
    return []


def read_locations(footage: dict[str, Footage]) -> None:
    """Fill in the location of footage imported before locations were read (gps None), from its file."""
    from .ffmpeg import probe
    for f in footage.values():
        if f.gps is None and not f.is_image:
            try:
                f.gps = parse_location(probe(f.path).tags)
            except Exception:  # noqa: BLE001  (missing file, no FFmpeg: no location)
                f.gps = []


def distance(a: list, b: list) -> float:
    """Metres between two [latitude, longitude] (haversine)."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


def time_groups(footage: dict[str, Footage], order: list[str], gap: float = SCENE_GAP,
                far: float = SCENE_DISTANCE) -> dict[str, int]:
    """Scene number of each file in `order` (story order): a new scene starts where a file begins more than
    `gap` seconds after the previous one ended, or was recorded more than `far` metres from the scene's last
    located file. A file without a recording time stays in the scene before it."""
    groups: dict[str, int] = {}
    g, end, where = 0, None, None
    for fid in order:
        f = footage.get(fid)
        if f is None:
            continue
        t = recorded_time(f)
        loc = f.gps or None
        if t is not None:
            if end is not None and t - end > gap:
                g, where = g + 1, None
            elif loc is not None and where is not None and distance(loc, where) > far:
                g += 1
            end = max(end or t, t + f.duration)
        if loc is not None:
            where = loc
        groups[fid] = g
    return groups


def join_scenes(groups: dict[str, int], footage: dict[str, Footage], extra: dict[str, Footage]) -> dict[str, int]:
    """Scenes (time_groups of the edit's footage) with `extra` files (the B-roll bin) added to them: each joins
    the scene of the edit's file recorded closest in time (within SCENE_GAP), without starting or splitting a
    scene. Extra files without a recording time, or far from every scene, are left out."""
    spans = []
    for fid, g in groups.items():
        t = recorded_time(footage[fid]) if fid in footage else None
        if t is not None:
            spans.append((t, t + footage[fid].duration, g))
    out = dict(groups)
    for fid, f in extra.items():
        t = recorded_time(f)
        if t is None or not spans:
            continue
        a, b = t, t + f.duration
        dist, g = min((max(0.0, s0 - b, a - s1), g) for s0, s1, g in spans)
        if dist <= SCENE_GAP:
            out[fid] = g
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    """0..1 rank of each value (equal values share their rank)."""
    if x.size == 0:
        return x.astype(np.float32)
    vals, inv = np.unique(x, return_inverse=True)
    if len(vals) == 1:
        return np.full(x.size, 0.5, np.float32)
    return (np.arange(len(vals), dtype=np.float32) / (len(vals) - 1))[inv]


def scenic_scores(analyses: dict, footage: dict[str, Footage]) -> dict[str, np.ndarray]:
    """How much each analysis sample looks like a scenic / travel shot, roughly 0..1, ranked across the given
    footage: nobody's face filling it, no one speaking, sharp, colourful, well exposed, and either steady or
    moving smoothly (a pan, a drive, a walk), never shaky."""
    ids = [i for i in analyses if i in footage]
    if not ids:
        return {}
    cat = lambda name: np.concatenate([np.asarray(getattr(analyses[i], name), np.float32) for i in ids])  # noqa: E731
    sizes = [len(analyses[i].times) for i in ids]
    sharp, color, jerk = _rank(cat("sharp")), _rank(cat("colorful")), _rank(cat("jerk"))
    no_face = 1.0 - np.clip(cat("faces") * 40.0, 0.0, 1.0)
    quiet = np.clip((-22.0 - cat("audio")) / 16.0, 0.0, 1.0)          # -38 dBFS and below: no one talking
    # With the smarter analysis, "no one talking" is heard as such: loud waves, wind or traffic are a view.
    speech = [smart.at_samples(getattr(analyses[i], "sound", None), "speech", analyses[i].times) for i in ids]
    if any(x is not None for x in speech):
        heard = np.concatenate([np.clip(1.0 - 1.6 * x, 0.0, 1.0) if x is not None else np.full(n, np.nan, np.float32)
                                for x, n in zip(speech, sizes)])
        quiet = np.where(np.isnan(heard), quiet, heard)
    has_audio = np.concatenate([np.full(n, footage[i].has_audio, bool) for i, n in zip(ids, sizes)])
    quiet = np.where(has_audio, quiet, 0.7)
    bright, clipped, motion = cat("brightness"), cat("clipped"), cat("motion")
    expo = np.clip(1.0 - np.maximum(0.0, np.abs(bright - 0.5) - 0.3) * 5.0, 0, 1) - np.clip(clipped * 2.0, 0, 1)
    glide = np.clip(motion / 6.0, 0.0, 1.0) * (1.0 - jerk)             # a smooth pan / drive
    shake = np.clip((jerk - 0.75) / 0.25, 0.0, 1.0) + np.clip((motion - 30.0) / 30.0, 0.0, 1.0)
    total = (0.30 * no_face + 0.20 * sharp + 0.20 * color + 0.15 * expo + 0.10 * quiet + 0.10 * glide
             - 0.40 * shake) / 1.05
    total = total * (0.4 + 0.6 * no_face) * (0.6 + 0.4 * quiet)     # a face or a voice makes it no view
    out, k = {}, 0
    for i, n in zip(ids, sizes):
        sc = total[k:k + n].copy()
        k += n
        t = np.asarray(analyses[i].times)
        sc[(t < 0.6) | (t > footage[i].duration - 0.6)] -= 0.5           # record start / stop bumps
        for c in np.asarray(analyses[i].cuts):
            sc[np.abs(t - c) < 0.3] -= 1.0
        out[i] = sc
    return out


def _overlaps(fid: str, a: float, b: float, used: list[TimelineClip], gap: float) -> bool:
    return any(c.footage_id == fid and a < c.src_out + gap and b > c.src_in - gap for c in used)


def find_opener(fids: list[str], analyses: dict, footage: dict[str, Footage], length: float,
                used: list[TimelineClip], gap: float = 0.0, scores: Optional[dict] = None,
                prefer: Optional[set] = None) -> Optional[TimelineClip]:
    """The best scenic shot `length` seconds long in these files (one scene's), clear of the `used` clips by
    `gap` seconds, a little earlier in the scene preferred (arriving somewhere), and files in `prefer` (the
    B-roll bin) first. None when there is none worth showing (too short footage, or no moment scenic enough)."""
    scores = scores if scores is not None else scenic_scores({i: analyses[i] for i in fids if i in analyses},
                                                             footage)
    best: Optional[tuple[float, str, float]] = None
    for rank, fid in enumerate(fids):
        if fid not in analyses or fid not in scores:
            continue
        f, sc = footage[fid], scores[fid]
        t = np.asarray(analyses[fid].times)
        L = min(length, f.duration - 1.2)
        n = int(round(L * SAMPLE_RATE))
        if n < 2 or len(sc) < n:
            continue
        csum = np.r_[0.0, np.cumsum(sc)]
        mean = (csum[n:] - csum[:-n]) / n                                  # window starting at each sample
        starts = t[:len(mean)]
        mean = mean - 0.02 * rank                                          # earlier files of the scene first
        if prefer and fid in prefer:
            mean = mean + PREFER_BONUS
        for k in np.argsort(-mean)[:50]:
            a = float(starts[k])
            b = a + L
            if b > f.duration - 0.3 or _overlaps(fid, a, b, used, gap):
                continue
            if best is None or mean[k] > best[0]:
                best = (float(mean[k]), fid, a)
            break
    if best is None or best[0] < 0.35:
        return None
    _, fid, a = best
    L = min(length, footage[fid].duration - 1.2)
    return TimelineClip(fid, round(a, 3), round(a + L, 3), label="scene")


def add_scene_openers(clips: list[TimelineClip], groups: dict[str, int], analyses: dict,
                      footage: dict[str, Footage], length: float, gap: float = 0.0,
                      used: Optional[list[TimelineClip]] = None, prefer: Optional[set] = None) -> int:
    """Open every scene of an edit after the first with a scenic shot from that scene (in place): before the
    first clip of each new time group, past any hook; footage in `prefer` (the B-roll bin) first. Returns how
    many were added."""
    used = list(used or []) + list(clips)
    start = next((k for k, c in enumerate(clips) if c.label != "hook"), len(clips))
    added, seen = 0, set()
    prev = None
    k = start
    while k < len(clips):
        g = groups.get(clips[k].footage_id)
        if g is not None and prev is not None and g != prev and g not in seen and clips[k].label != "closing":
            fids = [f for f, gg in groups.items() if gg == g]
            opener = find_opener(fids, analyses, footage, length, used, gap, prefer=prefer)
            if opener is not None:
                clips.insert(k, opener)
                used.append(opener)
                added += 1
                k += 1
        if g is not None:
            if prev is not None and g != prev:
                seen.add(prev)
            prev = g
        k += 1
    return added
