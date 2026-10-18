"""YouTube chapters for the edit: where each scene (a recording day or place, see scenes.py) starts on the
timeline, named by its day and what is said first in it. YouTube shows chapters from the timestamps in a video's
description when the first is at 0:00, there are at least three, and each is at least 10 s long; with fewer
scenes the edit is split into parts at its clips."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from .models import Project

MIN_CHAPTER = 10.0       # seconds: YouTube's shortest chapter
MIN_CHAPTERS = 3
TITLE_WORDS = 7          # a title from speech: its first words


@dataclass
class Chapter:
    start: float         # timeline time
    title: str


def stamp(t: float) -> str:
    """A timestamp as YouTube reads it: m:ss, or h:mm:ss past an hour."""
    s = int(max(0.0, t))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _first_words(project: Project, i0: int, i1: int) -> str:
    """The first words said in clips i0..i1-1 (a short title), or ""."""
    from .polish import clip_words
    for c in project.clips[i0:i1]:
        said = clip_words(c, project.footage.get(c.footage_id))
        if said:
            words = re.split(r"(?<=[.!?…])\s", said)[0].split()
            title = " ".join(words[:TITLE_WORDS]).strip(" ,;:-")
            return title + ("…" if len(words) > TITLE_WORDS else "")
    return ""


def _day(project: Project, i0: int, i1: int) -> Optional[str]:
    from .scenes import recorded_time
    for c in project.clips[i0:i1]:
        f = project.footage.get(c.footage_id)
        t = recorded_time(f) if f is not None else None
        if t is not None:
            return time.strftime("%a %d %b", time.localtime(t))
    return None


def _merge_short(cuts: list[tuple[int, float]], end: float) -> list[tuple[int, float]]:
    """Chapter starts (clip index, time), the ones shorter than MIN_CHAPTER joined to the one before."""
    out: list[tuple[int, float]] = []
    for k, (i, t) in enumerate(cuts):
        nxt = cuts[k + 1][1] if k + 1 < len(cuts) else end
        if out and (t - out[-1][1] < MIN_CHAPTER or nxt - t < MIN_CHAPTER and k + 1 == len(cuts)):
            continue
        out.append((i, t))
    return out


def make(project: Project) -> tuple[list[Chapter], str]:
    """The edit's chapters and how they were made ("scenes" or "parts"); none for an edit under 30 s."""
    from .bookends import story_order
    from .scenes import time_groups
    clips = project.clips
    end = project.duration
    if end < MIN_CHAPTER * MIN_CHAPTERS or not clips:
        return [], ""
    starts = project.clip_starts()
    ids = list(dict.fromkeys(c.footage_id for c in clips if c.footage_id in project.footage))
    footage = {i: project.footage[i] for i in ids}
    groups = time_groups(footage, story_order(footage, ids))
    # A new scene starts a chapter; the hook, opening and scenic openers belong to the scene they open.
    cuts: list[tuple[int, float]] = [(0, 0.0)]
    prev = None
    for k, c in enumerate(clips):
        if c.label in ("hook", "opening"):
            continue
        g = groups.get(c.footage_id)
        if prev is not None and g is not None and g != prev and k > 0:
            cuts.append((k, starts[k]))
        if g is not None:
            prev = g
    cuts = _merge_short(cuts, end)
    how = "scenes"
    if len(cuts) < MIN_CHAPTERS:              # too few scenes: parts of about a minute, at clip starts
        n = max(MIN_CHAPTERS, min(8, round(end / 60.0)))
        cuts = [(0, 0.0)]
        for k in range(1, n):
            target = end * k / n
            i = min(range(len(clips)), key=lambda j: abs(starts[j] - target))
            if starts[i] - cuts[-1][1] >= MIN_CHAPTER and end - starts[i] >= MIN_CHAPTER:
                cuts.append((i, starts[i]))
        how = "parts"
        if len(cuts) < MIN_CHAPTERS:          # (too few clips to cut at): YouTube would ignore them
            return [], ""
    out = []
    for k, (i, t) in enumerate(cuts):
        i1 = cuts[k + 1][0] if k + 1 < len(cuts) else len(clips)
        said = _first_words(project, i, i1)
        day = _day(project, i, i1) if how == "scenes" else None
        if k == 0 and any(c.label == "hook" for c in clips[:1]):
            title = "Intro"
        elif said and day:
            title = f"{day}: {said}"
        else:
            title = said or day or f"Part {k + 1}"
        out.append(Chapter(round(t, 2), title))
    return out, how


def text(chapters: list[Chapter]) -> str:
    """The chapters as lines for a YouTube description."""
    return "\n".join(f"{stamp(c.start)} {c.title}" for c in chapters)


def parse(lines: str) -> list[Chapter]:
    """Chapters back from (edited) description lines: "m:ss Title" / "h:mm:ss Title"."""
    out = []
    for line in lines.splitlines():
        m = re.match(r"\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s+(.*\S)", line)
        if m:
            h, mi, s, title = m.groups()
            out.append(Chapter(int(h or 0) * 3600 + int(mi) * 60 + int(s), title))
    return out
