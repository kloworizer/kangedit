"""Black frames and gaps on the main track.

The main track is a row of clips with nothing between them. Black is a clip too: of a black picture KangEdit
makes itself (Footage.generated == "black"), so the preview, the export, transitions and undo treat it like any
photo. "Insert black" puts one in; with the timeline's magnet off, moving a clip leaves black where it was and
it can be dropped anywhere, also into a black gap or past the end (filling the space between with black), and
deleting a clip leaves a gap instead of closing it up.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import images
from .models import Footage, Project, TimelineClip

BLACK = "black"
MIN_GAP = 1e-3            # shorter black left over is dropped
BLACK_LENGTHS = [("1 frame", 0.0), ("0.5 s", 0.5), ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0)]   # 0: one frame


def black_path() -> str:
    """The black picture's file (made the first time)."""
    from .config import app_data_dir
    path = os.path.join(str(app_data_dir()), "black.png")
    if not os.path.exists(path):
        import cv2
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, np.zeros((16, 16, 3), np.uint8))
    return path


def is_black(project: Project, clip) -> bool:
    f = project.footage.get(getattr(clip, "footage_id", ""))
    return f is not None and f.generated == BLACK


def black_footage(project: Project) -> Footage:
    """The project's black picture (added, hidden from the Media panel, the first time)."""
    for f in project.footage.values():
        if f.generated == BLACK:
            if not os.path.exists(f.path):
                f.path = black_path()
            return f
    f = Footage(black_path(), images.STILL_LENGTH, 16, 16, 30.0, False, codec="png", kind="image",
                generated=BLACK)
    project.footage[f.id] = f
    return f


def repair(project: Project) -> None:
    """Point the black footage at this computer's black picture (a project opened elsewhere)."""
    for f in project.footage.values():
        if f.generated == BLACK and not os.path.exists(f.path):
            f.path = black_path()


def black_clip(project: Project, length: float) -> TimelineClip:
    f = black_footage(project)
    return TimelineClip(f.id, 0.0, max(MIN_GAP, min(images.STILL_LENGTH, length)))


def frame_length(project: Project) -> float:
    return 1.0 / max(1.0, project.settings.fps)


def insert_black(project: Project, t: float, length: float) -> TimelineClip:
    """Black `length` seconds long (0: one frame) at timeline time t: at the cut nearest t, or splitting the black
    already there. Returns the new clip."""
    length = length if length > 0 else frame_length(project)
    clip = black_clip(project, length)
    starts = project.clip_starts()
    at = len(project.clips)
    for k, (s, c) in enumerate(zip(starts, project.clips)):
        if t < s + c.duration / 2:
            at = k
            break
    project.clips.insert(at, clip)
    tidy(project, keep=clip.id)
    return clip


def tidy(project: Project, keep: str = "", trailing: bool = True) -> None:
    """Join black clips next to each other into one and drop empty ones (and black at the very end:
    trailing). keep: a black clip that stays as it is (just inserted: a black frame on its own)."""
    out: list[TimelineClip] = []
    for c in project.clips:
        black = is_black(project, c)
        if black and c.duration < MIN_GAP:
            continue
        if black and out and is_black(project, out[-1]) and keep not in (c.id, out[-1].id) \
                and not c.transition and not out[-1].transition:
            prev = out[-1]
            prev.src_out = prev.src_in + prev.duration + c.duration
            continue
        out.append(c)
    while trailing and out and is_black(project, out[-1]) and out[-1].id != keep:
        out.pop()
    project.clips[:] = out


def lift(project: Project, ids: list[str]) -> int:
    """Take clips out leaving black gaps of their length (deleting with the magnet off). Returns how many."""
    n = 0
    for i, c in enumerate(list(project.clips)):
        if c.id in ids and not is_black(project, c):
            gap = black_clip(project, c.duration)
            project.clips[i] = gap
            n += 1
    tidy(project)
    return n


def place(project: Project, clip_id: str, start: float) -> Optional[int]:
    """Move a clip so it starts at timeline time `start`, the magnet off: where it was is left black; dropped
    on black it takes that space (pushing what follows only when the black is too short), dropped on a clip it
    goes in at the nearest cut, dropped past the end it comes after black filling the space between. Returns
    the clip's new index (None if it isn't on the track)."""
    i = project.index_of(clip_id)
    if i < 0:
        return None
    clip = project.clips[i]
    project.clips[i] = black_clip(project, clip.duration)       # its place stays empty
    tidy(project, trailing=False)
    start = max(0.0, start)
    starts = project.clip_starts()
    end = project.duration
    if start >= end - 1e-6:
        if start - end > MIN_GAP:
            project.clips.append(black_clip(project, start - end))
        project.clips.append(clip)
        tidy(project)
        return project.index_of(clip.id)
    for k, (s, c) in enumerate(zip(starts, project.clips)):
        if not (s - 1e-9 <= start < s + c.duration):
            continue
        if is_black(project, c):
            before = start - s
            after = c.duration - before - clip.duration   # black left after it (< 0: not enough room)
            pieces = []
            if before > MIN_GAP:
                pieces.append(black_clip(project, before))
            pieces.append(clip)
            if after > MIN_GAP:
                pieces.append(black_clip(project, after))
            project.clips[k:k + 1] = pieces
            if after < -MIN_GAP:            # too short: take what it needs from black right after, if any
                j = k + len(pieces)
                if j < len(project.clips) and is_black(project, project.clips[j]):
                    nxt = project.clips[j]
                    nxt.src_out = nxt.src_in + max(0.0, nxt.duration + after)
        else:
            at = k if start < s + c.duration / 2 else k + 1
            project.clips.insert(at, clip)
        break
    tidy(project)
    return project.index_of(clip.id)
