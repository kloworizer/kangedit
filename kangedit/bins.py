"""Media panel organisation: bins (folders of footage, saved with the project) and filters.

A bin is just a name on the footage (Footage.bin, "" = not in a bin); empty bins the user made are kept in
ProjectSettings.bins. Bins by date sort footage into "YYYY-MM-DD" bins by when it was recorded (the oldest of
the file's creation time tag, creation date and modification date). Filters show only video or only pictures, and only a shape
of picture (as shown: a phone clip recorded sideways counts as portrait).
"""

from __future__ import annotations

import datetime

from .models import Footage, Project
from .scenes import recorded_time

KINDS = {"all": "All types", "video": "Videos", "image": "Photos / images"}
# key -> (label, aspect ratio or None for a range); exact ratios match within 3 %
RATIOS = {
    "all": ("All ratios", None),
    "landscape": ("Landscape (wider than tall)", None),
    "portrait": ("Portrait (taller than wide)", None),
    "square": ("Square", 1.0),
    "16:9": ("16:9", 16 / 9),
    "9:16": ("9:16", 9 / 16),
    "4:3": ("4:3", 4 / 3),
    "3:4": ("3:4", 3 / 4),
    "21:9": ("21:9 (cinema)", 2.39),
}
ALL, UNSORTED = "\x00all", ""          # bin choices besides the bins themselves
BROLL = "B-roll"                       # footage for cutaways: scenic openers come from it first


def shown_size(f: Footage) -> tuple[int, int]:
    """Width and height of the picture as shown (rotation metadata applied)."""
    if abs(int(f.rotation or 0)) % 180 == 90:
        return f.height, f.width
    return f.width, f.height


def ratio_matches(f: Footage, key: str) -> bool:
    w, h = shown_size(f)
    if key == "all" or w <= 0 or h <= 0:
        return key == "all"
    r = w / h
    if key == "landscape":
        return r > 1.03
    if key == "portrait":
        return r < 0.97
    target = RATIOS.get(key, (None, None))[1]
    if key == "21:9":
        return 2.2 <= r <= 2.45
    return target is not None and abs(r / target - 1.0) <= 0.03


def kind_matches(f: Footage, key: str) -> bool:
    return key == "all" or (key == "image") == f.is_image


def visible(f: Footage, bin_name: str = ALL, kind: str = "all", ratio: str = "all") -> bool:
    """Whether the Media panel shows footage f with this bin (ALL, UNSORTED or a bin) and these filters."""
    if bin_name != ALL and (f.bin or "") != bin_name:
        return False
    return kind_matches(f, kind) and ratio_matches(f, ratio)


def all_bins(project: Project) -> list[str]:
    """Every bin: the ones made by hand (even empty) and the ones footage is in, sorted."""
    names = set(project.settings.bins or []) | {f.bin for f in project.footage.values() if f.bin}
    return sorted(names, key=str.lower)


def add_bin(project: Project, name: str) -> str:
    name = name.strip()
    if name and name not in (project.settings.bins or []):
        project.settings.bins = list(project.settings.bins or []) + [name]
    return name


def rename_bin(project: Project, old: str, new: str) -> None:
    new = new.strip()
    if not new or new == old:
        return
    project.settings.bins = [new if b == old else b for b in (project.settings.bins or [])]
    if new not in project.settings.bins:
        project.settings.bins.append(new)
    for f in project.footage.values():
        if f.bin == old:
            f.bin = new


def delete_bin(project: Project, name: str) -> None:
    """Remove a bin; its footage stays in the project, out of any bin."""
    project.settings.bins = [b for b in (project.settings.bins or []) if b != name]
    for f in project.footage.values():
        if f.bin == name:
            f.bin = ""


def move_to_bin(project: Project, ids: list[str], name: str) -> None:
    if name:
        add_bin(project, name)
    for fid in ids:
        if fid in project.footage:
            project.footage[fid].bin = name


def recorded_date(f: Footage) -> str:
    """The day the footage was recorded, "YYYY-MM-DD" (local time): the oldest of its creation time tag, the
    file's creation date and its modification date (see scenes.recorded_time)."""
    t = recorded_time(f)
    if t is None:
        return "Unknown date"
    return datetime.date.fromtimestamp(t).isoformat()


def clear_bins(project: Project) -> int:
    """Remove every bin; all the media stays in the project, out of any bin. Returns how many items were in
    a bin."""
    n = sum(1 for f in project.footage.values() if f.bin)
    project.settings.bins = []
    for f in project.footage.values():
        f.bin = ""
    return n


def bins_by_date(project: Project, ids: list[str] | None = None) -> dict[str, int]:
    """Put footage (all, or `ids`) in a bin per recording day. Returns {bin: how many went in}."""
    counts: dict[str, int] = {}
    for fid in ids if ids is not None else list(project.footage):
        f = project.footage.get(fid)
        if f is None:
            continue
        day = recorded_date(f)
        add_bin(project, day)
        f.bin = day
        counts[day] = counts.get(day, 0) + 1
    return counts
