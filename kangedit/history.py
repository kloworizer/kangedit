"""Version history: earlier versions of a saved project, kept so the user can go back further than Undo.

Undo lives in memory and is gone when the app closes. Here every explicit save (and an auto-save at most once
every few minutes) keeps a copy of the project in the per-user data folder, under a folder per project file:
history/<name>-<hash of its path>/<timestamp>.kgproj, with an index.json listing each version's time, clip
count, edit length and an optional label ("Before auto edit"). A version identical to the newest one is not
stored again. The newest MAX_VERSIONS are kept; versions older than MAX_AGE_DAYS go, except the newest
KEEP_RECENT.

Untitled (never-saved) projects have no history: the auto-save recovery copy covers them until the first save,
and their versions start then.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import app_data_dir
from .models import Project

MAX_VERSIONS = 50          # versions kept per project
MAX_AGE_DAYS = 30          # older versions are dropped...
KEEP_RECENT = 10           # ...except the newest this many
AUTOSAVE_EVERY = 300.0     # seconds: at most one auto-save version per project in this time
INDEX_FILE = "index.json"


def history_root() -> Path:
    path = app_data_dir() / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def history_dir(project_path: str) -> Path:
    """The history folder of one project file (named after it, unique per path)."""
    full = os.path.normcase(os.path.abspath(project_path))
    stem = Path(project_path).stem
    return history_root() / f"{stem}-{hashlib.sha1(full.encode('utf-8')).hexdigest()[:8]}"


def _content(project: Project) -> tuple[str, str]:
    """The project as saved JSON, and a hash of it (key order fixed, so equal projects hash alike)."""
    data = project.to_dict()
    text = json.dumps(data, indent=1)
    digest = hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
    return text, digest


def _read_index(d: Path) -> list[dict]:
    try:
        data = json.loads((d / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [v for v in data.get("versions", []) if isinstance(v, dict) and (d / str(v.get("file", ""))).is_file()]


def _write_index(d: Path, project_path: str, entries: list[dict]) -> None:
    tmp = d / (INDEX_FILE + ".tmp")
    tmp.write_text(json.dumps({"source": os.path.abspath(project_path), "versions": entries}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, d / INDEX_FILE)


def versions(project_path: Optional[str]) -> list[dict]:
    """The versions of a project file, newest first: {"file", "time", "clips", "duration", "label", "hash"}."""
    if not project_path:
        return []
    d = history_dir(project_path)
    return sorted(_read_index(d), key=lambda v: v.get("time", 0.0), reverse=True) if d.is_dir() else []


def version_path(project_path: str, entry: dict) -> Path:
    return history_dir(project_path) / entry["file"]


def to_prune(entries: list[dict], now: float, keep: Optional[int] = None, max_age_days: Optional[float] = None,
             keep_recent: Optional[int] = None) -> list[dict]:
    """The versions to drop (entries newest first): beyond the newest `keep`, or older than max_age_days
    unless among the newest keep_recent (the module's limits by default)."""
    keep = MAX_VERSIONS if keep is None else keep
    max_age_days = MAX_AGE_DAYS if max_age_days is None else max_age_days
    keep_recent = KEEP_RECENT if keep_recent is None else keep_recent
    cutoff = now - max_age_days * 86400
    return [v for k, v in enumerate(entries)
            if k >= keep or (k >= keep_recent and v.get("time", 0.0) < cutoff)]


def record(project: Project, label: str = "", min_interval: float = 0.0,
           now: Optional[float] = None) -> Optional[dict]:
    """Keep the project as it is now as a version of its file. Skipped (None) for an untitled project, when
    nothing changed since the newest version (that version then takes the label if it had none), or when
    min_interval is set and the newest version is younger than that. Returns the new index entry."""
    if not project.path:
        return None
    now = time.time() if now is None else now
    text, digest = _content(project)
    d = history_dir(project.path)
    entries = versions(project.path)
    if entries and entries[0].get("hash") == digest:
        if label and not entries[0].get("label"):
            entries[0]["label"] = label
            _write_index(d, project.path, entries)
        return None
    if min_interval and not label and entries and now - entries[0].get("time", 0.0) < min_interval:
        return None
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H-%M-%S")
    name, k = f"{stamp}{Project.FILE_EXT}", 1
    while (d / name).exists():                     # two versions in the same second
        k += 1
        name = f"{stamp} ({k}){Project.FILE_EXT}"
    (d / name).write_text(text, encoding="utf-8")
    entry = {"file": name, "time": now, "clips": len(project.clips), "duration": round(project.duration, 3),
             "label": label, "hash": digest}
    entries.insert(0, entry)
    for old in to_prune(entries, now):
        entries.remove(old)
        try:
            (d / old["file"]).unlink()
        except OSError:
            pass
    _write_index(d, project.path, entries)
    return entry


def load(project_path: str, entry: dict) -> Project:
    """A version as a project (its path is the version file: set it before saving)."""
    return Project.load(str(version_path(project_path, entry)))


def ago(t: float, now: Optional[float] = None) -> str:
    """'just now', '5 min ago', '3 h ago', '2 days ago'."""
    s = max(0.0, (time.time() if now is None else now) - t)
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    days = int(s // 86400)
    return "1 day ago" if days == 1 else f"{days} days ago"


def diff_summary(old: Project, current: Project) -> str:
    """What changed from a version to the current project, in a few words: clips added / removed on the
    main track (by the moment they show) and the change in the edit's length."""
    def moments(p: Project) -> Counter:
        return Counter((c.footage_id, round(c.src_in, 2), round(c.src_out, 2)) for c in p.clips)
    a, b = moments(old), moments(current)
    added, removed = sum((b - a).values()), sum((a - b).values())
    delta = current.duration - old.duration
    parts = []
    if added:
        parts.append(f"{added} clip{'s' if added != 1 else ''} added")
    if removed:
        parts.append(f"{removed} clip{'s' if removed != 1 else ''} removed")
    if abs(delta) >= 0.05:
        parts.append(f"edit {'longer' if delta > 0 else 'shorter'} by {abs(delta):.1f} s")
    if not parts:
        return "The current project has the same clips and length as this version."
    return "Since this version: " + ", ".join(parts) + "."
