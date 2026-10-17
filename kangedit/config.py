"""Application paths and constants."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# The folder KangEdit runs from (holds main.py, or a packaged build's executable); its downloaded FFmpeg
# and default cache live here.
APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent.parent)


def settings():
    """The app's QSettings: the system store, or KangEdit.ini in $KANGEDIT_SETTINGS_DIR (tests, CI, portable use)."""
    from PySide6.QtCore import QSettings
    folder = os.environ.get("KANGEDIT_SETTINGS_DIR")
    if folder:
        return QSettings(os.path.join(folder, "KangEdit.ini"), QSettings.IniFormat)
    return QSettings("KangEdit", "KangEdit")


def app_data_dir() -> Path:
    """Per-user data (LUT and sound libraries): %LOCALAPPDATA%, ~/Library/Application Support or
    ~/.local/share ($KANGEDIT_DATA_DIR overrides it)."""
    if os.environ.get("KANGEDIT_DATA_DIR"):
        base = os.environ["KANGEDIT_DATA_DIR"]
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    path = Path(base) / "KangEdit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ffmpeg_dir() -> Path:
    """Where the automatic FFmpeg download goes (checked before PATH)."""
    from .ffmpeg_setup import bin_dir
    return bin_dir().parent


_cache_root: Optional[Path] = None   # user-chosen cache folder; None = default_cache_root()

# Cache sub-folders and what they hold (shown in the cache dialog).
CACHE_KINDS = {
    "proxies": "Proxies",
    "analysis": "Analysis (best-moment data, thumbnails)",
    "tmp": "Temporary files (export, preview audio)",
}


def lut_dir() -> Path:
    """Imported LUTs. Kept outside the cache so clearing the cache never breaks a grade."""
    path = app_data_dir() / "luts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sfx_dir() -> Path:
    """Sound-effect library (imported and downloaded sounds)."""
    path = app_data_dir() / "sfx"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".kangedit_write_test"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def legacy_cache_root() -> Path:
    """Where caches were kept before they moved next to the app."""
    return app_data_dir() / "cache"


def default_cache_root() -> Path:
    """The "cache" folder next to the app, or the per-user data folder when the app folder is read-only
    (e.g. installed under Program Files or /Applications)."""
    here = APP_DIR / "cache"
    return here if _writable(here) else legacy_cache_root()


def cache_root() -> Path:
    path = _cache_root or default_cache_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_cache_root(path: Optional[str]) -> None:
    """Use `path` for all caches from now on (None or "" restores the default)."""
    global _cache_root
    _cache_root = Path(path) if path else None


def cache_dir(sub: str) -> Path:
    path = cache_root() / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def folder_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def clear_cache(kinds) -> tuple[int, int]:
    """Delete the contents of the given cache sub-folders. Returns (bytes freed, files that were in use)."""
    freed, locked = 0, 0
    for kind in kinds:
        base = cache_root() / kind
        if not base.is_dir():
            continue
        for entry in list(base.iterdir()):
            size = folder_size(entry) if entry.is_dir() else entry.stat().st_size
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                freed += size
            except OSError:
                locked += 1
    return freed, locked


def fmt_bytes(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
    return ""


def file_key(path: str) -> str:
    """Stable key for a media file: path + size + mtime, so caches survive across projects."""
    try:
        st = os.stat(path)
        raw = f"{os.path.abspath(path).lower()}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        raw = os.path.abspath(path).lower()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


VIDEO_EXTENSIONS = (
    ".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v", ".mts", ".m2ts",
    ".webm", ".wmv", ".3gp", ".insv", ".osv", ".lrv", ".ts", ".mpg", ".mpeg",
)

ASPECT_RATIOS = {
    "16:9 (Landscape)": (16, 9),
    "9:16 (Vertical)": (9, 16),
    "1:1 (Square)": (1, 1),
    "4:5 (Portrait)": (4, 5),
    "4:3 (Classic)": (4, 3),
    "2.39:1 (Cinema)": (2.39, 1),
}

# Short side of the frame in pixels.
RESOLUTIONS = {
    "720p": 720,
    "1080p": 1080,
    "1440p": 1440,
    "4K (2160p)": 2160,
}

FRAME_RATES = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]


def frame_size(aspect: str, resolution: str) -> tuple[int, int]:
    """Output width/height (even numbers) for an aspect + short-side resolution."""
    aw, ah = ASPECT_RATIOS[aspect]
    short = RESOLUTIONS[resolution]
    if aw >= ah:
        h = short
        w = short * aw / ah
    else:
        w = short
        h = short * ah / aw
    return int(round(w / 2)) * 2, int(round(h / 2)) * 2


def open_path(path) -> None:
    """Open a file or folder with the system's default application."""
    path = str(path)
    if os.name == "nt":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
