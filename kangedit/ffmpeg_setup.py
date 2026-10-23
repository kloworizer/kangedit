"""Automatic FFmpeg download (standard library only: also used by setup.py before dependencies exist).

Static builds with the GPU encoders (NVENC, AMF, Quick Sync / VideoToolbox) are fetched into
<app folder>/ffmpeg/bin, which ffmpeg.ffmpeg_bin() checks before PATH:
  * Windows and Linux: BtbN's daily FFmpeg builds (github.com/BtbN/FFmpeg-Builds)
  * macOS: Martin Riedl's static builds (ffmpeg.martin-riedl.de)
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, NamedTuple, Optional

_BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
_RIEDL = "https://ffmpeg.martin-riedl.de/redirect/latest/macos/{arch}/release/{tool}.zip"
_GYAN = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"          # Windows, when GitHub fails
_JVS = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-{arch}-static.tar.xz"   # Linux, likewise

# The export writes its filter graphs to files read with -/filter_complex, new in FFmpeg 7.0.
MIN_VERSION = 7
STALL = 30      # seconds a download may receive nothing before that server is given up

Progress = Callable[[float, str], None]


def bin_dir() -> Path:
    """<app folder>/ffmpeg/bin; for an installed build whose folder is read-only (Program Files,
    /Applications) the per-user data folder instead."""
    from .config import APP_DIR, _writable, app_data_dir   # standard library only
    here = APP_DIR / "ffmpeg" / "bin"
    if getattr(sys, "frozen", False) and not here.is_dir() and not _writable(APP_DIR / "ffmpeg"):
        return app_data_dir() / "ffmpeg" / "bin"
    return here


def exe(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def local_binary(name: str) -> Optional[str]:
    path = bin_dir() / exe(name)
    return str(path) if path.is_file() else None


class Installed(NamedTuple):
    """An FFmpeg installed on the computer (not the app's own download)."""
    ffmpeg: Optional[str]
    ffprobe: Optional[str]
    version: Optional[tuple]       # (major, minor); (99, 0) for a build from the development tree
    usable: bool                   # both programs, and new enough (MIN_VERSION)


def _windows_dirs() -> list[Path]:
    """Where FFmpeg installers put it on Windows, also when that folder isn't on PATH (yet): winget,
    Chocolatey, Scoop, C:\\ffmpeg."""
    if os.name != "nt":
        return []
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    out = [local / "Microsoft" / "WinGet" / "Links",
           Path(os.environ.get("ProgramData") or r"C:\ProgramData") / "chocolatey" / "bin",
           home / "scoop" / "shims", Path(r"C:\ffmpeg\bin"),
           Path(os.environ.get("ProgramFiles") or r"C:\Program Files") / "ffmpeg" / "bin"]
    try:
        out += sorted((local / "Microsoft" / "WinGet" / "Packages").glob("*FFmpeg*/*/bin"), reverse=True)
    except OSError:
        pass
    return out


def version_of(path: str) -> Optional[tuple]:
    """FFmpeg's version (major, minor) from `ffmpeg -version`; None when it doesn't run."""
    import re
    import subprocess
    try:
        out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=15,
                             creationflags=0x08000000 if os.name == "nt" else 0).stdout or ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    m = re.search(r"version\s+n?(\d+)\.(\d+)", out)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (99, 0) if out.strip() else None     # "N-117000-g…", "2026-01-01-git-…": newer than any release


_installed: Optional[Installed] = None


def installed(refresh: bool = False) -> Installed:
    """The FFmpeg installed on the computer: the first new enough on PATH or in an installer's folder, else
    the first found (to say why it can't be used). Looked for once."""
    global _installed
    if _installed is not None and not refresh:
        return _installed
    dirs = [Path(d) for d in os.environ.get("PATH", "").split(os.pathsep) if d] + _windows_dirs()
    first = None
    for d in dict.fromkeys(dirs):
        f, p = d / exe("ffmpeg"), d / exe("ffprobe")
        try:
            if not f.is_file():
                continue
        except OSError:
            continue
        v = version_of(str(f))
        if v is None:
            continue
        probe = str(p) if p.is_file() else shutil.which("ffprobe")
        found = Installed(str(f), probe, v, bool(probe) and v[0] >= MIN_VERSION)
        if found.usable:
            _installed = found
            return found
        first = first or found
    _installed = first or Installed(None, shutil.which("ffprobe"), None, False)
    return _installed


def installed_binary(name: str) -> Optional[str]:
    """ffmpeg / ffprobe of a usable FFmpeg installed on the computer, else None."""
    found = installed()
    return getattr(found, name) if found.usable else None


def missing_note() -> str:
    """Why KangEdit has no FFmpeg it can use: what was found on the computer, if anything."""
    found = installed()
    if found.ffmpeg and found.version and found.version[0] < MIN_VERSION:
        return (f"FFmpeg {found.version[0]}.{found.version[1]} is installed ({found.ffmpeg}), but KangEdit needs "
                f"FFmpeg {MIN_VERSION} or newer.")
    if found.ffmpeg and not found.ffprobe:
        return f"FFmpeg is installed ({found.ffmpeg}), but not FFprobe, which comes with it and KangEdit needs too."
    if found.ffprobe and not found.ffmpeg:
        return f"FFprobe is installed ({found.ffprobe}), but not FFmpeg itself."
    return "KangEdit needs FFmpeg, which was not found on this computer."


def _arch() -> str:
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64", "armv8l") else "x86_64"


def mirrors() -> list[list[str]]:
    """Where FFmpeg is downloaded from on this platform, in order: each a list of archive URLs that together
    hold ffmpeg and ffprobe (the next is tried when one fails: a server blocked or not answering)."""
    arch = _arch()
    if os.name == "nt":
        if arch == "arm64":
            return [[_BTBN + "ffmpeg-master-latest-winarm64-gpl.zip"]]
        return [[_BTBN + "ffmpeg-master-latest-win64-gpl.zip"], [_GYAN]]
    if sys.platform == "darwin":
        a = "arm64" if arch == "arm64" else "amd64"
        return [[_RIEDL.format(arch=a, tool="ffmpeg"), _RIEDL.format(arch=a, tool="ffprobe")]]
    if sys.platform.startswith("linux"):
        if arch == "arm64":
            return [[_BTBN + "ffmpeg-master-latest-linuxarm64-gpl.tar.xz"], [_JVS.format(arch="arm64")]]
        return [[_BTBN + "ffmpeg-master-latest-linux64-gpl.tar.xz"], [_JVS.format(arch="amd64")]]
    raise RuntimeError(f"No automatic FFmpeg download for {sys.platform}; install FFmpeg and put it on PATH.")


def sources() -> list[str]:
    """The first mirror's archive URLs (see mirrors)."""
    return mirrors()[0]


def host(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc


def _download(url: str, progress: Optional[Progress], label: str, cancel=None) -> bytes:
    if progress:
        progress(0.0, f"{label}: connecting to {host(url)}…")
    req = urllib.request.Request(url, headers={"User-Agent": "KangEdit-setup"})
    # (the timeout is per network wait: a server that stops sending is given up after STALL seconds)
    with urllib.request.urlopen(req, timeout=STALL) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        while True:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            chunk = resp.read(1 << 18)
            if not chunk:
                break
            buf.write(chunk)
            if progress:
                mb = buf.tell() / 1e6
                progress(buf.tell() / total if total else 0.0,
                         f"{label} from {host(url)}: {mb:.0f}" + (f" / {total / 1e6:.0f} MB" if total else " MB"))
        if total and buf.tell() < total:
            raise RuntimeError(f"the download stopped at {buf.tell() / 1e6:.0f} of {total / 1e6:.0f} MB")
        return buf.getvalue()


def _extract(data: bytes, url: str, dest: Path) -> list[str]:
    """Write the ffmpeg / ffprobe executables found in the archive into dest."""
    wanted = {exe("ffmpeg"), exe("ffprobe")}
    found = []
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                name = os.path.basename(info.filename)
                if name in wanted and not info.is_dir():
                    with zf.open(info) as src, open(dest / name, "wb") as out:
                        shutil.copyfileobj(src, out)
                    found.append(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for member in tf.getmembers():
                name = os.path.basename(member.name)
                if name in wanted and member.isfile():
                    with tf.extractfile(member) as src, open(dest / name, "wb") as out:
                        shutil.copyfileobj(src, out)
                    found.append(name)
    return found


def _fetch(urls: list[str], tmp: Path, progress: Optional[Progress], cancel) -> list[str]:
    """ffmpeg and ffprobe from one mirror's archives, extracted into tmp."""
    found: list[str] = []
    for k, url in enumerate(urls):
        label = f"Downloading FFmpeg ({k + 1}/{len(urls)})" if len(urls) > 1 else "Downloading FFmpeg"
        data = _download(url, (lambda p, m, k=k: progress((k + p) / len(urls), m)) if progress else None, label,
                         cancel)
        found += _extract(data, url, tmp)
    missing = {exe("ffmpeg"), exe("ffprobe")} - set(found)
    if missing:
        raise RuntimeError("the download did not contain " + ", ".join(sorted(missing)))
    return found


def install(progress: Optional[Progress] = None, cancel=None) -> Path:
    """Download FFmpeg + FFprobe into bin_dir(), from the first mirror that works. Returns that folder."""
    dest = bin_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="bin_", dir=dest.parent))
    try:
        found: list[str] = []
        errors: list[str] = []
        options = mirrors()
        for n, urls in enumerate(options):
            try:
                found = _fetch(urls, tmp, progress, cancel)
                break
            except Exception as e:  # noqa: BLE001 - blocked, not answering, cut off: the next mirror
                if cancel is not None and cancel.is_set():
                    raise
                errors.append(f"{host(urls[0])}: {e}")
                for p in tmp.iterdir():
                    p.unlink()
                if progress and n + 1 < len(options):
                    progress(0.0, f"{host(urls[0])} didn't work ({e}); trying {host(options[n + 1][0])}…")
        if not found:
            raise RuntimeError("\n".join(errors))
        for name in found:
            p = tmp / name
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            if sys.platform == "darwin":   # files fetched by a program normally aren't quarantined; be sure
                os.system(f"xattr -d com.apple.quarantine '{p}' 2>/dev/null")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure(progress: Optional[Progress] = None) -> Optional[str]:
    """Path of a usable ffmpeg: the downloaded one, else PATH, else download it now."""
    if local_binary("ffmpeg") and local_binary("ffprobe"):
        return local_binary("ffmpeg")
    if installed().usable:
        return installed().ffmpeg
    install(progress)
    return local_binary("ffmpeg")


if __name__ == "__main__":   # python -m kangedit.ffmpeg_setup [--force]
    last = [-1]

    def show(p: float, msg: str):
        pct = int(p * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r{msg} ({pct}%)   ", end="", flush=True)

    if "--force" in sys.argv:
        print("Installed to", install(show))
    else:
        print("\nFFmpeg:", ensure(show))
