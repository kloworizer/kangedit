"""Updates from the GitHub releases of KangEdit.

A release (see .github/workflows/release.yml) carries the packaged builds, KangEdit-<version>-<platform>.zip
(.tar.gz on Linux), and SHA256SUMS.txt. How an update is installed depends on how KangEdit runs:

* packaged build: the platform's archive is downloaded, checked against SHA256SUMS.txt and unpacked; a small
  script waits for KangEdit to close, copies the new files over the app folder and starts it again;
* from source (install.py): the release's source archive replaces the app's files (the virtual
  environment, FFmpeg and caches stay), the requirements are installed, and KangEdit restarts;
* from a git checkout: `git pull --ff-only` (local changes are never overwritten).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import __version__
from .config import APP_DIR

REPO = "kloworizer/kangedit"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
_UA = {"User-Agent": f"KangEdit/{__version__} (update check)", "Accept": "application/vnd.github+json"}
# Never copied over by a source update: the user's environment, downloads and caches.
_KEEP = {".venv", "venv", "ffmpeg", "cache", ".git", "dist", "build", "__pycache__"}

ProgressCb = Callable[[float, str], None]


class UpdateError(RuntimeError):
    pass


@dataclass
class Asset:
    name: str
    url: str
    size: int = 0


@dataclass
class Release:
    version: str
    tag: str
    title: str
    notes: str
    page: str
    source_zip: str
    assets: list[Asset] = field(default_factory=list)


def parse_version(v: str) -> tuple[int, ...]:
    """"v1.2.3" / "1.2.3-beta" -> (1, 2, 3); unknown parts count as 0."""
    nums = re.findall(r"\d+", (v or "").split("-")[0].split("+")[0])
    return tuple(int(n) for n in nums[:3]) + (0,) * (3 - len(nums[:3]))


def is_newer(latest: str, current: str = __version__) -> bool:
    return parse_version(latest) > parse_version(current)


def platform_tag() -> str:
    """Same naming as scripts/build.py: windows-x64, macos-arm64, linux-x64…"""
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        platform.machine().lower(), platform.machine().lower())
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    return f"{system}-{arch}"


def install_kind() -> str:
    """"frozen" (packaged build), "git" (a git checkout) or "source" (unpacked source with install.py)."""
    if getattr(sys, "frozen", False):
        return "frozen"
    if (APP_DIR / ".git").exists():
        return "git"
    return "source"


def _get(url: str, timeout: float = 15.0):
    return urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout)


def release_from_json(data: dict) -> Release:
    tag = data.get("tag_name") or ""
    assets = [Asset(a.get("name", ""), a.get("browser_download_url", ""), int(a.get("size") or 0))
              for a in data.get("assets") or []]
    return Release(version=tag.lstrip("vV"), tag=tag, title=data.get("name") or tag, notes=data.get("body") or "",
                   page=data.get("html_url") or RELEASES_PAGE,
                   source_zip=data.get("zipball_url") or f"https://github.com/{REPO}/archive/refs/tags/{tag}.zip",
                   assets=assets)


def latest_release(timeout: float = 15.0) -> Release:
    """The newest published release (drafts and pre-releases are not offered)."""
    try:
        with _get(API_LATEST, timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except OSError as e:
        raise UpdateError(f"Could not reach GitHub: {e}") from e
    if not data.get("tag_name"):
        raise UpdateError("GitHub did not return a release")
    return release_from_json(data)


def platform_asset(release: Release, tag: Optional[str] = None) -> Optional[Asset]:
    tag = tag or platform_tag()
    for a in release.assets:
        if f"-{tag}." in a.name and a.name.endswith((".zip", ".tar.gz")):
            return a
    return None


def download(url: str, dest: str, progress: Optional[ProgressCb] = None,
             cancel: Optional[threading.Event] = None, label: str = "Downloading the update") -> str:
    part = dest + ".part"
    with _get(url, 60) as r, open(part, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            if cancel is not None and cancel.is_set():
                raise UpdateError("cancelled")
            chunk = r.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if progress:
                progress(min(0.9, 0.9 * got / total) if total else 0.0, f"{label}… {got / 1e6:.0f} MB")
    os.replace(part, dest)
    return dest


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_checksum(release: Release, name: str) -> Optional[str]:
    """The archive's SHA-256 from the release's SHA256SUMS.txt (None when the release has none)."""
    sums = next((a for a in release.assets if a.name.upper().startswith("SHA256SUMS")), None)
    if sums is None:
        return None
    with _get(sums.url) as r:
        for line in r.read().decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lstrip("*") == name:
                return parts[0].lower()
    return None


def _extract(archive: str, dest: str) -> None:
    """Unpack, refusing paths that would land outside dest."""
    root = os.path.realpath(dest)

    def safe(name: str) -> bool:
        target = os.path.realpath(os.path.join(dest, name))
        return target == root or target.startswith(root + os.sep)

    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not safe(info.filename):
                    raise UpdateError(f"unsafe path in the archive: {info.filename}")
            z.extractall(dest)
            if sys.platform != "win32":   # keep the executable bits zip files store
                for info in z.infolist():
                    mode = (info.external_attr >> 16) & 0o777
                    if mode:
                        os.chmod(os.path.join(dest, info.filename), mode)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not safe(m.name) or m.issym() and not safe(os.path.join(os.path.dirname(m.name), m.linkname)):
                    raise UpdateError(f"unsafe path in the archive: {m.name}")
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, filter="data")
            else:
                t.extractall(dest)


def _writable(folder: Path) -> bool:
    try:
        probe = folder / ".kangedit_update_test"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


@dataclass
class Prepared:
    """A downloaded update, ready to install when KangEdit closes."""
    kind: str
    version: str
    staging: str                  # unpacked new version
    command: list[str]            # starts the installer helper (frozen) or the restarted app (source)
    message: str = ""


def app_bundle() -> Path:
    """The folder a packaged update replaces: KangEdit.app on macOS, else the app's folder."""
    if sys.platform == "darwin" and getattr(sys, "frozen", False):
        return APP_DIR.parents[1]    # KangEdit.app/Contents/MacOS -> KangEdit.app
    return APP_DIR


def update_log() -> Path:
    """Where the Windows update helper writes what it did (the copy's summary and result)."""
    from .crashlog import log_dir
    return log_dir() / "update.log"


def _helper_frozen(staging_app: Path, target: Path, workdir: str, log: Optional[Path] = None) -> list[str]:
    """A script that waits for this process to end, copies the new build over the old one and restarts it."""
    pid = os.getpid()
    if sys.platform == "win32":
        exe = target / "KangEdit.exe"
        script = Path(workdir) / "install_update.bat"
        log = log or update_log()
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        # Runs in a hidden console (see launch): every command shares it, so no window opens. "timeout" needs
        # an interactive console, so the one-second pause is a ping; after ~2 minutes it goes on regardless.
        # robocopy copies every file (/IS /IT): by default it skips files of the same size and time stamp,
        # which a new build can have although they changed. What it did goes to the update log.
        script.write_text(
            "@echo off\r\n"
            f'echo KangEdit update to "{target}" >"{log}" 2>NUL\r\n'
            "set n=0\r\n"
            ":wait\r\n"
            f'tasklist /FI "PID eq {pid}" /NH 2>NUL | find "{pid}" >NUL || goto copy\r\n'
            "set /a n+=1\r\n"
            "if %n% GEQ 120 goto copy\r\n"
            "ping -n 2 127.0.0.1 >NUL\r\n"
            "goto wait\r\n"
            ":copy\r\n"
            f'robocopy "{staging_app}" "{target}" /E /IS /IT /R:10 /W:1 /NP /NFL /NDL >>"{log}" 2>&1\r\n'
            f'echo robocopy result %ERRORLEVEL% (0-7: copied, 8 or more: failed) >>"{log}" 2>NUL\r\n'
            f'start "" "{exe}"\r\n'
            f'rmdir /S /Q "{workdir}"\r\n', encoding="utf-8")
        return ["cmd", "/c", str(script)]
    script = Path(workdir) / "install_update.sh"
    if sys.platform == "darwin":
        copy = f'rsync -a --delete "{staging_app}/" "{target}/"'
        start = f'open "{target}"'
    else:
        copy = f'cp -a "{staging_app}/." "{target}/"'
        start = f'"{target / "KangEdit"}" >/dev/null 2>&1 &'
    script.write_text("#!/bin/sh\n"
                      f"while kill -0 {pid} 2>/dev/null; do sleep 1; done\n"
                      f"{copy}\n{start}\n"
                      f'rm -rf "{workdir}"\n', encoding="utf-8")
    os.chmod(script, 0o755)
    return ["/bin/sh", str(script)]


def prepare(release: Release, progress: Optional[ProgressCb] = None,
            cancel: Optional[threading.Event] = None, workdir: Optional[str] = None) -> Prepared:
    """Download and unpack the update for this installation (in the background)."""
    kind = install_kind()
    if kind == "git":
        return Prepared(kind, release.version, "", ["git", "pull", "--ff-only"],
                        "This copy of KangEdit is a git checkout: it is updated with “git pull”.")
    work = workdir or tempfile.mkdtemp(prefix="kangedit-update-")
    if kind == "frozen":
        asset = platform_asset(release)
        if asset is None:
            raise UpdateError(f"The release has no build for {platform_tag()}: download it from {release.page}")
        target = app_bundle()
        if not _writable(target.parent if sys.platform == "darwin" else target):
            raise UpdateError(f"KangEdit's folder can't be written to ({target}). Download the update from "
                              f"{release.page}")
        archive = download(asset.url, os.path.join(work, asset.name), progress, cancel)
        if progress:
            progress(0.92, "Checking the download…")
        want = expected_checksum(release, asset.name)
        if want and sha256(archive) != want:
            raise UpdateError("The download is damaged (checksum mismatch). Try again later.")
        staging = os.path.join(work, "new")
        _extract(archive, staging)
        name = "KangEdit.app" if sys.platform == "darwin" else "KangEdit"
        staging_app = Path(staging) / name
        if not staging_app.exists():
            raise UpdateError(f"The archive has no {name} folder")
        if progress:
            progress(1.0, "Update ready")
        return Prepared(kind, release.version, str(staging_app), _helper_frozen(staging_app, target, work))
    # source
    if not _writable(APP_DIR):
        raise UpdateError(f"KangEdit's folder can't be written to ({APP_DIR})")
    archive = download(release.source_zip, os.path.join(work, "source.zip"), progress, cancel)
    staging = os.path.join(work, "new")
    _extract(archive, staging)
    roots = [p for p in Path(staging).iterdir() if p.is_dir()]
    if len(roots) != 1 or not (roots[0] / "kangedit" / "__init__.py").exists():
        raise UpdateError("The source archive doesn't look like KangEdit")
    if progress:
        progress(1.0, "Update ready")
    return Prepared(kind, release.version, str(roots[0]), [sys.executable, str(APP_DIR / "main.py")])


def install_source(p: Prepared, progress: Optional[ProgressCb] = None) -> None:
    """Copy a prepared source update over the app's folder and install its requirements."""
    src = Path(p.staging)
    for item in src.iterdir():
        if item.name in _KEEP:
            continue
        dst = APP_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*_KEEP, "*.pyc"))
        else:
            shutil.copy2(item, dst)
    req = APP_DIR / "requirements.txt"
    if req.exists():
        if progress:
            progress(0.95, "Installing requirements…")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise UpdateError("The new files are in place, but installing the requirements failed:\n"
                              + (r.stderr or r.stdout)[-1500:])


def git_pull() -> str:
    r = subprocess.run(["git", "pull", "--ff-only"], cwd=APP_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise UpdateError((r.stderr or r.stdout).strip() or "git pull failed")
    return r.stdout.strip()


WIN_NO_WINDOW = 0x08000000           # CREATE_NO_WINDOW
WIN_NEW_PROCESS_GROUP = 0x00000200   # CREATE_NEW_PROCESS_GROUP


def launch(command: list[str]) -> None:
    """Start the helper / the new app detached from this process (it must outlive it)."""
    kw: dict = {"cwd": str(APP_DIR), "close_fds": True}
    if sys.platform == "win32":
        # A hidden console of its own (CREATE_NO_WINDOW), not none at all (DETACHED_PROCESS): without one,
        # every console command in the helper script opened a window of its own.
        kw["creationflags"] = WIN_NO_WINDOW | WIN_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
