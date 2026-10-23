#!/usr/bin/env python3
"""KangEdit setup for Windows, macOS and Linux (standard library only).

    python install.py              create/update .venv, install the Python packages, download FFmpeg
    python install.py --run [...]  the same (quick when nothing changed), then start KangEdit
    python install.py --no-ffmpeg  skip the FFmpeg download (use the one on PATH)

The launchers run.bat / run.sh / run.command call this on every start, so a changed
requirements.txt is picked up automatically.
"""

from __future__ import annotations

import ctypes.util
import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / ".kangedit-requirements"
WINDOWS = os.name == "nt"


def say(msg: str) -> None:
    print(f"[KangEdit] {msg}", flush=True)


def venv_python(gui: bool = False) -> Path:
    if WINDOWS:
        return VENV / "Scripts" / ("pythonw.exe" if gui else "python.exe")
    return VENV / "bin" / "python"


def check_python() -> None:
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10 or newer is required (this is {sys.version.split()[0]}). "
                 "Get it from https://www.python.org/downloads/")


def ensure_venv() -> None:
    if venv_python().exists():
        return
    say("Creating the virtual environment (.venv)...")
    try:
        venv.EnvBuilder(with_pip=True, clear=VENV.exists()).create(VENV)
    except Exception as e:  # noqa: BLE001
        hint = ("\nOn Debian/Ubuntu install it with:  sudo apt install python3-venv"
                if sys.platform.startswith("linux") else "")
        sys.exit(f"Could not create the virtual environment: {e}{hint}")


def ensure_packages() -> None:
    digest = hashlib.sha256(REQUIREMENTS.read_bytes() + sys.version.encode()).hexdigest()
    if STAMP.exists() and STAMP.read_text().strip() == digest:
        return
    py = str(venv_python())
    say("Installing the Python packages (first run takes a few minutes)...")
    subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", "--upgrade", "pip"], check=False)
    r = subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)])
    if r.returncode != 0:
        sys.exit("Installing the Python packages failed (see the messages above).")
    STAMP.write_text(digest)


def ensure_ffmpeg() -> None:
    sys.path.insert(0, str(ROOT))
    from kangedit import ffmpeg_setup
    if ffmpeg_setup.local_binary("ffmpeg") and ffmpeg_setup.local_binary("ffprobe"):
        return
    found = ffmpeg_setup.installed()
    if found.usable:
        say(f"Using the installed FFmpeg: {found.ffmpeg}")
        return
    last = [-1]

    def show(p: float, msg: str):
        pct = int(p * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r[KangEdit] {msg} ({pct}%)   ", end="", flush=True)

    say(f"{ffmpeg_setup.missing_note()} Downloading a static build...")
    try:
        path = ffmpeg_setup.install(show)
        print()
        say(f"FFmpeg installed in {path}")
    except Exception as e:  # noqa: BLE001
        print()
        say(f"FFmpeg download failed ({e}). Install FFmpeg yourself and put it on PATH "
            "(Windows: winget install Gyan.FFmpeg, macOS: brew install ffmpeg, Linux: your package manager).")


def linux_hints() -> None:
    """Qt needs a few system libraries that minimal Linux installs lack."""
    if not sys.platform.startswith("linux"):
        return
    missing = [lib for lib, name in (("xcb-cursor", "libxcb-cursor0"), ("xkbcommon-x11", "libxkbcommon-x11-0"),
                                     ("EGL", "libegl1"), ("GL", "libgl1"), ("pulse", "libpulse0"))
               if ctypes.util.find_library(lib) is None]
    if missing:
        say("Qt may need these system libraries: " + ", ".join(n for _, n in missing))
        say("  Debian/Ubuntu: sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1 libpulse0")
        say("  Fedora:        sudo dnf install xcb-util-cursor libxkbcommon-x11 mesa-libEGL pulseaudio-libs")
        say("  Arch:          sudo pacman -S xcb-util-cursor libxkbcommon-x11 libglvnd libpulse")


def main() -> None:
    args = sys.argv[1:]
    check_python()
    ensure_venv()
    ensure_packages()
    if "--no-ffmpeg" not in args:
        ensure_ffmpeg()
    linux_hints()
    if "--run" in args:
        rest = args[args.index("--run") + 1:]
        cmd = [str(venv_python(gui=True)), str(ROOT / "main.py")] + rest
        if WINDOWS:   # detach so the console window can close
            subprocess.Popen(cmd, cwd=ROOT, creationflags=0x00000008 | 0x00000200)
        else:
            os.chdir(ROOT)
            os.execv(cmd[0], cmd)
    else:
        say("Ready. Start KangEdit with run.bat (Windows) or ./run.sh (macOS / Linux).")


if __name__ == "__main__":
    main()
