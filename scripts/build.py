#!/usr/bin/env python3
"""Build a packaged KangEdit for this platform and archive it in dist/.

    python scripts/build.py              build, self-test, archive
    python scripts/build.py --no-test    skip running the built app's self-test

Writes kangedit/_build_info.py (commit and date, shown in About and --version), runs PyInstaller with
packaging/kangedit.spec, starts the built app with --self-test, then archives it as
dist/KangEdit-<version>-<platform>.zip (.tar.gz on Linux). Prints the archive path last.
FFmpeg isn't bundled: the app downloads it on first start, as the source version does.
"""

from __future__ import annotations

import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def version() -> str:
    text = (ROOT / "kangedit" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


def platform_tag() -> str:
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        platform.machine().lower(), platform.machine().lower())
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    return f"{system}-{arch}"


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_build_info() -> str:
    sha = os.environ.get("GITHUB_SHA", "")[:7] or git("rev-parse", "--short", "HEAD") or "local"
    build = f"{sha}, {datetime.date.today().isoformat()}"
    (ROOT / "kangedit" / "_build_info.py").write_text(f'BUILD = "{build}"\n', encoding="utf-8")
    return build


def app_executable() -> Path:
    if sys.platform == "win32":
        return DIST / "KangEdit" / "KangEdit.exe"
    if sys.platform == "darwin":
        return DIST / "KangEdit.app" / "Contents" / "MacOS" / "KangEdit"
    return DIST / "KangEdit" / "KangEdit"


def self_test(exe: Path) -> None:
    """Start the built app with --self-test; it reports through a log file (a windowed app has no console)."""
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "selftest.log"
        env = dict(os.environ, KANGEDIT_SELFTEST_LOG=str(log), QT_QPA_PLATFORM="offscreen")
        r = subprocess.run([str(exe), "--self-test"], env=env, timeout=300)
        report = log.read_text(encoding="utf-8") if log.exists() else "(no report)"
        print(report)
        if r.returncode != 0 or "self-test OK" not in report:
            sys.exit(f"Self-test of the built app failed (exit code {r.returncode}).")


def copy_license() -> None:
    """The license and third-party notices go with every build (the license requires its notice to)."""
    target = DIST / "KangEdit.app" / "Contents" / "Resources" if sys.platform == "darwin" else DIST / "KangEdit"
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT / name, target / name)


def archive() -> Path:
    name = f"KangEdit-{version()}-{platform_tag()}"
    if sys.platform == "darwin":   # ditto keeps the framework symlinks and code signatures inside the .app
        out = DIST / f"{name}.zip"
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(DIST / "KangEdit.app"), str(out)], check=True)
        return out
    fmt = "zip" if sys.platform == "win32" else "gztar"   # tar keeps the executable bit
    return Path(shutil.make_archive(str(DIST / name), fmt, root_dir=DIST, base_dir="KangEdit"))


def main() -> None:
    build = write_build_info()
    print(f"Building KangEdit {version()} ({build}) for {platform_tag()}", flush=True)
    for path in (DIST, ROOT / "build"):
        shutil.rmtree(path, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(DIST), "--workpath", str(ROOT / "build"),
                    str(ROOT / "packaging" / "kangedit.spec")], cwd=ROOT, check=True)
    if "--no-test" not in sys.argv:
        self_test(app_executable())
    copy_license()
    out = archive()
    print(f"Archive: {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print(out)


if __name__ == "__main__":
    main()
