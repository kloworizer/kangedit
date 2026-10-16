"""KangEdit - automatic best-moment video editor."""

# Single source of the version: release tags (vX.Y.Z) must match it; scripts/bump_version.py changes it.
__version__ = "0.5.0"
APP_NAME = "KangEdit"

try:   # written by scripts/build.py for packaged builds: commit and date
    from ._build_info import BUILD
except ImportError:
    BUILD = ""


def version_string() -> str:
    """"1.2.3", plus the commit and date for packaged builds ("1.2.3 (abc1234, 2026-10-11)")."""
    return f"{__version__} ({BUILD})" if BUILD else __version__
