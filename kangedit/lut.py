"""Imported .cube LUTs: parsing, interpolation and the LUT library.

A LUT is used in one of two places (see color.apply_grade):
  * as the footage's log conversion ("lut:<path>" instead of a built-in camera profile), applied to the
    decoded signal before any grading, e.g. a manufacturer's "S-Log3 to Rec.709" LUT;
  * as a clip's creative look, applied to the graded Rec.709 image and mixed in by a strength amount.
"""

from __future__ import annotations

import filecmp
import os
import re
import shutil
import tempfile
import threading
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import lut_dir

LUT_PREFIX = "lut:"
LUT_EXTENSIONS = (".cube",)


class LutError(ValueError):
    pass


def _interp_1d(table: np.ndarray, dmin: np.ndarray, span: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    n = len(table) - 1
    x = np.clip((np.asarray(rgb, np.float32) - dmin) / span, 0.0, 1.0) * n
    i0 = np.minimum(x.astype(np.int32), n - 1)
    f = x - i0
    out = np.empty_like(x)
    for c in range(3):
        col = table[:, c]
        out[:, c] = col[i0[:, c]] * (1.0 - f[:, c]) + col[i0[:, c] + 1] * f[:, c]
    return out


def _interp_3d(t: np.ndarray, dmin: np.ndarray, span: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    n = t.shape[0] - 1
    x = np.clip((np.asarray(rgb, np.float32) - dmin) / span, 0.0, 1.0) * n
    i0 = np.minimum(x.astype(np.int32), n - 1)
    f = x - i0
    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    c00 = t[b0, g0, r0] * (1 - fr) + t[b0, g0, r0 + 1] * fr
    c01 = t[b0, g0 + 1, r0] * (1 - fr) + t[b0, g0 + 1, r0 + 1] * fr
    c10 = t[b0 + 1, g0, r0] * (1 - fr) + t[b0 + 1, g0, r0 + 1] * fr
    c11 = t[b0 + 1, g0 + 1, r0] * (1 - fr) + t[b0 + 1, g0 + 1, r0 + 1] * fr
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    return (c0 * (1 - fb) + c1 * fb).astype(np.float32)


class CubeLut:
    """A 1D or 3D LUT from a .cube file (Adobe/Resolve format). A Resolve-style file with both LUT_1D_SIZE and
    LUT_3D_SIZE is a 1D shaper (its rows come first, read through LUT_1D_INPUT_RANGE) followed by the 3D LUT
    (LUT_3D_INPUT_RANGE)."""

    def __init__(self, path: str):
        self.path = path
        self.title = ""
        size_1d = size_3d = 0
        rng1 = [np.zeros(3, np.float32), np.ones(3, np.float32)]      # 1D (shaper) input range
        rng3 = [np.zeros(3, np.float32), np.ones(3, np.float32)]      # 3D input range (DOMAIN_* sets both)
        rows: list[list[float]] = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    line = raw.split("#", 1)[0].strip()
                    if not line:
                        continue
                    head = line.split(None, 1)[0].upper()
                    if head[0].isdigit() or head[0] in "-+.":
                        vals = line.split()
                        if len(vals) != 3:
                            raise LutError(f"bad data line: {line!r}")
                        rows.append([float(v) for v in vals])
                    elif head == "TITLE":
                        self.title = line[5:].strip().strip('"')
                    elif head == "LUT_3D_SIZE":
                        size_3d = int(line.split()[1])
                    elif head == "LUT_1D_SIZE":
                        size_1d = int(line.split()[1])
                    elif head in ("LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE"):
                        lo, hi = (float(v) for v in line.split()[1:3])
                        target = rng1 if head.startswith("LUT_1D") else rng3
                        target[0][:], target[1][:] = lo, hi
                    elif head == "DOMAIN_MIN":
                        rng1[0][:] = rng3[0][:] = [float(v) for v in line.split()[1:4]]
                    elif head == "DOMAIN_MAX":
                        rng1[1][:] = rng3[1][:] = [float(v) for v in line.split()[1:4]]
                    # Other keywords (LUT_IN_VIDEO_RANGE, vendor tags...) are ignored.
        except OSError as e:
            raise LutError(str(e)) from e
        except (ValueError, IndexError) as e:
            raise LutError(f"not a valid .cube file ({e})") from e

        data = np.asarray(rows, np.float32)
        self.shaper = None
        if size_3d:
            n3 = size_3d ** 3
            extra = len(data) - n3
            if size_1d and extra >= 2:
                # Resolve: the 1D shaper first, the 3D LUT after it (the 3D part is always the last n3 rows).
                self.shaper = data[:extra]
                data = data[extra:]
            elif len(data) != n3:
                raise LutError(f"expected {n3} entries for a {size_3d}³ LUT, found {len(data)}")
            # .cube order: red changes fastest -> index as [b, g, r].
            self.table = data.reshape(size_3d, size_3d, size_3d, 3)
            self.kind = "3d"
            self.size = size_3d
            rng = rng3
        elif size_1d:
            if len(data) != size_1d:
                raise LutError(f"expected {size_1d} entries for a 1D LUT, found {len(data)}")
            self.table = data
            self.kind = "1d"
            self.size = size_1d
            rng = rng1
        else:
            raise LutError("missing LUT_3D_SIZE / LUT_1D_SIZE")
        if self.size < 2:
            raise LutError("LUT size must be at least 2")
        self.dmin = rng[0]
        self.span = np.maximum(rng[1] - rng[0], 1e-6)
        self.shaper_dmin = rng1[0]
        self.shaper_span = np.maximum(rng1[1] - rng1[0], 1e-6)

    @property
    def name(self) -> str:
        return lut_name(self.path)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (N, 3) float -> (N, 3) float32, trilinear (3D) or linear (1D) interpolation; through the shaper
        first when the file has one."""
        if self.shaper is not None:
            rgb = _interp_1d(self.shaper, self.shaper_dmin, self.shaper_span, rgb)
        if self.kind == "1d":
            return _interp_1d(self.table, self.dmin, self.span, rgb)
        return _interp_3d(self.table, self.dmin, self.span, rgb)


_cache: dict[str, tuple[float, CubeLut]] = {}
_cache_lock = threading.Lock()


def load(path: str) -> CubeLut:
    """Parsed LUT, cached until the file changes."""
    mtime = os.path.getmtime(path)
    with _cache_lock:
        hit = _cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
    lut = CubeLut(path)
    with _cache_lock:
        _cache[path] = (mtime, lut)
    return lut


def try_load(path: str) -> Optional[CubeLut]:
    """LUT or None when the file is missing/broken (the image is then left unchanged)."""
    try:
        return load(path) if path else None
    except (OSError, LutError):
        return None


def lut_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def library() -> list[tuple[str, str]]:
    """(name, path) for every LUT in the library folder, sorted by name."""
    items = []
    for entry in os.scandir(lut_dir()):
        if entry.is_file() and entry.name.lower().endswith(LUT_EXTENSIONS):
            items.append((lut_name(entry.path), entry.path))
    return sorted(items, key=lambda it: it[0].lower())


def import_lut(src: str) -> str:
    """Validate a .cube file and copy it into the library. Returns the library path."""
    CubeLut(src)  # raises LutError if unusable
    dst_dir = lut_dir()
    base, ext = os.path.splitext(os.path.basename(src))
    dst = str(dst_dir / f"{base}{ext.lower()}")
    if os.path.abspath(src).lower() == os.path.abspath(dst).lower():
        return dst
    k = 2
    while os.path.exists(dst):
        if filecmp.cmp(src, dst, shallow=False):
            return dst  # already imported
        dst = str(dst_dir / f"{base} ({k}){ext.lower()}")
        k += 1
    shutil.copy2(src, dst)
    return dst


# ----------------------------------------------------------------------------- official camera LUTs
# Manufacturers publish their own log -> Rec.709 conversion LUTs. Most sit behind download pages and
# license terms, so the app doesn't fetch them: the LUT dialog links each page, and once the files are
# imported (.cube or the .zip as downloaded) they are recognised by name and used instead of the
# built-in conversion curve for that log profile. DJI serves its LUTs as plain files (dji.com/lut), so
# those can also be downloaded from DJI's servers straight into the library (DJI_LUTS).
#
# Some log profiles have no published curve at all (DJI D-Log M and D-Log2): they are "LUT-only"
# profiles that convert with their official LUT, and fall back to the closest built-in curve (`base`)
# until it is imported.

@dataclass(frozen=True)
class OfficialLut:
    profile: str                  # log profile the LUT converts (built-in curve or LUT-only profile)
    maker: str
    label: str
    url: str                      # official download page ("" = see `hint`)
    patterns: tuple[str, ...]     # regexes on the lower-case file name, best match first
    hint: str = ""
    exclude: str = ""             # regex: names a pattern matches that belong to another entry
    base: str = ""                # LUT-only profile: the built-in curve used until its LUT is imported
    group: str = ""               # menu position (a built-in curve) when `base` is a hidden fallback curve

    @property
    def menu_label(self) -> str:
        return f"{self.maker} {self.label}" + (" (official LUT)" if self.base else "")


_S = r"[ _.+\-]*"   # separators file names put between words: "D-Log M", "DLog_M", "D-Log+M"...
# "D-Log M", not "DLog_Mavic3" / "DLog_Mini4"
_DLOGM = rf"d{_S}log{_S}m(?!avic|ini)"
# "D-Log2", not "DLOG2Rec709" / "DLOG2sRGB" (older files use "2" for "to")
_DLOG2 = rf"d{_S}log{_S}2(?!{_S}(rec|srgb|709|\d))"
# plain "D-Log" / "Dlog" (neither of the above)
_DLOG = rf"d{_S}log(?!{_S}m(?!avic|ini)|{_S}2(?!{_S}(rec|srgb|709|\d)))"
_DJI_HINT = ("D-Log M: Mini 4/5 Pro, Air 3/3S, Avata 2/360, Flip, Osmo Pocket 3, Action 4-6, Osmo 360, Nano. "
             "D-Log: Mavic 3/4 Pro, Osmo Pocket 4/4P, Zenmuse, Phantom 3/4, Inspire 1. D-Log2: Osmo Pocket 4P.")

OFFICIAL_LUTS = [
    OfficialLut("dlog", "DJI", "D-Log → Rec.709", "https://www.dji.com/lut",
                (rf"{_DLOG}.*709", rf"{_DLOG}.*srgb"), _DJI_HINT, r"vivid|hlg|2020"),
    OfficialLut("dlog_vivid", "DJI", "D-Log → Rec.709 Vivid", "https://www.dji.com/lut",
                (rf"{_DLOG}.*709.*vivid", rf"{_DLOG}.*vivid.*709"), _DJI_HINT, base="dlog"),
    OfficialLut("dlogm", "DJI", "D-Log M → Rec.709", "https://www.dji.com/lut",
                (rf"{_DLOGM}.*709",), _DJI_HINT + " D-Log M has no published curve: without its LUT, KangEdit "
                "approximates it with a gentler D-Log-style curve.", base="dlogm_approx", group="dlog"),
    OfficialLut("dlog2", "DJI", "D-Log2 → Rec.709", "https://www.dji.com/lut",
                (rf"{_DLOG2}.*709",), _DJI_HINT, r"vivid", base="dlog"),
    OfficialLut("dlog2_vivid", "DJI", "D-Log2 → Rec.709 Vivid", "https://www.dji.com/lut",
                (rf"{_DLOG2}.*709.*vivid", rf"{_DLOG2}.*vivid.*709"), _DJI_HINT, base="dlog"),
    OfficialLut("slog3_sg3c", "Sony", "S-Log3 / S-Gamut3.Cine → Rec.709",
                "https://pro.sony/bp_BR/technology/professional-video-lut-look-up-table",
                (r"slog3sgamut3\.?cine.*cine\+?709", r"slog3sgamut3\.?cine.*709", r"s-?log3.*gamut3\.?cine.*709")),
    OfficialLut("slog3_sg3", "Sony", "S-Log3 / S-Gamut3 → Rec.709",
                "https://pro.sony/bp_BR/technology/professional-video-lut-look-up-table",
                (r"slog3sgamut3to.*709", r"s-?log3.*gamut3(?!\.?cine).*709")),
    OfficialLut("slog2", "Sony", "S-Log2 → Rec.709",
                "https://pro.sony/bp_BR/technology/professional-video-lut-look-up-table", (r"s-?log2.*709",)),
    OfficialLut("vlog", "Panasonic", "V-Log / V-Log L → V-709",
                "https://av.jpn.support.panasonic.com/support/global/cs/dsc/download/index3.html",
                (r"v-?log.*v-?709", r"v-?log.*709")),
    OfficialLut("clog3", "Canon", "Canon Log 3 → BT.709", "",
                (r"canonlog3.*709", r"c-?log ?3.*709"),
                "Your camera's page on Canon's support site → Downloads → \"Canon lookup table\"."),
    OfficialLut("clog2", "Canon", "Canon Log 2 → BT.709", "",
                (r"canonlog2.*709", r"c-?log ?2.*709"),
                "Your camera's page on Canon's support site → Downloads → \"Canon lookup table\"."),
    OfficialLut("logc3", "ARRI", "LogC3 → Rec.709", "https://www.arri.com/log-c",
                (r"logc3.*709", r"logc2video.*709", r"logc(?!4).*709")),
    OfficialLut("logc4", "ARRI", "LogC4 → Rec.709", "https://www.arri.com/log-c", (r"logc4.*709",)),
    OfficialLut("flog", "Fujifilm", "F-Log → BT.709", "https://www.fujifilm-x.com/en-gb/support/download/lut/",
                (r"f-?log_?f-?gamut_to_.*709", r"f-?log(?!2).*709")),
    OfficialLut("flog2", "Fujifilm", "F-Log2 → BT.709", "https://www.fujifilm-x.com/en-gb/support/download/lut/",
                (r"f-?log2_?f-?gamut_to_.*709", r"f-?log2(?!c).*709")),
    OfficialLut("nlog", "Nikon", "N-Log → Rec.709", "https://downloadcenter.nikonimglib.com/en/download/sw/258.html",
                (r"^n-?log_bt2020_to_rec709", r"(^|[^a-z])n-?log.*709"),
                "Nikon ships the LUTs in an installer: run it, then import the .cube files it unpacks."),
    OfficialLut("bmdfilm5", "Blackmagic", "Film Gen5 → Rec.709", "",
                (r"(blackmagic|bmd).*gen ?5.*709",),
                "Installed with DaVinci Resolve (use \"Import from DaVinci Resolve\")."),
]

# Where DaVinci Resolve keeps the LUTs it ships (several makers' official ones among them).
RESOLVE_LUT_DIRS = [
    os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), "Blackmagic Design", "DaVinci Resolve", "Support", "LUT"),
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT",
    os.path.expanduser("~/.local/share/DaVinciResolve/LUT"),
]


# Log profiles without a built-in curve, converted only by their official LUT.
LUT_ONLY = {e.profile: e for e in OFFICIAL_LUTS if e.base}


@dataclass(frozen=True)
class DjiLut:
    """One DJI camera's conversion LUT as listed on dji.com/lut."""
    product: str
    profile: str                  # OFFICIAL_LUTS profile it converts
    page: str                     # download page slug on dji.com
    file: str = ""                # direct .cube / .zip on DJI's servers ("" = page only)

    @property
    def page_url(self) -> str:
        return f"https://www.dji.com/downloads/softwares/{self.page}"


_T = "https://terra-1-g.djicdn.com/851d20f7b9f64838a34cd02351370894/"
_DL = "https://dl.djicdn.com/downloads/"
# Every log -> Rec.709 LUT on https://www.dji.com/lut (D-Log -> Rec.2020 HLG, Linear -> D-Log and the
# transcoding tools are left out: they don't convert to Rec.709). Older or duplicate versions of a file
# are replaced by DJI's newest (V2.0, 65-point where offered).
DJI_LUTS = [
    DjiLut("Mavic 4 Pro", "dlogm", "mavic-4-pro-dlog-m-to-rec709-lut",
           _T + "WA341%20LUT/DJI%20Mavic%204%20Pro%20D-Log%20M%20to%20Rec.709%20V1.cube"),
    DjiLut("Mavic 3 series", "dlogm", "dji-mavic-3-d-log-m-709-lut",
           _T + "M3/DJI%20Mavic%203%20D-Log%20M%20to%20Rec.709%20V1.cube"),
    DjiLut("Mini 5 Pro", "dlogm", "mini-5-pro-dlog-m-to-rec709-lut",
           _T + "WA150%20LUT/DJI%20Mini%205%20Pro%20D-Log%20M%20to%20Rec.709%20LUT.cube"),
    DjiLut("Mini 4 Pro", "dlogm", "mini-4-pro-dlog-to-rec709",
           _T + "140%20lut/DJI%20Mini%204%20Pro%20D-Log%20M%20to%20Rec.709%20V1_.cube"),
    DjiLut("Air 3S", "dlogm", "air-3s-dlog-to-rec709",
           _T + "234_lut/DJI%20Air%203S%20%20D-Log%20M%20to%20Rec.709%20V1_.cube"),
    DjiLut("Air 3", "dlogm", "air-3-dlog-to-rec709",
           _T + "DJI%20Air%203%20Lut/DJI%20Air%203%20D-Log%20M%20to%20Rec.709%20V1_.cube"),
    DjiLut("Avata 2", "dlogm", "avata-2-dlog-to-rec709",
           _T + "avata2%20d-log/DJI%20Avata%202%20D-Log%20M%20to%20Rec.709%20V1._.cube"),
    DjiLut("Avata 360", "dlogm", "avata-360-dlog-m-to-rec709-lut",
           _T + "WA530%20Lut/DJI%20Avata%20360%20D-Log%20M%20to%20Rec.709%20V1.cube"),
    DjiLut("Flip", "dlogm", "flip-dlog-to-rec709",
           _T + "141%20LUT/DJI%20Flip%20D-Log%20M%20to%20Rec.709%20V1_.cube"),
    DjiLut("Lito X1", "dlogm", "lito-x1-dlog-m-to-rec709-lut",
           _T + "WA151%20Lut/DJI%20Lito%20X1%20D-Log%20M%20to%20Rec.709%20LUT.cube"),
    DjiLut("O4 Air Unit series", "dlogm", "o4-air-unit-dlog-to-rec709",
           _T + "ZA530%20LUT/DJI%20O4%20Air%20Unit%20Series%20D-Log%20M%20to%20Rec.709%20V1_.cube"),
    DjiLut("Osmo Pocket 3", "dlogm", "osmo-pocket-3-dlog-to-rec709",
           "https://www-dl.djicdn.com/5e45168b46b342d5b88f72c458ba6e79/OP3%20LUT%E6%96%87%E4%BB%B6%E7%89%B9%E6%AE%8A"
           "%E5%A4%84%E7%90%86/DJI%20OSMO%20Pocket%203%20D-Log%20M%20to%20Rec.709%20V1.cube"),
    DjiLut("Osmo Action 6", "dlogm", "dji-osmo-action-6-d-log-m-to-rec-709-lut",
           _T + "AC206%20LUT/DJI%20OSMO%20Action%206%20D-LogM%20to%20Rec.709%20LUT-11.17.cube"),
    DjiLut("Osmo Action 5 Pro (vivid)", "dlogm", "dji-osmo-action-5-pro-d-log-m-to-rec-709-vivid-lut",
           _T + "AC204%E8%BD%AF%E4%BB%B6/DJI%20OSMO%20Action%205%20Pro%20D-Log%20M%20to%20Rec.709%20V1.cube"),
    DjiLut("Osmo Action 4 (vivid)", "dlogm", "dji-osmo-action-4-d-log-m-to-rec-709-vivid-lut",
           _T + "203/DJI%20OSMO%20Action%204%20D-Log%20M%20to%20Rec.709%20V1.cube"),
    DjiLut("Osmo Nano (vivid)", "dlogm", "osmo-nano-d-log-m-to-rec-709-vivid-lut",
           _T + "OW001%20LUT/DJI%20OSMO%20Osmo%20Nano%20D-Log%20M%20to%20Rec.709%20V1.cube"),
    DjiLut("Osmo 360", "dlogm", "osmo-360-dlog-m-to-rec709-lut",
           _T + "OQ101%20LUT/DJI%20Osmo%20360%20D-Log%20M%20to%20Rec.709%20V1.cube"),
    DjiLut("Mavic 4 Pro", "dlog", "mavic-4-pro-dlog-to-rec709-lut",
           _T + "WA341%20LUT/DJI%20Mavic%204%20Pro%20D-Log%20to%20Rec.709%20V1.cube"),
    DjiLut("Mavic 3 series", "dlog", "transcoding-mavic-3"),
    DjiLut("Osmo Pocket 4", "dlog", "osmo-pocket-4-dlog-to-rec709-lut",
           _T + "HG214%20Lut/DJI%20OSMO%20Pocket%204%20D-Log%20to%20Rec.709%20V2.0.cube"),
    DjiLut("Osmo Pocket 4P", "dlog", "osmo-pocket-4p-dlog-to-rec709-lut-size33-2",
           _T + "HG224%20LUT/DJI%20OSMO%20Pocket%204P%20D-Log%20to%20Rec.709%20V2.0%20size33.cube"),
    DjiLut("Zenmuse X9", "dlog", "dji-zenmuse-x9-dlog2rec709-lut"),
    DjiLut("Zenmuse X5S / X7", "dlog", "transcoding-zenmuse-x7-rec", _DL + "zenmuse+x7/20171017/D-Log+to+Rec709.zip"),
    DjiLut("Phantom 4", "dlog", "transcoding-phantom-4", _DL + "phantom_4/Phantom_4_Dlog_3DLUT_20160606.zip"),
    DjiLut("Phantom 3 (to sRGB)", "dlog", "transcoding-phantom-3", _DL + "phantom_3/Phantom_3_Dlog_3DLUT.zip"),
    DjiLut("Inspire 1 / Zenmuse X5 (to sRGB)", "dlog", "transcoding-zenmuse-x5r",
           _DL + "zenmuse_x5s/X5_series_Dlog_3DLUT.zip"),
    DjiLut("Mavic 4 Pro", "dlog_vivid", "mavic-4-pro-dlog-to-rec709-vivid-lut",
           _T + "WA341%20LUT/DJI%20Mavic%204%20Pro%20D-Log%20to%20Rec.709%20vivid%20V1.cube"),
    DjiLut("Mavic 3 series", "dlog_vivid", "vivid-lut",
           _T + "mavic%203/DJI%20Mavic%203%20D-Log%20to%20Rec.709%20vivid%20V1.cube"),
    DjiLut("Osmo Pocket 4", "dlog_vivid", "osmo-pocket-4-dlog-to-rec709-lut-vivid",
           _T + "HG214%20Lut/DJI%20OSMO%20Pocket%204%20D-Log%20to%20Rec.709%20vivid%20V2.0.cube"),
    DjiLut("Osmo Pocket 4P", "dlog_vivid", "osmo-pocket-4p-dlog-to-rec709-lut-vivid-size33-2",
           _T + "HG224%20LUT/DJI%20OSMO%20Pocket%204P%20D-Log%20to%20Rec.709%20vivid%20V2.0%20size33.cube"),
    DjiLut("Osmo Pocket 4P", "dlog2", "osmo-pocket-4p-dlog-2-to-rec709-lut-size65",
           _T + "HG224%20LUT/DJI%20OSMO%20Pocket%204P%20D-Log2%20to%20Rec.709%20V1.0%20size65.cube"),
    DjiLut("Osmo Pocket 4P", "dlog2_vivid", "osmo-pocket-4p-dlog-2-to-rec709-lut-vivid-size65",
           _T + "HG224%20LUT/DJI%20OSMO%20Pocket%204P%20D-Log2%20to%20Rec.709%20vivid%20V1.0%20size65.cube"),
]


def official_matches(entry: OfficialLut, luts: list[tuple[str, str]]) -> list[str]:
    """Library LUT paths that look like `entry`'s official file, best first."""
    out: list[str] = []
    skip = re.compile(entry.exclude) if entry.exclude else None
    for pat in entry.patterns:
        rx = re.compile(pat)
        out += [path for name, path in luts
                if rx.search(name.lower()) and not (skip and skip.search(name.lower())) and path not in out]
    return out


# Official LUT chosen for each profile in the LUT dialog ({profile: path}); set by the UI.
_choices: dict[str, str] = {}
_resolved: dict[str, Optional[str]] = {}
_resolved_stamp: tuple = ()


def set_official_choices(mapping: dict[str, str]) -> None:
    global _choices
    with _cache_lock:
        _choices = dict(mapping)
        _resolved.clear()


def official_lut_path(profile: str) -> Optional[str]:
    """The LUT that converts `profile`: the one chosen in the LUT dialog, else the best-matching library
    file. None when there is none. Cheap enough to call per frame (re-resolved when the library changes)."""
    global _resolved_stamp
    try:
        stamp = (os.stat(_library_dir()).st_mtime_ns,)
    except OSError:
        stamp = ()
    with _cache_lock:
        if stamp != _resolved_stamp:
            _resolved.clear()
            _resolved_stamp = stamp
        if profile in _resolved:
            return _resolved[profile]
        chosen = _choices.get(profile)
    path = chosen if chosen and os.path.exists(chosen) else None
    if path is None:
        entry = next((e for e in OFFICIAL_LUTS if e.profile == profile), None)
        matches = official_matches(entry, library()) if entry else []
        path = matches[0] if matches else None
    with _cache_lock:
        _resolved[profile] = path
    return path


_lib_dir: Optional[str] = None


def _library_dir() -> str:
    global _lib_dir
    if _lib_dir is None:
        _lib_dir = str(lut_dir())
    return _lib_dir


def download_official(url: str, timeout: float = 60.0) -> list[str]:
    """Download an official LUT (.cube, or a .zip of them) from the maker's server into the library."""
    name = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])
    base, ext = os.path.splitext(name)
    base = re.sub(r"[._ ]+$", "", base).replace("  ", " ") or "LUT"   # "…V1._.cube" -> "…V1.cube"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (KangEdit video editor)"})
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, re.sub(r'[<>:"/\\|?*]', "_", base) + ext.lower())
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dst, "wb") as fo:
            shutil.copyfileobj(resp, fo)
        return import_any(dst)


def import_any(src: str) -> list[str]:
    """Import a .cube file, every .cube inside a .zip, or every .cube under a folder. Returns library paths."""
    if os.path.isdir(src):
        found = [os.path.join(root, n) for root, _, names in os.walk(src) for n in names
                 if n.lower().endswith(LUT_EXTENSIONS)]
        return [import_lut(p) for p in sorted(found)]
    if src.lower().endswith(".zip"):
        out = []
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(src) as zf:
            for info in zf.infolist():
                name = os.path.basename(info.filename)
                if info.is_dir() or not name.lower().endswith(LUT_EXTENSIONS) or name.startswith("._"):
                    continue
                dst = os.path.join(tmp, name)
                with zf.open(info) as fi, open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                try:
                    out.append(import_lut(dst))
                except LutError:
                    pass   # e.g. a 1D curve or a broken file next to the real LUTs
        if not out:
            raise LutError("no usable .cube LUT inside the .zip file")
        return out
    return [import_lut(src)]
