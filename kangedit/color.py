"""Color science: camera log -> Rec.709 conversion, basic grading and auto-correction.

Every color operation is a pure per-pixel function of RGB, so the whole grade is baked
into a single 3D LUT. The viewer applies a 96^3 LUT to 8-bit frames and the exporter
writes a 33^3 .cube file for ffmpeg's lut3d filter -- both come from `apply_grade`,
which keeps preview and export identical. Imported .cube LUTs (log conversion or creative look,
see lut.py) are evaluated inside `apply_grade` too, so they get baked in the same way.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np

from . import lut as lutlib
from .lut import LUT_PREFIX

# ----------------------------------------------------------------------------- gamuts

_D65 = (0.3127, 0.3290)

_PRIMARIES = {
    "rec709": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
    "sgamut3cine": ((0.766, 0.275), (0.225, 0.800), (0.089, -0.087)),
    "sgamut3": ((0.730, 0.280), (0.140, 0.855), (0.100, -0.050)),
    "vgamut": ((0.730, 0.280), (0.165, 0.840), (0.100, -0.030)),
    "cinemagamut": ((0.740, 0.270), (0.170, 1.140), (0.080, -0.100)),
    "awg3": ((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020)),
    "awg4": ((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308)),
    "dgamut": ((0.710, 0.310), (0.210, 0.880), (0.090, -0.080)),
    "bmdwg5": ((0.7177, 0.3171), (0.2280, 0.8616), (0.1006, -0.0820)),
}


def _rgb_to_xyz(prim, white=_D65) -> np.ndarray:
    def xyz(xy):
        x, y = xy
        return np.array([x / y, 1.0, (1 - x - y) / y])

    m = np.stack([xyz(p) for p in prim], axis=1)
    s = np.linalg.solve(m, xyz(white))
    return m * s


def gamut_to_709(name: str) -> np.ndarray:
    if name == "rec709":
        return np.eye(3)
    src = _rgb_to_xyz(_PRIMARIES[name])
    dst = _rgb_to_xyz(_PRIMARIES["rec709"])
    return np.linalg.solve(dst, src)


# ----------------------------------------------------------------------------- transfer curves
# Each decoder maps a normalized signal to scene-linear (0.18 = mid grey).
# `domain` tells which signal the published formula expects:
#   "cv"  -> 10-bit code value / 1023 (legal-range encoded file)
#   "ire" -> legal range expanded to 0..1 (what ffmpeg/PyAV RGB conversion produces)


def _slog3(x):
    return np.where(
        x >= 171.2102946929 / 1023.0,
        (10.0 ** ((x * 1023.0 - 420.0) / 261.5)) * (0.18 + 0.01) - 0.01,
        (x * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0),
    )


def _slog2(x):  # x in IRE domain
    lin = np.where(
        x >= 0.030001222851889303,
        10.0 ** ((x - 0.616596 - 0.03) / 0.432699) - 0.037584,
        (x - 0.030001222851889303) / 5.0,
    )
    return lin * 0.9 * 219.0 / 155.0


def _vlog(x):
    return np.where(x < 0.181, (x - 0.125) / 5.6, 10.0 ** ((x - 0.598206) / 0.241514) - 0.00873)


def _clog3(x):  # IRE domain
    lin = np.where(
        x < 0.097465473,
        -(10.0 ** ((0.12783901 - x) / 0.36726845) - 1.0) / 14.98325,
        np.where(
            x <= 0.15277891,
            (x - 0.12512219) / 1.9754798,
            (10.0 ** ((x - 0.12240537) / 0.36726845) - 1.0) / 14.98325,
        ),
    )
    return lin * 0.9


def _clog2(x):  # IRE domain
    lin = np.where(
        x < 0.092864125,
        -(10.0 ** ((0.092864125 - x) / 0.24136077) - 1.0) / 87.099375,
        (10.0 ** ((x - 0.092864125) / 0.24136077) - 1.0) / 87.099375,
    )
    return lin * 0.9


def _logc3(x):
    cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    return np.where(x > e * cut + f, (10.0 ** ((x - d) / c) - b) / a, (x - f) / e)


def _logc4(x):
    a = (2.0 ** 18 - 16.0) / 117.45
    b = (1023.0 - 95.0) / 1023.0
    c = 95.0 / 1023.0
    s = (7.0 * np.log(2.0) * 2.0 ** (7.0 - 14.0 * c / b)) / (a * b)
    t = (2.0 ** (14.0 * (-c / b) + 6.0) - 64.0) / a
    return np.where(x >= 0.0, (2.0 ** (14.0 * (x - c) / b + 6.0) - 64.0) / a, x * s + t)


def _flog(x):
    a, b, c, d, e, f = 0.555556, 0.009468, 0.344676, 0.790453, 8.735631, 0.092864
    return np.where(x >= 0.100537775223865, (10.0 ** ((x - d) / c)) / a - b / a, (x - f) / e)


def _flog2(x):
    a, b, c, d, e, f = 5.555556, 0.064829, 0.245281, 0.384316, 8.799461, 0.092864
    return np.where(x >= 0.100686685370811, (10.0 ** ((x - d) / c)) / a - b / a, (x - f) / e)


def _nlog(x):
    cv = x * 1023.0
    return np.where(cv < 452.0, (np.maximum(cv, 0.0) / 650.0) ** 3 - 0.0075, np.exp((cv - 619.0) / 150.0))


def _dlog(x):
    return np.where(x <= 0.14, (x - 0.0929) / 6.025, (10.0 ** (3.89616 * x - 2.27752) - 0.0108) / 0.9892)


def _dlogm_approx(x):
    """DJI D-Log M has no published curve. It is a milder log than D-Log (more contrast and colour in the
    signal), so decoding it as D-Log overshoots both; this gentler pure-log fit (mid grey near 42 %, ten
    stops per 1.0) is used until the official LUT is imported."""
    mid, per_stop, toe = 0.42, 0.10, 0.10
    return np.where(x > toe, 0.18 * 2.0 ** ((x - mid) / per_stop), 0.18 * 2.0 ** ((toe - mid) / per_stop) * x / toe)


def _applelog(x):  # IRE domain
    r0, rt, c, b, g, d = -0.05641088, 0.01, 47.28711236, 0.00964052, 0.08550479, 0.69336945
    pt = c * (rt - r0) ** 2
    return np.where(x >= pt, 2.0 ** ((x - d) / g) - b, np.where(x >= 0, np.sqrt(np.maximum(x, 0) / c) + r0, r0))


def _bmdfilm5(x):  # IRE domain
    A, B, C, D, E = 0.08692876065491224, 0.005494072432257808, 0.5300133392291939, 8.283605932402494, 0.09246575342465753
    log_cut = D * 0.005 + E
    return np.where(x < log_cut, (x - E) / D, np.exp((x - C) / A) - B)


def _hlg(x):  # IRE domain; scaled so HLG mid grey lands near 0.18
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    lin = np.where(x <= 0.5, (np.maximum(x, 0) ** 2) / 3.0, (np.exp((x - c) / a) + b) / 12.0)
    return lin * 3.75


@dataclass(frozen=True)
class LogProfile:
    label: str
    decode: Callable[[np.ndarray], np.ndarray]
    domain: str
    gamut: str


LOG_PROFILES: dict[str, LogProfile] = {
    "slog3_sg3c": LogProfile("Sony S-Log3 / S-Gamut3.Cine", _slog3, "cv", "sgamut3cine"),
    "slog3_sg3": LogProfile("Sony S-Log3 / S-Gamut3", _slog3, "cv", "sgamut3"),
    "slog2": LogProfile("Sony S-Log2 / S-Gamut", _slog2, "ire", "sgamut3"),
    "vlog": LogProfile("Panasonic V-Log / V-Gamut", _vlog, "cv", "vgamut"),
    "clog3": LogProfile("Canon C-Log3 / Cinema Gamut", _clog3, "ire", "cinemagamut"),
    "clog2": LogProfile("Canon C-Log2 / Cinema Gamut", _clog2, "ire", "cinemagamut"),
    "logc3": LogProfile("ARRI LogC3 (EI 800) / AWG3", _logc3, "cv", "awg3"),
    "logc4": LogProfile("ARRI LogC4 / AWG4", _logc4, "cv", "awg4"),
    "flog": LogProfile("Fujifilm F-Log / F-Gamut", _flog, "cv", "rec2020"),
    "flog2": LogProfile("Fujifilm F-Log2 / F-Gamut", _flog2, "cv", "rec2020"),
    "nlog": LogProfile("Nikon N-Log / Rec.2020", _nlog, "cv", "rec2020"),
    "dlog": LogProfile("DJI D-Log / D-Gamut", _dlog, "cv", "dgamut"),
    "applelog": LogProfile("Apple Log / Rec.2020", _applelog, "ire", "rec2020"),
    "bmdfilm5": LogProfile("Blackmagic Film Gen5 / BMD WG", _bmdfilm5, "ire", "bmdwg5"),
    "hlg": LogProfile("HLG HDR (phones) / Rec.2020", _hlg, "ire", "rec2020"),
    # Fallbacks for LUT-only profiles (lut.OFFICIAL_LUTS base), not offered in the menu themselves.
    "dlogm_approx": LogProfile("DJI D-Log M (approximation)", _dlogm_approx, "cv", "rec2020"),
}
_HIDDEN_PROFILES = {"dlogm_approx"}

# Built-in curves, each followed by the LUT-only profiles of the same family (DJI D-Log M, D-Log2...).
LOG_CHOICES = [("none", "None (Rec.709 / standard)")]
for _k, _p in LOG_PROFILES.items():
    if _k in _HIDDEN_PROFILES:
        continue
    LOG_CHOICES.append((_k, _p.label))
    LOG_CHOICES += [(e.profile, e.menu_label) for e in lutlib.OFFICIAL_LUTS
                    if e.base == _k or (e.base in _HIDDEN_PROFILES and e.group == _k)]


def is_lut_profile(log_profile: str) -> bool:
    return log_profile.startswith(LUT_PREFIX)


def lut_profile(path: str) -> str:
    """Log-profile key that converts with an imported LUT instead of a built-in camera curve."""
    return LUT_PREFIX + path


def effective_profile(log_profile: str) -> str:
    """What converts `log_profile`: a LUT-only profile (DJI D-Log M...) becomes "lut:<its official LUT>", or
    the built-in curve that approximates it while that LUT isn't imported. Other profiles are unchanged."""
    entry = lutlib.LUT_ONLY.get(log_profile)
    if entry is None:
        return log_profile
    path = lutlib.official_lut_path(log_profile)
    return lut_profile(path) if path else entry.base


def missing_lut(log_profile: str) -> bool:
    """A LUT-only profile whose official LUT isn't in the library (shown approximated)."""
    return log_profile in lutlib.LUT_ONLY and lutlib.official_lut_path(log_profile) is None


def profile_label(log_profile: str) -> str:
    if is_lut_profile(log_profile):
        return "LUT: " + lutlib.lut_name(log_profile[len(LUT_PREFIX):])
    label = dict(LOG_CHOICES).get(log_profile, log_profile)
    return label + " — LUT not imported, approximated" if missing_lut(log_profile) else label


def profile_short(log_profile: str) -> str:
    """Compact label for the viewer's status line."""
    if log_profile == "none":
        return ""
    if is_lut_profile(log_profile):
        return "LUT " + lutlib.lut_name(log_profile[len(LUT_PREFIX):])
    entry = lutlib.LUT_ONLY.get(log_profile)
    if entry is not None:
        return entry.label.split(" →")[0].upper() + (" (APPROX.)" if missing_lut(log_profile) else "")
    return log_profile.upper()

TONEMAPS = [("filmic", "Filmic (soft highlight roll-off)"), ("standard", "Standard Rec.709 (hard clip)")]


# ----------------------------------------------------------------------------- grade


@dataclass
class Grade:
    exposure: float = 0.0      # stops, -4..+4
    contrast: float = 0.0      # -100..100
    highlights: float = 0.0    # -100..100
    shadows: float = 0.0       # -100..100
    temperature: float = 0.0   # -100 (cool) .. 100 (warm)
    tint: float = 0.0          # -100 (green) .. 100 (magenta)
    saturation: float = 100.0  # 0..200 %
    vibrance: float = 0.0      # -100..100
    look: str = ""             # creative .cube LUT applied after the grade ("" = none)
    look_mix: float = 100.0    # look strength 0..100 %
    sharpness: float = 0.0     # -100 (soften) .. 100 (sharpen); spatial, not part of the LUT
    halation: float = 0.0      # 0..100: film's warm glow around highlights; spatial, not part of the LUT

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Grade":
        g = Grade()
        for k, v in (d or {}).items():
            if hasattr(g, k):
                cur = getattr(g, k)
                setattr(g, k, str(v or "") if isinstance(cur, str) else float(v))
        return g

    def is_identity(self) -> bool:
        return self == Grade()

    def key(self) -> tuple:
        """The color part of the grade (what a LUT holds): sharpness and halation are left out."""
        return tuple(v if isinstance(v, str) else round(v, 4) for k, v in asdict(self).items()
                     if k not in SPATIAL_PARAMS)


SPATIAL_PARAMS = ("sharpness", "halation")


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def wb_gains(temperature: float, tint: float) -> np.ndarray:
    t = temperature / 100.0
    n = tint / 100.0
    g = 2.0 ** np.array([0.5 * t, -0.4 * n, -0.5 * t])
    return (g / float(np.dot(_LUMA, g))).astype(np.float32)


def _aces_fit(x):
    x = np.maximum(x, 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def _signal(rgb: np.ndarray, domain: str, full_range: bool) -> np.ndarray:
    """The decoded picture (0..1 over the file's range: legal 64..940 or full 0..1023 code values) -> the
    domain a log curve is written in: 10-bit code values / 1023 ("cv") or legal-range IRE ("ire").
    The camera's curve is the same in either kind of file; only how the decoder scaled it differs."""
    if domain == "cv":
        return rgb if full_range else (rgb * 876.0 + 64.0) / 1023.0
    return (rgb * 1023.0 - 64.0) / 876.0 if full_range else rgb


_DENSE = np.linspace(0.0, 1.0, 65536)
_dense_tables: dict[tuple, np.ndarray] = {}


def _dense_table(profile: str, full_range: bool) -> np.ndarray:
    key = (profile, full_range)
    if key not in _dense_tables:
        p = LOG_PROFILES[profile]
        _dense_tables[key] = np.asarray(p.decode(_signal(_DENSE, p.domain, full_range)), dtype=np.float64)
    return _dense_tables[key]


def _to_linear(rgb: np.ndarray, log_profile: str, full_range: bool = False) -> np.ndarray:
    """Signal (0..1 as decoded) -> scene/display linear Rec.709."""
    if log_profile in LOG_PROFILES:
        p = LOG_PROFILES[log_profile]
        if rgb.size > _DENSE.size:      # (a LUT's grid) the curve through a dense table: same values, far faster
            table = _dense_table(log_profile, full_range)
            pos = np.clip(rgb, 0.0, 1.0).astype(np.float32) * np.float32(_DENSE.size - 1)
            i = np.minimum(pos.astype(np.int32), _DENSE.size - 2)
            lin = table[i] + (table[i + 1] - table[i]) * (pos - i)
        else:
            lin = p.decode(_signal(rgb, p.domain, full_range).astype(np.float64)).astype(np.float32)
        m = gamut_to_709(p.gamut).astype(np.float32)
        return lin @ m.T
    if is_lut_profile(log_profile):
        # A conversion LUT outputs a Rec.709 display signal; grade on from there like standard footage.
        conv = lutlib.try_load(log_profile[len(LUT_PREFIX):])
        if conv is not None:
            rgb = conv.apply(rgb)
    return np.power(np.clip(rgb, 0.0, 1.0), 2.4)


WHITE_ROLLOFF = 0.75    # tone-mapped level where bright colours start to fade to white


def _filmic(lin: np.ndarray) -> np.ndarray:
    """Scene-linear Rec.709 -> display-linear 0..1 with the filmic curve (ACES fit), hue preserving: the curve
    maps a colour's brightness (a smooth norm close to its brightest channel) and its channels keep their
    ratios, so a bright red stays red instead of turning orange and skies don't go cyan (per-channel curves
    shift hue by up to 40 degrees there). Near display white colours fade to white, as film and the eye do,
    rather than clipping as flat patches. Neutral tones (middle grey, the grey scale) come out exactly as the
    per-channel curve gives them. The norm, sum(c^3) / sum(c^2), is smooth (max(r, g, b) has a crease along
    the grey axis that a LUT can't interpolate across cleanly)."""
    lin = np.asarray(lin, dtype=np.float32)
    pos = np.maximum(lin, 0.0)
    sq = pos * pos
    peak = np.sum(sq * pos, axis=-1, keepdims=True) / np.maximum(np.sum(sq, axis=-1, keepdims=True), 1e-20)
    mapped = _aces_fit(peak * 0.6)
    out = np.where(peak > 1e-9, lin * (mapped / np.maximum(peak, 1e-9)), 0.0)
    fade = np.clip((mapped - WHITE_ROLLOFF) / (1.0 - WHITE_ROLLOFF), 0.0, 1.0) ** 2
    out = out + (mapped - out) * fade
    return np.clip(out, 0.0, 1.0).astype(np.float32)


GAMUT_THRESHOLD, GAMUT_LIMIT = 0.9, 1.2


def _compress_gamut(lin: np.ndarray) -> np.ndarray:
    """Colours the camera saw outside Rec.709 (negative channels after the gamut conversion: neon, deep blue
    lights, saturated flowers) pulled smoothly into it, as the ACES reference gamut compression does: each
    channel's distance from the brightest one is compressed past GAMUT_THRESHOLD so that GAMUT_LIMIT lands on
    the gamut edge. Colours well inside are untouched; instead of clipping to flat patches, the most saturated
    ones keep their gradations (and a LUT of the transform stays smooth)."""
    thr, lim, p = GAMUT_THRESHOLD, GAMUT_LIMIT, 1.2
    lin = np.asarray(lin, dtype=np.float32)
    ach = np.max(lin, axis=-1, keepdims=True)
    dist = (ach - lin) / np.maximum(ach, np.float32(1e-9))
    far = (dist > thr) & (ach > 1e-9)
    if not far.any():
        return lin
    scale = (lim - thr) / (((1.0 - thr) / (lim - thr)) ** -p - 1.0) ** (1.0 / p)
    x = (dist[far] - thr) / np.float32(scale)
    dist[far] = thr + np.float32(scale) * x / (1.0 + x ** np.float32(p)) ** np.float32(1.0 / p)
    out = lin.copy()
    rows = np.broadcast_to(ach, lin.shape)[far]
    out[far] = rows - dist[far] * rows
    return out


def _to_display(lin: np.ndarray, log_profile: str, tonemap: str) -> np.ndarray:
    if log_profile in LOG_PROFILES:
        lin = _compress_gamut(lin)
    if log_profile in LOG_PROFILES and tonemap == "filmic":
        disp_lin = _filmic(lin)
    else:
        disp_lin = np.clip(lin, 0.0, 1.0)
    return np.power(disp_lin, 1.0 / 2.4)


# The grade follows the order colorists use (scene-referred, then display-referred):
#   1. input transform: the camera's log curve and gamut -> scene-linear Rec.709. With a conversion LUT,
#      exposure and white balance are applied before it, on the log signal (as an Offset correction is
#      placed before the LUT node), so the LUT receives a correctly exposed, neutral image;
#   2. primary balance in linear light: exposure (stops) and white balance (channel gains);
#   3. tone: contrast, shadows and highlights in stops around middle grey (18 %), before the output
#      transform, so pushes roll off in the tone map instead of clipping;
#   4. saturation and vibrance in linear light, around Rec.709 luminance;
#   5. output transform: filmic tone map (or a plain clip) to the Rec.709 display signal;
#   6. creative look LUT, which expects a Rec.709 display image, blended by its strength;
#   7. (apply_spatial) halation and sharpening on the finished picture.
# Footage that is already a display image (standard Rec.709, or converted by a LUT) has no scene-linear
# range above 1: its tone controls work on the display signal with a curve that keeps black and white.

MID_GREY = 0.18
LOG_STOP = 0.075     # log signal per stop for an unknown camera curve (S-Log3 0.074, V-Log 0.073, D-Log 0.08)
_GRID = np.linspace(0.0, 1.0, 4096, dtype=np.float64)
_curve_tables: dict[tuple, Optional[np.ndarray]] = {}


def _decode_table(profile: str, full_range: bool = False) -> Optional[np.ndarray]:
    """Linear value of each _GRID signal level for a built-in log curve (camera gamut), or None when it
    doesn't rise steadily (then it can't be inverted)."""
    key = (profile, full_range)
    if key not in _curve_tables:
        p = LOG_PROFILES[profile]
        lin = np.maximum.accumulate(np.asarray(p.decode(_signal(_GRID, p.domain, full_range)), dtype=np.float64))
        _curve_tables[key] = lin if lin[-1] > lin[0] else None
    return _curve_tables[key]


def _balance_log(x: np.ndarray, grade: Grade, base: str, full_range: bool = False) -> np.ndarray:
    """Exposure and white balance on a log signal, before a conversion LUT. With the camera's curve known,
    the signal is decoded, multiplied and encoded again (exact); otherwise each stop is LOG_STOP of signal."""
    if not grade.exposure and not grade.temperature and not grade.tint:
        return x
    gains = (2.0 ** grade.exposure) * wb_gains(grade.temperature, grade.tint).astype(np.float64)
    table = _decode_table(base, full_range) if base in LOG_PROFILES else None
    if table is None:
        return (x + np.float32(LOG_STOP) * np.log2(gains).astype(np.float32)).astype(np.float32)
    out = np.empty_like(x)
    for c in range(3):
        lin = np.interp(x[:, c], _GRID, table) * gains[c]
        out[:, c] = np.interp(lin, table, _GRID)
    return out


def _tone_scene(lin: np.ndarray, grade: Grade) -> np.ndarray:
    """Contrast (pivot: middle grey), shadows and highlights, in stops, on scene-linear RGB."""
    if not (grade.contrast or grade.shadows or grade.highlights):
        return lin
    lin = np.maximum(lin, 0.0)
    stops = np.log2(np.maximum(lin, 1e-7) / MID_GREY)
    if grade.contrast:
        stops = stops * np.float32(2.0 ** (grade.contrast / 100.0))
    if grade.shadows:      # up to ±2 stops, fading out above middle grey
        stops = stops + np.float32(grade.shadows / 50.0) / (1.0 + np.exp((stops + 1.5) * 1.6))
    if grade.highlights:   # up to ±2 stops, fading in from 1.5 stops over middle grey
        stops = stops + np.float32(grade.highlights / 50.0) / (1.0 + np.exp(-(stops - 1.5) * 1.6))
    return np.where(lin > 0, MID_GREY * np.exp2(stops), 0.0).astype(np.float32)


def _tone_display(d: np.ndarray, grade: Grade) -> np.ndarray:
    """Shadows, highlights and contrast on a display signal (0..1), keeping black and white in place."""
    if grade.shadows:
        d = d + (grade.shadows / 100.0 * 1.5) * d * (1.0 - d) ** 2
    if grade.highlights:
        d = d + (grade.highlights / 100.0 * 1.5) * d * d * (1.0 - d)
    if grade.contrast:
        d = np.clip(d, 0.0, 1.0)
        k = 2.0 ** (grade.contrast / 100.0)
        p = MID_GREY ** (1.0 / 2.4)            # middle grey on the display
        d = np.where(d < p, p * (d / p) ** k, 1.0 - (1.0 - p) * ((1.0 - d) / (1.0 - p)) ** k)
    return d


def _saturate(lin: np.ndarray, grade: Grade) -> np.ndarray:
    """Saturation and vibrance in linear light around Rec.709 luminance (hue and brightness kept)."""
    if grade.saturation == 100.0 and not grade.vibrance:
        return lin
    luma = (lin @ _LUMA)[:, None]
    sat = np.float32(grade.saturation / 100.0)
    if grade.vibrance:   # stronger on muted colours than on already saturated ones
        hi = np.max(lin, axis=1, keepdims=True)
        cur = (hi - np.min(lin, axis=1, keepdims=True)) / np.maximum(hi, 1e-6)
        sat = sat * (1.0 + np.float32(grade.vibrance / 100.0) * (1.0 - np.clip(cur * 1.5, 0.0, 1.0)))
    return luma + (lin - luma) * sat


def apply_grade(rgb: np.ndarray, grade: Grade, log_profile: str = "none", tonemap: str = "filmic",
                full_range: bool = False) -> np.ndarray:
    """rgb: (..., 3) float in 0..1 (gamma-encoded signal as decoded). Returns graded display RGB 0..1 (see the
    order above). full_range: the file uses full-range code values (0..1023), which a log curve needs to know."""
    rgb = np.asarray(rgb, dtype=np.float32)
    shape = rgb.shape
    x = rgb.reshape(-1, 3)
    entry = lutlib.LUT_ONLY.get(log_profile)
    base = entry.base if entry is not None else ""
    profile = effective_profile(log_profile)

    if profile in LOG_PROFILES:                                  # scene-referred
        lin = _to_linear(x, profile, full_range)
        lin = lin * np.float32(2.0 ** grade.exposure)
        if grade.temperature or grade.tint:
            lin = lin * wb_gains(grade.temperature, grade.tint)
        lin = _saturate(_tone_scene(lin, grade), grade)
        d = _to_display(lin, profile, tonemap).astype(np.float32)
    else:                                                        # display-referred
        if is_lut_profile(profile):
            lin = _to_linear(_balance_log(x, grade, base, full_range), profile)
        else:
            lin = _to_linear(x, profile) * np.float32(2.0 ** grade.exposure)
            if grade.temperature or grade.tint:
                lin = lin * wb_gains(grade.temperature, grade.tint)
        d = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.4).astype(np.float32)
        d = _tone_display(d, grade)
        if grade.saturation != 100.0 or grade.vibrance:
            lin = _saturate(np.power(np.clip(d, 0.0, 1.0), 2.4), grade)
            d = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.4).astype(np.float32)

    if grade.look and grade.look_mix > 0:
        look = lutlib.try_load(grade.look)
        if look is not None:
            d = np.clip(d, 0.0, 1.0)
            d = d + (look.apply(d) - d) * np.float32(grade.look_mix / 100.0)

    return np.clip(d, 0.0, 1.0).reshape(shape)


def needs_color(grade: Grade, log_profile: str) -> bool:
    return (log_profile in LOG_PROFILES or log_profile in lutlib.LUT_ONLY or is_lut_profile(log_profile)
            or grade.key() != Grade().key())


# ----------------------------------------------------------------------------- sharpness / halation
# Both depend on neighbouring pixels, so they run on the picture after the LUT: in the viewer with OpenCV
# (apply_spatial) and in the export with matching ffmpeg filters (ffmpeg_halation / ffmpeg_sharpen).
# Sizes scale with the frame height so the viewer and every export resolution look alike.

HALATION_THRESHOLD = 0.7                     # highlights brighter than this (luma) glow
HALATION_TINT = (1.0, 0.28, 0.06)            # red-orange, as the red layer of film catches the bounce
HALATION_GAIN = 1.4                          # glow strength at 100


def has_spatial(grade: Grade) -> bool:
    return abs(grade.sharpness) >= 0.5 or grade.halation >= 0.5


def sharpen_size(height: int) -> int:
    """Odd blur size of the unsharp mask (5 px at 1080 p)."""
    n = int(round(5 * height / 1080.0))
    return max(3, min(23, n | 1))


def sharpen_amount(sharpness: float) -> float:
    """Unsharp mask amount: up to 1.5 when sharpening, down to -1 (the blurred picture) when softening."""
    return sharpness / 100.0 * (1.5 if sharpness > 0 else 1.0)


def halation_sigma(height: int) -> float:
    return 0.012 * height


def apply_spatial(img: np.ndarray, grade: Grade) -> np.ndarray:
    """Halation, then sharpness, on an RGB uint8 picture (the viewer's version of the export filters)."""
    import cv2
    h = img.shape[0]
    out = img
    if grade.halation >= 0.5:
        x = out.astype(np.float32) * np.float32(1.0 / 255.0)
        lum = np.clip((x @ _LUMA - HALATION_THRESHOLD) / (1.0 - HALATION_THRESHOLD), 0.0, 1.0)
        # Blur at quarter size: the glow is soft anyway, and this keeps playback fast.
        small = cv2.resize(lum, (max(1, lum.shape[1] // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(0.5, halation_sigma(h) / 4))
        lum = cv2.resize(small, (lum.shape[1], h), interpolation=cv2.INTER_LINEAR)
        k = np.float32(grade.halation / 100.0)
        glow = np.clip(lum[..., None] * (np.asarray(HALATION_TINT, np.float32) * (HALATION_GAIN * k)), 0.0, 1.0)
        x = 1.0 - (1.0 - x) * (1.0 - glow)
        out = (x * 255.0 + 0.5).astype(np.uint8)
    if abs(grade.sharpness) >= 0.5:
        n = sharpen_size(h)
        amt = sharpen_amount(grade.sharpness)
        out = cv2.addWeighted(out, 1.0 + amt, cv2.GaussianBlur(out, (n, n), 0), -amt, 0.0)
    return out


def ffmpeg_halation(grade: Grade, height: int, inp: str, out: str) -> list[str]:
    """Filter graph lines adding halation to RGB stream [inp] as [out] (nothing when off)."""
    if grade.halation < 0.5:
        return []
    t = HALATION_THRESHOLD
    luma = ":".join(f"{c}{d}={_LUMA[j]:.4f}" for c in "rgb" for j, d in enumerate("rgb"))
    tint = ":".join(f"{c}{c}={v * HALATION_GAIN * grade.halation / 100.0:.5f}" for c, v in zip("rgb", HALATION_TINT))
    return [f"[{inp}]split[{inp}h0][{inp}h1]",
            f"[{inp}h1]colorchannelmixer={luma},colorlevels=rimin={t}:gimin={t}:bimin={t},"
            f"colorchannelmixer={tint},format=gbrp16le,"
            f"gblur=sigma={halation_sigma(height):.2f}[{inp}hg]",
            f"[{inp}h0]format=gbrp16le[{inp}h2]",
            f"[{inp}h2][{inp}hg]blend=all_mode=screen[{out}]"]


def ffmpeg_sharpen(grade: Grade, height: int) -> str:
    """unsharp filter (luma) for a YUV stream, "" when off."""
    if abs(grade.sharpness) < 0.5:
        return ""
    n = sharpen_size(height)
    return f"unsharp={n}:{n}:{sharpen_amount(grade.sharpness):.3f}:{n}:{n}:0"


# ----------------------------------------------------------------------------- LUTs


class PreviewLut:
    """The grade as a table for fast per-frame grading of 8-bit RGB images (the preview).

    Blue is exact (all 256 levels: blue changes fastest from shadow to sky on a log picture, where a coarse
    grid shows as banding) and red and green are a 64-level grid, interpolated (bilinear). The table is an
    atlas image, a 64 x 64 red-green tile per blue level stacked vertically, so one cv2.remap call looks a
    pixel up and interpolates it. The error is about one 8-bit step on steep log curves; on extreme colours far
    outside Rec.709 (where the gamut compression bends sharply) it can reach a few percent, in the preview only:
    the export grades exactly through a 65-point LUT."""

    N = 64
    _x = np.arange(256, dtype=np.float32) * np.float32((N - 1) / 255.0)       # red -> atlas column
    _y = (np.arange(256, dtype=np.float32)[:, None] * N + _x[None, :]).ravel()  # blue * 256 + green -> row
    _grid: dict[int, np.ndarray] = {}
    _MAX = 32000                    # cv2.remap needs image sides under 32767
    # cv2 and numpy release the GIL, so row bands run truly in parallel.
    _threads = max(1, min(8, (os.cpu_count() or 2) - 1))
    _pool = ThreadPoolExecutor(_threads, thread_name_prefix="lut") if _threads > 1 else None

    @classmethod
    def grid(cls) -> np.ndarray:
        """Input RGB of every atlas entry, in atlas order (blue, green, red)."""
        if cls.N not in cls._grid:
            v = np.linspace(0.0, 1.0, cls.N, dtype=np.float32)
            b = np.arange(256, dtype=np.float32) / 255.0
            bb, gg, rr = np.meshgrid(b, v, v, indexing="ij")
            cls._grid[cls.N] = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
        return cls._grid[cls.N]

    def __init__(self, grade: Grade, log_profile: str, tonemap: str, full_range: bool = False):
        out = apply_grade(self.grid(), grade, log_profile, tonemap, full_range)
        self.atlas = (out * 255.0 + 0.5).astype(np.uint8).reshape(256 * self.N, self.N, 3)

    def _band(self, img: np.ndarray, out: np.ndarray) -> None:
        import cv2
        x = self._x[img[..., 0]]
        y = self._y[(img[..., 2].astype(np.uint16) << 8) | img[..., 1]]
        cv2.remap(self.atlas, x, y, cv2.INTER_LINEAR, dst=out)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Grade an RGB(A) image (8-bit values; extra channels dropped) to a new RGB uint8 image."""
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        img = img[..., :3]
        h, w = img.shape[:2]
        if img.size == 0:
            return np.empty((h, w, 3), np.uint8)
        if w >= self._MAX or h >= self._MAX:      # (not a picture: a long strip) graded in pieces
            flat = img.reshape(-1, 3)
            return np.concatenate([self.apply(flat[i:i + 4096][None])[0]
                                   for i in range(0, len(flat), 4096)]).reshape(h, w, 3)
        out = np.empty((h, w, 3), dtype=np.uint8)
        if self._pool is None or h < 64:
            self._band(img, out)
        else:
            step = -(-h // self._threads)
            list(self._pool.map(lambda y: self._band(img[y:y + step], out[y:y + step]), range(0, h, step)))
        return out


class LutCache:
    def __init__(self, max_items: int = 24):
        self._items: dict[tuple, PreviewLut] = {}
        self._max = max_items
        self._lock = threading.Lock()

    # Keyed by the effective profile, so a LUT-only profile's table is rebuilt once its official LUT arrives.
    def peek(self, grade: Grade, log_profile: str, tonemap: str, full_range: bool = False) -> Optional[PreviewLut]:
        return self._items.get((grade.key(), effective_profile(log_profile), tonemap, full_range))

    def get(self, grade: Grade, log_profile: str, tonemap: str, full_range: bool = False) -> Optional[PreviewLut]:
        if not needs_color(grade, log_profile):
            return None
        log_profile = effective_profile(log_profile)
        key = (grade.key(), log_profile, tonemap, full_range)
        lut = self._items.get(key)
        if lut is None:
            lut = PreviewLut(grade, log_profile, tonemap, full_range)   # built outside the lock (~0.2 s)
            with self._lock:
                if key not in self._items and len(self._items) >= self._max:
                    self._items.pop(next(iter(self._items)))
                self._items[key] = lut
        return lut


def write_cube(path: str, grade: Grade, log_profile: str, tonemap: str, size: int = 65,
               full_range: bool = False) -> None:
    """The grade as a .cube 3D LUT for FFmpeg (lut3d, tetrahedral). 65 points: log curves are steep in the
    shadows, where a 33-point grid errs by several 10-bit steps."""
    v = np.linspace(0.0, 1.0, size, dtype=np.float32)
    # .cube ordering: red changes fastest.
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    pts = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    out = apply_grade(pts, grade, log_profile, tonemap, full_range)
    with open(path, "w", encoding="ascii") as fh:
        fh.write("TITLE \"KangEdit grade\"\n")
        fh.write(f"LUT_3D_SIZE {size}\n")
        fh.write("DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        np.savetxt(fh, out, fmt="%.6f")


# ----------------------------------------------------------------------------- auto correction


def auto_correct(frames: list[np.ndarray], log_profile: str, tonemap: str = "filmic",
                 base: Optional[Grade] = None, full_range: bool = False) -> Grade:
    """Estimate exposure, white balance and contrast from sample RGB uint8 frames."""
    g = Grade() if base is None else Grade.from_dict(base.to_dict())
    if not frames:
        return g
    log_profile = effective_profile(log_profile)
    px = np.concatenate([f.reshape(-1, 3)[:: max(1, f.size // 3 // 40000)] for f in frames]).astype(np.float32) / 255.0
    lin = _to_linear(px, log_profile, full_range)
    lin = np.maximum(lin, 1e-5)
    luma = lin @ _LUMA

    # White balance: grey-world over mid-tones (ignore clipped and very dark pixels).
    lo, hi = np.percentile(luma, [10, 97])
    mask = (luma > lo) & (luma < hi) & (np.max(px, axis=1) < 0.98)
    if mask.sum() > 100:
        m = lin[mask].mean(axis=0)
        mr, mg, mb = (float(c) for c in m)
        t = np.log2(mb / mr)                       # 0.5T - (-0.5T) = log2(mb/mr)
        n = -np.log2(np.sqrt(mr * mb) / mg) / 0.4
        strength = 0.8
        g.temperature = float(np.clip(t * 100.0 * strength, -60, 60))
        g.tint = float(np.clip(n * 100.0 * strength, -40, 40))

    # Exposure: bring the log-average luminance to mid grey.
    lin_wb = lin * wb_gains(g.temperature, g.tint)
    luma = lin_wb @ _LUMA
    key = float(np.exp(np.mean(np.log(np.maximum(luma, 1e-4)))))
    target = 0.16 if log_profile in LOG_PROFILES else 0.14
    ev = np.log2(target / max(key, 1e-4))
    # Protect highlights: limit the push if many pixels would clip.
    p99 = float(np.percentile(luma, 99.5))
    clip_level = 2.5 if (log_profile in LOG_PROFILES and tonemap == "filmic") else 1.0
    if p99 > 0 and ev > 0:
        ev = min(ev, np.log2(clip_level / p99) + 0.5)
    g.exposure = float(np.clip(ev * 0.85, -3.0, 3.0))

    # Contrast: add punch to flat images, soften harsh ones.
    tmp = Grade(exposure=g.exposure, temperature=g.temperature, tint=g.tint)
    disp = apply_grade(px[::4], tmp, log_profile, tonemap, full_range) @ _LUMA
    p5, p95 = np.percentile(disp, [5, 95])
    spread = float(p95 - p5)
    g.contrast = float(np.clip((0.62 - spread) * 120.0, -25.0, 35.0))
    g.vibrance = 15.0 if log_profile in LOG_PROFILES and log_profile not in _HIDDEN_PROFILES else 5.0
    return g
