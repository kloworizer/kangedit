import re
import urllib.parse

import numpy as np
import pytest

from kangedit import lut as lutlib
from kangedit.color import (LOG_CHOICES, Grade, LutCache, apply_grade, effective_profile, lut_profile, missing_lut,
                            needs_color, profile_label)

from .conftest import write_cube


def _file_name(url: str) -> str:
    base = urllib.parse.unquote(url.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
    return re.sub(r"[._ ]+$", "", base)


def _profiles_matching(name: str) -> list[str]:
    return [e.profile for e in lutlib.OFFICIAL_LUTS if e.maker == "DJI" and lutlib.official_matches(e, [(name, "x")])]


def test_cube_identity(tmp_path):
    lut = lutlib.CubeLut(write_cube(tmp_path / "id.cube", size=3))
    rgb = np.random.default_rng(1).random((50, 3), dtype=np.float32)
    np.testing.assert_allclose(lut.apply(rgb), rgb, atol=1e-5)


def test_cube_rejects_wrong_entry_count(tmp_path):
    p = tmp_path / "bad.cube"
    p.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")
    with pytest.raises(lutlib.LutError):
        lutlib.CubeLut(str(p))


@pytest.mark.parametrize("item", [d for d in lutlib.DJI_LUTS if d.file.endswith(".cube")],
                         ids=lambda d: f"{d.product}-{d.profile}")
def test_every_dji_lut_is_recognised_as_its_own_profile(item):
    assert _profiles_matching(_file_name(item.file)) == [item.profile]


@pytest.mark.parametrize("name, profile", [
    ("DJI_Phantom4_DLOG2Rec709", "dlog"),          # "2" means "to" in DJI's older files, not D-Log2
    ("DJI_X5_DLOG2sRGB_Improv", "dlog"),
    ("D-Log to Rec709", "dlog"),
    ("DLog-M to Rec709", "dlogm"),
    ("DLog_Mavic3_to_Rec709", "dlog"),             # "Mavic", not D-Log M
    ("DJI Zenmuse X9 DLog2Rec2020HLG", None),       # HDR output, not Rec.709
    ("gamma18TosRGB", None),
    ("SLog3SGamut3.CineToLC-709", None),
])
def test_dji_name_edge_cases(name, profile):
    assert _profiles_matching(name) == ([profile] if profile else [])


def test_dji_catalog_is_consistent():
    profiles = {e.profile for e in lutlib.OFFICIAL_LUTS}
    assert {d.profile for d in lutlib.DJI_LUTS} <= profiles
    assert set(lutlib.LUT_ONLY) <= {d.profile for d in lutlib.DJI_LUTS}, "every LUT-only profile can be downloaded"
    keys = [(d.product, d.profile) for d in lutlib.DJI_LUTS]
    assert len(keys) == len(set(keys))
    assert all(d.file.startswith("https://") or not d.file for d in lutlib.DJI_LUTS)


def test_log_menu_lists_dji_profiles_after_dlog():
    keys = [k for k, _ in LOG_CHOICES]
    i = keys.index("dlog")
    assert keys[i + 1:i + 5] == ["dlog_vivid", "dlogm", "dlog2", "dlog2_vivid"]


def test_lut_only_profile_falls_back_to_its_curve(lut_library):
    assert effective_profile("dlogm") == "dlogm_approx"
    assert effective_profile("dlog2") == "dlog"
    assert missing_lut("dlogm")
    assert "approximated" in profile_label("dlogm")
    assert needs_color(Grade(), "dlogm")


def test_dlogm_approximation_is_milder_than_dlog():
    # D-Log M carries more contrast and colour than D-Log: decoding it as D-Log made it far too vivid.
    px = np.array([[0.55, 0.40, 0.48]], np.float32)
    chroma = lambda a: float(a.max() - a.min())  # noqa: E731
    assert chroma(apply_grade(px, Grade(), "dlogm_approx")) < 0.8 * chroma(apply_grade(px, Grade(), "dlog"))
    assert "dlogm_approx" not in [k for k, _ in LOG_CHOICES]


def test_lut_only_profile_uses_official_lut_once_imported(lut_library):
    # Halves every value, so the result is easy to tell apart from the D-Log curve.
    path = write_cube(lut_library / "DJI Air 3 D-Log M to Rec.709 V1.cube", fn=lambda r, g, b: (r / 2, g / 2, b / 2))
    assert effective_profile("dlogm") == lut_profile(path)
    assert not missing_lut("dlogm")
    out = apply_grade(np.array([[0.8, 0.8, 0.8]], np.float32), Grade(), "dlogm")
    np.testing.assert_allclose(out, [[0.4, 0.4, 0.4]], atol=0.01)


def test_choice_in_dialog_wins_over_best_match(lut_library):
    write_cube(lut_library / "DJI Air 3 D-Log M to Rec.709 V1.cube")
    chosen = write_cube(lut_library / "My D-Log M fix.cube")
    lutlib.set_official_choices({"dlogm": chosen})
    assert lutlib.official_lut_path("dlogm") == chosen


def test_preview_lut_cache_rebuilds_when_official_lut_arrives(lut_library):
    cache = LutCache()
    approx = cache.get(Grade(), "dlogm", "filmic")
    write_cube(lut_library / "DJI Mini 4 Pro D-Log M to Rec.709 V1.cube")
    assert cache.peek(Grade(), "dlogm", "filmic") is None
    assert cache.get(Grade(), "dlogm", "filmic") is not approx


def _resolve_cube(path, n1, size, n1_rows=None, range3=(0.0, 1.0)):
    """A Resolve-style .cube: a gamma-0.5 shaper (LUT_1D) followed by an identity 3D LUT over range3."""
    lines = ["TITLE \"Resolve shaper\"", f"LUT_1D_SIZE {n1}", "LUT_1D_INPUT_RANGE 0.0 1.0",
             f"LUT_3D_SIZE {size}", f"LUT_3D_INPUT_RANGE {range3[0]} {range3[1]}"]
    rows = n1 if n1_rows is None else n1_rows
    lines += [" ".join([f"{(k / (rows - 1)) ** 0.5:.6f}"] * 3) for k in range(rows)]
    g = np.linspace(range3[0], range3[1], size)
    lines += [f"{r:.6f} {gg:.6f} {b:.6f}" for b in g for gg in g for r in g]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_resolve_cube_with_a_shaper(tmp_path):
    from kangedit.lut import CubeLut
    lut = CubeLut(_resolve_cube(tmp_path / "r.cube", 4096, 65))       # the sizes of the reported file
    assert lut.kind == "3d" and lut.size == 65 and lut.shaper is not None and len(lut.shaper) == 4096
    x = np.array([[0.0, 0.25, 0.81], [0.04, 0.5, 1.0]], np.float32)
    assert np.allclose(lut.apply(x), np.sqrt(x), atol=2e-3)            # shaper, then the 3D LUT
    # one shaper row short of what LUT_1D_SIZE says (as in the reported file): still read, aligned on the 3D
    lut = CubeLut(_resolve_cube(tmp_path / "s.cube", 4096, 17, n1_rows=4095))
    assert len(lut.shaper) == 4095 and np.allclose(lut.apply(x), np.sqrt(x), atol=2e-3)
    # the 3D part reads its own input range
    lut = CubeLut(_resolve_cube(tmp_path / "t.cube", 1024, 17, range3=(0.0, 2.0)))
    assert np.allclose(lut.apply(np.array([[0.36, 0.36, 0.36]], np.float32)), 0.6, atol=3e-3)
