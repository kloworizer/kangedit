"""Log -> Rec.709 conversion quality: hue-preserving tone map, gamut compression, full-range files, and how
closely the preview table and the export LUT follow the exact transform."""

import numpy as np

from kangedit import color as C
from kangedit.lut import CubeLut

G = C.Grade()


def _slog3(lin):
    """Scene linear -> S-Log3 signal as a legal-range file decodes (0..1 over code values 64..940)."""
    cv = np.where(lin >= 0.01125, 420 + np.log10((lin + 0.01) / 0.19) * 261.5,
                  lin * (171.2102946929 - 95) / 0.01125 + 95)
    return ((cv - 64) / 876).astype(np.float32)


def _hue(rgb):
    lin = np.power(np.clip(rgb, 0, 1), 2.4)
    y = lin @ C._LUMA
    return np.degrees(np.arctan2(lin[:, 1] - y, lin[:, 0] - y)), lin


def test_neutrals_unchanged_and_bright_colours_keep_their_hue():
    grey = np.full((1, 3), 0.18, np.float32)
    assert np.allclose(C._filmic(grey)[0], C._aces_fit(np.float32(0.18 * 0.6)), atol=1e-6)   # grey as before
    # A saturated orange-red, 2.5 stops over middle grey: per channel it would drift to yellow.
    scene = np.array([[1.0, 0.3, 0.08]], np.float32) * 0.18 * 2 ** 2.5
    per_channel = C._aces_fit(scene * 0.6)
    ratio = lambda c: c[0, 1] / c[0, 0]                                   # noqa: E731  green / red
    assert abs(ratio(C._filmic(scene)) - 0.3) < 0.05 < abs(ratio(per_channel) - 0.3)
    # Very bright colours fade to white instead of clipping as a flat patch.
    hot = C._filmic(np.array([[1.0, 0.3, 0.08]], np.float32) * 40)[0]
    assert hot.min() > 0.8 and hot.max() <= 1.0


def test_out_of_gamut_colours_compressed_smoothly():
    inside = np.array([[0.5, 0.3, 0.2]], np.float32)
    assert np.array_equal(C._compress_gamut(inside), inside)              # well inside: untouched
    # A ramp of ever more saturated cyan (red going negative past the gamut edge): stays positive-ish and
    # keeps getting more saturated (no flat clipped patch), each step small.
    red = np.linspace(0.2, -0.6, 81, dtype=np.float32)
    ramp = np.stack([red, np.full_like(red, 0.6), np.full_like(red, 0.5)], 1)
    out = C._compress_gamut(ramp)
    assert out[:, 0].min() > -0.13                                        # the limit lands near the edge
    assert np.all(np.diff(out[:, 0]) < 0) and np.abs(np.diff(out[:, 0])).max() < 0.02


def test_full_range_file_reads_the_same_code_values():
    lin = np.array([0.0, 0.01, 0.18, 0.9, 4.0])
    legal = _slog3(lin)
    full = ((legal * 876 + 64) / 1023).astype(np.float32)                 # the same code values, full range
    grey = lambda s: np.repeat(s[:, None], 3, 1)                          # noqa: E731
    a = C.apply_grade(grey(legal), G, "slog3_sg3c")
    b = C.apply_grade(grey(full), G, "slog3_sg3c", full_range=True)
    wrong = C.apply_grade(grey(full), G, "slog3_sg3c")
    assert np.abs(a - b).max() < 1e-4 and np.abs(a - wrong).max() > 0.02
    # The preview tables and the export LUT are keyed / written for the file's range too.
    cache = C.LutCache()
    g = C.Grade(exposure=0.5)
    assert cache.get(g, "slog3_sg3c", "filmic", True) is not cache.get(g, "slog3_sg3c", "filmic", False)


def test_preview_table_follows_the_exact_grade():
    lut = C.PreviewLut(C.Grade(exposure=0.3, saturation=120), "slog3_sg3c", "filmic")
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    exact = C.apply_grade(img.astype(np.float32) / 255, C.Grade(exposure=0.3, saturation=120), "slog3_sg3c") * 255
    err = np.abs(lut.apply(img).astype(np.float32) - exact)
    assert np.percentile(err, 99) < 2.0 and err.mean() < 0.5
    ramp = np.repeat(np.arange(30, 200, dtype=np.uint8)[:, None], 3, 1)[None]       # a smooth grey ramp
    steps = np.diff(lut.apply(ramp)[0, :, 1].astype(int))
    assert steps.min() >= -1 and steps.max() <= 3                         # no banding (8-bit rounding only)
    strip = img.reshape(1, -1, 3)[:, :40000]                              # wider than cv2.remap takes
    assert np.array_equal(lut.apply(strip)[0, :19200], lut.apply(img).reshape(-1, 3))


def test_export_lut_follows_the_exact_grade(tmp_path):
    path = tmp_path / "g.cube"
    C.write_cube(str(path), G, "slog3_sg3c", "filmic")
    cube = CubeLut(str(path))
    assert cube.size == 65
    rng = np.random.default_rng(1)
    lum = 0.18 * 2 ** rng.uniform(-6, 5, 50000)
    scene = lum[:, None] * (0.3 + rng.random((50000, 3)) ** 2 * rng.uniform(0, 2.5, (50000, 1)))
    sig = np.clip(_slog3(scene), 0, 1)                                     # what a camera actually records
    err = np.abs(cube.apply(sig) - C.apply_grade(sig, G, "slog3_sg3c")) * 1023
    assert err.mean() < 1.5 and np.percentile(err, 99) < 6
