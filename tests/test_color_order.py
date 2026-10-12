"""The grade's order: input transform, exposure / white balance, tone around middle grey, saturation, output
transform, then the creative look LUT (see color.apply_grade)."""

import numpy as np
import pytest

from kangedit.color import MID_GREY, Grade, apply_grade, lut_profile

from .conftest import write_cube


def _grey(level: float) -> np.ndarray:
    return np.full((1, 3), level, np.float32)


@pytest.mark.parametrize("profile", ["none", "slog3_sg3c", "vlog"])
def test_contrast_pivots_on_middle_grey(profile):
    # the signal level of middle grey: 18 % scene light (log footage) or display light (Rec.709)
    from kangedit.color import _to_linear
    levels = np.linspace(0, 1, 4001, dtype=np.float32)[:, None].repeat(3, 1)
    lin = _to_linear(levels, profile)[:, 1]
    grey = levels[np.argmin(np.abs(lin - MID_GREY)), :1]
    base = apply_grade(_grey(grey[0]), Grade(), profile)[0, 0]
    for c in (-60, 60):
        assert apply_grade(_grey(grey[0]), Grade(contrast=c), profile)[0, 0] == pytest.approx(base, abs=0.02)


def test_exposure_comes_before_a_conversion_lut(lut_library):
    # A conversion LUT that clips its brightest input: correcting exposure after it could not bring those
    # highlights back, before it (on the log signal) it does.
    conv = write_cube(lut_library / "conv.cube", 17, lambda r, g, b: tuple(min(1.0, v * 1.25) for v in (r, g, b)))
    hi = np.array([[0.82] * 3, [0.86] * 3], np.float32)
    flat = apply_grade(hi, Grade(), lut_profile(conv))[:, 0]
    assert flat[0] == pytest.approx(flat[1], abs=1e-3)            # both clipped by the LUT
    darker = apply_grade(hi, Grade(exposure=-2), lut_profile(conv))[:, 0]
    assert darker[1] - darker[0] > 0.03                           # detail kept: the LUT saw them darker


def test_look_lut_comes_last(lut_library):
    # An inverting look sees the graded display image: the grade can't undo it afterwards.
    look = write_cube(lut_library / "invert.cube", 2, lambda r, g, b: (1 - r, 1 - g, 1 - b))
    out = apply_grade(_grey(0.2), Grade(exposure=1, look=look), "none")[0, 0]
    graded = apply_grade(_grey(0.2), Grade(exposure=1), "none")[0, 0]
    assert out == pytest.approx(1 - graded, abs=0.01)


def test_scene_tone_rolls_off_instead_of_clipping():
    # On log footage, lifting the highlights is compressed by the filmic output transform.
    levels = np.linspace(0.3, 0.9, 7, dtype=np.float32)[:, None].repeat(3, 1)
    out = apply_grade(levels, Grade(highlights=80), "slog3_sg3c")[:, 0]
    assert np.all(np.diff(out) > 0) and out[-1] <= 1.0


def test_saturation_keeps_brightness():
    from kangedit.color import _LUMA
    c = np.array([[0.7, 0.4, 0.2]], np.float32)
    lin = lambda d: np.power(d, 2.4) @ _LUMA  # noqa: E731
    for sat in (0.0, 160.0):
        assert lin(apply_grade(c, Grade(saturation=sat), "none")) == pytest.approx(lin(c), rel=0.03)
