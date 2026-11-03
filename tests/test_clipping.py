import numpy as np

from aiculler.scorers.clipping import clipping_stats


def _frame(h=300, w=400, value=120):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_scattered_specks_are_one_small_region_each():
    rgb = _frame()
    rgb[::4, ::4] = 255  # 1/16 of pixels clipped, none touching
    s = clipping_stats(rgb)
    assert abs(s["clip_total"] - 1 / 16) < 0.01
    assert s["clip_region"] < 0.001


def test_blown_sky_is_one_large_region_across_the_top_third():
    rgb = _frame()
    rgb[:100] = 255  # the whole top third
    s = clipping_stats(rgb)
    assert abs(s["clip_region"] - 1 / 3) < 0.01 and s["top_clip"] == 1.0


def test_diagonal_pixels_connect():
    rgb = _frame()
    for i in range(100):
        rgb[i, i] = 252  # a diagonal line, blown to white: 8-connected into one region
    s = clipping_stats(rgb)
    assert abs(s["clip_region"] - 100 / (300 * 400)) < 1e-4  # one region of 100 px (rounded to 4 places)
    assert s["clip_region"] == s["clip_total"] == s["sat_region"]


def test_saturated_colour_is_not_blown():
    """China 8-26 _DSC1055: a glowing yellow light ring (R 255, G 243, B 89) is colour, not lost detail."""
    rgb = _frame()
    rgb[:150, :, :] = (255, 243, 89)  # half the frame, the whole top third
    s = clipping_stats(rgb)
    assert s["clip_region"] == 0.0 and s["top_clip"] == 0.0  # nothing for the gates
    assert abs(s["sat_region"] - 0.5) < 0.01  # but it is measured, for the penalty


def test_washed_out_measures():
    flat = _frame(value=230)
    s = clipping_stats(flat)
    assert s["luma_mean"] > 0.85 and s["luma_spread"] == 0.0 and s["clip_region"] == 0.0
    contrasty = _frame(value=230)
    contrasty[:, :100] = 20
    assert clipping_stats(contrasty)["luma_spread"] > 0.5
