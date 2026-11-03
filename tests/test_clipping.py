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


def test_any_channel_counts_and_diagonal_pixels_connect():
    rgb = _frame()
    for i in range(100):
        rgb[i, i, 0] = 252  # red only, a diagonal line: 8-connected into one region
    s = clipping_stats(rgb)
    assert abs(s["clip_region"] - 100 / (300 * 400)) < 1e-4  # one region of 100 px (rounded to 4 places)
    assert s["clip_region"] == s["clip_total"]


def test_washed_out_measures():
    flat = _frame(value=230)
    s = clipping_stats(flat)
    assert s["luma_mean"] > 0.85 and s["luma_spread"] == 0.0 and s["clip_region"] == 0.0
    contrasty = _frame(value=230)
    contrasty[:, :100] = 20
    assert clipping_stats(contrasty)["luma_spread"] > 0.5
