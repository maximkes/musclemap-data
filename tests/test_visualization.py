"""Tests for ``src.visualization``."""

from __future__ import annotations

import numpy as np

from src.visualization import _mean_act_for_segment


def test_mean_act_for_segment_right_shank_matches_substrings_not_global_mean() -> None:
    t, n = 5, 10
    acts = np.full((t, n), 9.0, dtype=np.float32)
    acts[0, 2] = 0.8
    acts[0, 7] = 0.2
    names = [f"m{i}" for i in range(n)]
    names[2] = "soleus_r"
    names[7] = "tib_ant_r"
    got = _mean_act_for_segment(4, 7, 0, acts, names)
    assert np.isclose(got, 0.5)
    global_mean = float(np.mean(acts[0, :]))
    assert not np.isclose(got, global_mean)


def test_mean_act_for_segment_unknown_pair_returns_zero() -> None:
    acts = np.ones((3, 4), dtype=np.float32)
    names = ["a", "b", "c", "d"]
    assert _mean_act_for_segment(0, 1, 0, acts, names) == 0.0
