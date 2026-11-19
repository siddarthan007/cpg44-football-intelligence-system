"""Tests for GPS 4-corner pitch transformation and density heatmaps."""

import numpy as np
import pytest
from soccer_analytics.sensors.gps_pitch import GPSPitchTransformer


def test_gps_4_corner_transformation():
    # Synthetic campus pitch corners (e.g. 105m x 68m orientation)
    corners = {
        "tl_corner": (12.9715987, 77.5945627),
        "tr_corner": (12.9715987, 77.5955310),   # ~105m east
        "br_corner": (12.9709860, 77.5955310),   # ~68m south
        "bl_corner": (12.9709860, 77.5945627),   # origin offset
    }

    transformer = GPSPitchTransformer(corners, pitch_length_m=105.0, pitch_width_m=68.0)

    # Top-left corner must map to (0.0, 0.0)
    tl_x, tl_y = transformer.gps_to_pitch(12.9715987, 77.5945627)
    assert abs(tl_x - 0.0) < 0.5
    assert abs(tl_y - 0.0) < 0.5

    # Top-right corner must map to (105.0, 0.0)
    tr_x, tr_y = transformer.gps_to_pitch(12.9715987, 77.5955310)
    assert abs(tr_x - 105.0) < 1.0
    assert abs(tr_y - 0.0) < 1.0

    # Bottom-right corner must map to (105.0, 68.0)
    br_x, br_y = transformer.gps_to_pitch(12.9709860, 77.5955310)
    assert abs(br_x - 105.0) < 1.0
    assert abs(br_y - 68.0) < 1.0

    # Pitch centre (halfway between lat/lon) must map near (52.5, 34.0)
    mid_lat = (12.9715987 + 12.9709860) / 2.0
    mid_lon = (77.5945627 + 77.5955310) / 2.0
    mid_x, mid_y = transformer.gps_to_pitch(mid_lat, mid_lon)
    assert abs(mid_x - 52.5) < 2.0
    assert abs(mid_y - 34.0) < 2.0


def test_gps_density_grid():
    corners = {
        "tl_corner": (12.9715987, 77.5945627),
        "tr_corner": (12.9715987, 77.5955310),
        "br_corner": (12.9709860, 77.5955310),
        "bl_corner": (12.9709860, 77.5945627),
    }
    transformer = GPSPitchTransformer(corners)
    fixes = [
        (12.9713, 77.5950),
        (12.9713, 77.5950),
        (12.9714, 77.5951),
    ]
    grid = transformer.generate_density_grid(fixes, grid_res_m=2.0)
    assert grid.shape[0] > 0
    assert grid.shape[1] > 0
    assert grid.max() <= 1.0
