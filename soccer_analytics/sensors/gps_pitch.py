"""GPS 4-Corner Pitch Georeferencing & Flat Heatmap Generator.

Allows operators to mark 4 field corner GPS coordinates on any campus pitch
and transforms wearable GPS fixes (lat, lon) into standard 2D metric pitch
coordinates (105m x 68m) without perspective distortion.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

EARTH_RADIUS_M = 6371000.0


class GPSPitchTransformer:
    """Transforms WGS84 GPS (lat, lon) to 2D pitch coordinates (0..length, 0..width)."""

    def __init__(
        self,
        corners_gps: Optional[Dict[str, Tuple[float, float]]] = None,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ):
        self.pitch_length = pitch_length_m
        self.pitch_width = pitch_width_m
        self.H_gps_to_pitch: Optional[np.ndarray] = None
        self.ref_lat: float = 0.0
        self.ref_lon: float = 0.0

        if corners_gps:
            self.fit_corners(corners_gps)

    def fit_corners(self, corners: Dict[str, Tuple[float, float]]):
        """Fits homography from 4 named GPS corners: tl_corner, tr_corner, br_corner, bl_corner."""
        required = ["tl_corner", "tr_corner", "br_corner", "bl_corner"]
        for k in required:
            if k not in corners:
                raise ValueError(f"Missing required GPS corner landmark: {k}")

        self.ref_lat, self.ref_lon = corners["tl_corner"]

        # Convert corners to local tangent plane meters
        local_pts = []
        target_pts = []

        corner_targets = {
            "tl_corner": (0.0, 0.0),
            "tr_corner": (self.pitch_length, 0.0),
            "br_corner": (self.pitch_length, self.pitch_width),
            "bl_corner": (0.0, self.pitch_width),
        }

        for k in required:
            lat, lon = corners[k]
            lx, ly = self._latlon_to_local_m(lat, lon)
            local_pts.append([lx, ly])
            target_pts.append(list(corner_targets[k]))

        src = np.asarray(local_pts, dtype=np.float64)
        dst = np.asarray(target_pts, dtype=np.float64)

        from ..view import fit_homography
        self.H_gps_to_pitch = fit_homography(src, dst)

    def _latlon_to_local_m(self, lat: float, lon: float) -> Tuple[float, float]:
        """Equirectangular local metric projection relative to reference origin."""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(self.ref_lat)
        ref_lon_rad = math.radians(self.ref_lon)

        x = EARTH_RADIUS_M * (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad)
        y = EARTH_RADIUS_M * (ref_lat_rad - lat_rad)
        return x, y

    def gps_to_pitch(self, lat: float, lon: float) -> Tuple[float, float]:
        """Transforms a single (lat, lon) GPS fix to (x_pitch, y_pitch) in meters."""
        if self.H_gps_to_pitch is None:
            return 52.5, 34.0

        lx, ly = self._latlon_to_local_m(lat, lon)
        from ..view import apply_homography
        res = apply_homography(self.H_gps_to_pitch, [[lx, ly]])[0]
        x, y = float(res[0]), float(res[1])

        x = max(-5.0, min(self.pitch_length + 5.0, x))
        y = max(-5.0, min(self.pitch_width + 5.0, y))
        return round(x, 2), round(y, 2)

    def generate_density_grid(
        self,
        gps_fixes: Sequence[Tuple[float, float]],
        grid_res_m: float = 1.0,
        sigma_m: float = 2.0,
    ) -> np.ndarray:
        """Generates a smoothed 2D spatial density matrix from a list of GPS fixes."""
        from scipy.ndimage import gaussian_filter

        nx = max(10, int(self.pitch_length / grid_res_m))
        ny = max(10, int(self.pitch_width / grid_res_m))
        grid = np.zeros((ny, nx), dtype=np.float32)

        for lat, lon in gps_fixes:
            px, py = self.gps_to_pitch(lat, lon)
            if 0.0 <= px <= self.pitch_length and 0.0 <= py <= self.pitch_width:
                gx = min(nx - 1, max(0, int((px / self.pitch_length) * nx)))
                gy = min(ny - 1, max(0, int((py / self.pitch_width) * ny)))
                grid[gy, gx] += 1.0

        if grid.sum() > 0:
            sigma_px = sigma_m / grid_res_m
            grid = gaussian_filter(grid, sigma=sigma_px)
            grid = grid / grid.max()

        return grid
