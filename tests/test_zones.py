"""
Unit Tests for Polygon Danger Zone Logic and Point-in-Zone Geometry.
"""

import unittest
import os
import tempfile
import json
import numpy as np
from backend.zones import ZoneManager
from backend.schemas import ZoneModel, ZonePoint

class TestZoneGeometry(unittest.TestCase):
    def setUp(self):
        # Create a temporary zones file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.manager = ZoneManager(zones_path=self.temp_file.name)

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_point_in_polygon_inside_and_outside(self):
        """Test that points inside the polygon return True, points outside return False."""
        # Square zone from (0.2, 0.2) to (0.8, 0.8)
        zone = ZoneModel(
            id="test-zone-1",
            name="Test Square Zone",
            zone_type="TRACK_INTRUSION",
            points=[
                ZonePoint(x=0.2, y=0.2),
                ZonePoint(x=0.8, y=0.2),
                ZonePoint(x=0.8, y=0.8),
                ZonePoint(x=0.2, y=0.8)
            ],
            is_demo=False
        )

        width, height = 1000, 1000

        # Center point (500, 500) -> inside
        self.assertTrue(self.manager.is_point_in_zone((500, 500), zone, width, height))

        # Far outside points
        self.assertFalse(self.manager.is_point_in_zone((50, 50), zone, width, height))
        self.assertFalse(self.manager.is_point_in_zone((900, 900), zone, width, height))
        self.assertFalse(self.manager.is_point_in_zone((150, 500), zone, width, height))

    def test_polygon_scaling(self):
        """Test scaling normalized coordinates to pixel coordinates."""
        zone = ZoneModel(
            id="test-zone-scale",
            name="Scale Zone",
            zone_type="TRACK_INTRUSION",
            points=[
                ZonePoint(x=0.1, y=0.2),
                ZonePoint(x=0.5, y=0.2),
                ZonePoint(x=0.5, y=0.6)
            ],
            is_demo=True
        )

        poly = self.manager.get_pixel_polygon(zone, width=1920, height=1080)
        self.assertEqual(poly.shape, (3, 2))
        self.assertEqual(poly[0][0], int(0.1 * 1920))
        self.assertEqual(poly[0][1], int(0.2 * 1080))

    def test_polygon_validation_min_three_points(self):
        """Test validation fails if polygon has fewer than 3 points."""
        with self.assertRaises(ValueError):
            invalid_zone = ZoneModel(
                id="invalid-zone",
                name="Too Few Points",
                zone_type="TRACK_INTRUSION",
                points=[
                    ZonePoint(x=0.1, y=0.1),
                    ZonePoint(x=0.5, y=0.5)
                ]
            )
            self.manager.save_zones([invalid_zone])

    def test_zone_persistence(self):
        """Test saving and loading zones from JSON."""
        zones = [
            ZoneModel(
                id="persisted-zone",
                name="Persisted Track Zone",
                zone_type="TRACK_INTRUSION",
                points=[
                    ZonePoint(x=0.0, y=0.0),
                    ZonePoint(x=1.0, y=0.0),
                    ZonePoint(x=0.5, y=1.0)
                ],
                is_demo=False,
                color="#ff0000"
            )
        ]
        self.manager.save_zones(zones)

        # Reload
        new_manager = ZoneManager(zones_path=self.temp_file.name)
        loaded = new_manager.zones
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, "persisted-zone")
        self.assertEqual(loaded[0].name, "Persisted Track Zone")
        self.assertFalse(loaded[0].is_demo)

if __name__ == "__main__":
    unittest.main()
