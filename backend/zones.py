"""
Zone Management & Polygon Point-in-Zone Geometry.
"""

import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

from backend.config import settings
from backend.schemas import ZoneModel, ZonePoint

DEFAULT_ZONES = [
    {
        "id": "zone-track-01",
        "name": "Railway Track Intrusion Zone (Main)",
        "zone_type": "TRACK_INTRUSION",
        "points": [
            {"x": 0.05, "y": 0.40},
            {"x": 0.65, "y": 0.40},
            {"x": 0.70, "y": 0.95},
            {"x": 0.02, "y": 0.95}
        ],
        "is_demo": True,
        "color": "#ef4444"
    },
    {
        "id": "zone-crossing-01",
        "name": "Level-Crossing Conflict Zone",
        "zone_type": "LEVEL_CROSSING",
        "points": [
            {"x": 0.45, "y": 0.35},
            {"x": 0.95, "y": 0.35},
            {"x": 0.98, "y": 0.85},
            {"x": 0.40, "y": 0.85}
        ],
        "is_demo": True,
        "color": "#f59e0b"
    }
]

class ZoneManager:
    def __init__(self, zones_path: str = None):
        self.zones_path = zones_path or settings.zones_path
        self.zones: List[ZoneModel] = []
        self.load_zones()

    def load_zones(self) -> List[ZoneModel]:
        """Load zones from JSON or initialize with default template."""
        if os.path.exists(self.zones_path):
            try:
                with open(self.zones_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.zones = [ZoneModel(**z) for z in data]
                    return self.zones
            except Exception as e:
                print(f"[ZoneManager] Error loading zones from {self.zones_path}: {e}. Using defaults.")
        
        # Save defaults
        self.zones = [ZoneModel(**z) for z in DEFAULT_ZONES]
        self.save_zones(self.zones)
        return self.zones

    def save_zones(self, new_zones: List[ZoneModel]) -> bool:
        """Validate and save zones to JSON file."""
        for z in new_zones:
            if len(z.points) < 3:
                raise ValueError(f"Zone '{z.name}' ({z.id}) must have at least 3 points.")
            for p in z.points:
                if not (0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0):
                    # Clamp or validate normalized coordinates
                    p.x = max(0.0, min(1.0, p.x))
                    p.y = max(0.0, min(1.0, p.y))

        self.zones = new_zones
        os.makedirs(os.path.dirname(self.zones_path), exist_ok=True)
        with open(self.zones_path, "w", encoding="utf-8") as f:
            json.dump([z.model_dump() for z in self.zones], f, indent=2)
        return True

    def get_pixel_polygon(self, zone: ZoneModel, width: int, height: int) -> np.ndarray:
        """Scale normalized zone coordinates to frame pixel coordinates."""
        pts = [[int(p.x * width), int(p.y * height)] for p in zone.points]
        return np.array(pts, dtype=np.int32)

    def is_point_in_zone(self, point: Tuple[float, float], zone: ZoneModel, width: int, height: int) -> bool:
        """
        Check if (x, y) pixel coordinates fall inside the polygon zone.
        Uses OpenCV cv2.pointPolygonTest.
        """
        poly = self.get_pixel_polygon(zone, width, height)
        # cv2.pointPolygonTest returns positive if inside, 0 if on edge, negative if outside
        dist = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
        return dist >= 0

    def check_detections(self, detections: list, width: int, height: int) -> List[Dict[str, Any]]:
        """
        Check which detections are inside which configured zones.
        Uses bottom-center of the bounding box as ground contact point.
        """
        matches = []
        for det in detections:
            bc = det["bottom_center"]
            for zone in self.zones:
                if self.is_point_in_zone(bc, zone, width, height):
                    matches.append({
                        "detection": det,
                        "zone": zone
                    })
        return matches

zone_manager = ZoneManager()
