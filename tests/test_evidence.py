"""
Unit Test for Evidence Video Generation & Frame Validation.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import cv2

from backend.evidence import EvidenceGenerator
from backend.schemas import EventItem, ZoneModel, ZonePoint

class TestEvidenceGeneration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = EvidenceGenerator(output_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_and_validate_clip(self):
        # Create synthetic synthetic surveillance frames (checkerboard/railway-like patterns with varied colors)
        frames = []
        for i in range(30):
            # 640x360 frame with colored shapes
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            img[100:260, :] = (40, 50, 60)  # track bed
            cv2.line(img, (0, 180), (640, 180), (180, 180, 180), 4)  # rail 1
            cv2.line(img, (0, 220), (640, 220), (180, 180, 180), 4)  # rail 2
            # Add a moving object box
            x_pos = 100 + i * 10
            cv2.rectangle(img, (x_pos, 150), (x_pos + 60, 240), (0, 200, 100), -1)

            det = [{
                "box": [x_pos, 150, x_pos + 60, 240],
                "conf": 0.91,
                "track_id": 99,
                "mapped_class": "pedestrian",
                "bottom_center": [x_pos + 30, 240]
            }]
            frames.append((i, i / 10.0, img, det))

        event = EventItem(
            event_id="EVT-TEST-EVID-001",
            video_timestamp=1.5,
            video_timestamp_str="00:01.50",
            processing_timestamp="2026-10-14T23:00:00Z",
            object_class="pedestrian",
            raw_class="person",
            tracking_id=99,
            confidence=0.91,
            zone_id="test-zone",
            zone_name="Main Track Danger Zone",
            event_type="TRACK_INTRUSION",
            barrier_state="UNKNOWN",
            risk_score=90.0,
            severity="CRITICAL",
            notes="Testing evidence generation"
        )

        zone = ZoneModel(
            id="test-zone",
            name="Main Track Danger Zone",
            zone_type="TRACK_INTRUSION",
            points=[
                ZonePoint(x=0.1, y=0.3),
                ZonePoint(x=0.9, y=0.3),
                ZonePoint(x=0.9, y=0.8),
                ZonePoint(x=0.1, y=0.8)
            ],
            is_demo=False
        )

        clip_name = self.generator.generate_clip(event, frames, [zone], fps=10.0)
        self.assertIsNotNone(clip_name, "Clip generation returned None")

        clip_path = os.path.join(self.temp_dir, clip_name)
        self.assertTrue(os.path.exists(clip_path), "Evidence clip file does not exist on disk")

        # Validate clip properties
        val = self.generator.validate_clip(clip_path)
        self.assertTrue(val["valid"], f"Validation failed: {val.get('reason')}")
        self.assertEqual(val["frame_count"], 30)
        self.assertGreaterEqual(val["duration"], 2.8)
        self.assertGreater(val["size_kb"], 5.0)

if __name__ == "__main__":
    unittest.main()
