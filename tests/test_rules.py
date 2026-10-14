"""
Unit Tests for Violation Rule Engine, Temporal Confirmation, and Barrier Logic.
"""

import unittest
from backend.rules import ViolationRuleEngine
from backend.schemas import ZoneModel, ZonePoint

class TestViolationRules(unittest.TestCase):
    def setUp(self):
        self.engine = ViolationRuleEngine(
            consecutive_frames_required=3,
            cooldown_seconds=5.0,
            barrier_state="UNKNOWN"
        )
        self.track_zone = ZoneModel(
            id="z-track",
            name="Restricted Tracks",
            zone_type="TRACK_INTRUSION",
            points=[ZonePoint(x=0.1, y=0.1), ZonePoint(x=0.9, y=0.1), ZonePoint(x=0.5, y=0.9)],
            is_demo=False
        )
        self.crossing_zone = ZoneModel(
            id="z-crossing",
            name="Crossing Zone",
            zone_type="LEVEL_CROSSING",
            points=[ZonePoint(x=0.2, y=0.2), ZonePoint(x=0.8, y=0.2), ZonePoint(x=0.5, y=0.8)],
            is_demo=False
        )

    def test_consecutive_frame_temporal_confirmation(self):
        """Test that alert triggers ONLY after consecutive_frames_required (3 frames)."""
        det = {
            "track_id": 101,
            "mapped_class": "pedestrian",
            "raw_class": "person",
            "conf": 0.85,
            "bottom_center": [500, 500],
            "box": [480, 400, 520, 500]
        }
        match = [{"detection": det, "zone": self.track_zone}]

        # Frame 1: inside -> No alert yet (count = 1 < 3)
        ev1 = self.engine.evaluate_frame(match, [det], frame_idx=1, video_timestamp=0.1)
        self.assertEqual(len(ev1), 0)

        # Frame 2: inside -> No alert yet (count = 2 < 3)
        ev2 = self.engine.evaluate_frame(match, [det], frame_idx=2, video_timestamp=0.2)
        self.assertEqual(len(ev2), 0)

        # Frame 3: inside -> Alert CONFIRMED (count = 3 >= 3)
        ev3 = self.engine.evaluate_frame(match, [det], frame_idx=3, video_timestamp=0.3)
        self.assertEqual(len(ev3), 1)
        self.assertEqual(ev3[0].event_type, "TRACK_INTRUSION")
        self.assertEqual(ev3[0].severity, "CRITICAL")
        self.assertEqual(ev3[0].tracking_id, 101)

    def test_duplicate_alert_suppression_while_remaining_inside(self):
        """Test that remaining inside does NOT fire repeated alerts on every frame."""
        det = {
            "track_id": 202,
            "mapped_class": "vehicle",
            "raw_class": "car",
            "conf": 0.90,
            "bottom_center": [500, 500],
            "box": [450, 450, 550, 550]
        }
        match = [{"detection": det, "zone": self.track_zone}]

        # Frames 1-3: confirm alert on frame 3
        self.engine.evaluate_frame(match, [det], frame_idx=1, video_timestamp=1.0)
        self.engine.evaluate_frame(match, [det], frame_idx=2, video_timestamp=1.1)
        ev_initial = self.engine.evaluate_frame(match, [det], frame_idx=3, video_timestamp=1.2)
        self.assertEqual(len(ev_initial), 1)

        # Frames 4 through 10: still inside, within cooldown (1.2 to 2.0s < cooldown 5.0s)
        for f in range(4, 11):
            ts = 1.2 + (f - 3) * 0.1
            ev = self.engine.evaluate_frame(match, [det], frame_idx=f, video_timestamp=ts)
            self.assertEqual(len(ev), 0, f"Spurious alert fired on frame {f}")

    def test_barrier_state_unknown_handling(self):
        """Verify that when barrier state is UNKNOWN, event is NOT classified as confirmed closed-barrier violation."""
        self.engine.set_barrier_state("UNKNOWN")
        det = {
            "track_id": 303,
            "mapped_class": "vehicle",
            "raw_class": "truck",
            "conf": 0.88,
            "bottom_center": [500, 500],
            "box": [450, 450, 550, 550]
        }
        match = [{"detection": det, "zone": self.crossing_zone}]

        for f in range(1, 4):
            evs = self.engine.evaluate_frame(match, [det], frame_idx=f, video_timestamp=f * 0.1)

        self.assertEqual(len(evs), 1)
        event = evs[0]
        self.assertEqual(event.event_type, "POTENTIAL_CROSSING_ALERT")
        self.assertEqual(event.barrier_state, "UNKNOWN")
        self.assertEqual(event.severity, "MEDIUM")
        self.assertIn("barrier state is unknown", event.notes.lower())
        self.assertNotEqual(event.event_type, "CLOSED_BARRIER_CROSSING")

    def test_barrier_state_closed_handling(self):
        """Verify that when barrier state is CLOSED, crossing entry is classified as CRITICAL closed-barrier violation."""
        self.engine.set_barrier_state("CLOSED")
        det = {
            "track_id": 404,
            "mapped_class": "vehicle",
            "raw_class": "car",
            "conf": 0.92,
            "bottom_center": [500, 500],
            "box": [450, 450, 550, 550]
        }
        match = [{"detection": det, "zone": self.crossing_zone}]

        for f in range(1, 4):
            evs = self.engine.evaluate_frame(match, [det], frame_idx=f, video_timestamp=f * 0.1)

        self.assertEqual(len(evs), 1)
        event = evs[0]
        self.assertEqual(event.event_type, "CLOSED_BARRIER_CROSSING")
        self.assertEqual(event.barrier_state, "CLOSED")
        self.assertEqual(event.severity, "CRITICAL")
        self.assertGreaterEqual(event.risk_score, 90.0)

    def test_barrier_state_open_handling(self):
        """Verify that when barrier state is OPEN, normal crossing transit is flagged as LOW severity permitted transit."""
        self.engine.set_barrier_state("OPEN")
        det = {
            "track_id": 505,
            "mapped_class": "vehicle",
            "raw_class": "car",
            "conf": 0.85,
            "bottom_center": [500, 500],
            "box": [450, 450, 550, 550]
        }
        match = [{"detection": det, "zone": self.crossing_zone}]

        for f in range(1, 4):
            evs = self.engine.evaluate_frame(match, [det], frame_idx=f, video_timestamp=f * 0.1)

        self.assertEqual(len(evs), 1)
        event = evs[0]
        self.assertEqual(event.event_type, "CROSSING_TRANSIT_PERMITTED")
        self.assertEqual(event.barrier_state, "OPEN")
        self.assertEqual(event.severity, "LOW")

if __name__ == "__main__":
    unittest.main()
