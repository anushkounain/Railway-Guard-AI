"""
Unit and Integration Tests for FastAPI REST Endpoints.
"""

import unittest
from starlette.testclient import TestClient
from backend.main import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "T225-Railway-Safety")
        self.assertIn("processor_status", data)

    def test_model_info(self):
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_path", data)
        self.assertIn("model_exists", data)
        self.assertIn("model_names", data)
        self.assertIn("device", data)

    def test_zones_retrieval_and_update(self):
        # 1. Get zones
        get_res = self.client.get("/api/zones")
        self.assertEqual(get_res.status_code, 200)
        zones = get_res.json()
        self.assertIsInstance(zones, list)
        self.assertGreaterEqual(len(zones), 1)

        # 2. Update zones with a valid 3-point zone
        payload = {
            "zones": [
                {
                    "id": "test-api-zone",
                    "name": "API Test Zone",
                    "zone_type": "TRACK_INTRUSION",
                    "points": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.9, "y": 0.1},
                        {"x": 0.5, "y": 0.9}
                    ],
                    "is_demo": False,
                    "color": "#10b981"
                }
            ]
        }
        put_res = self.client.put("/api/zones", json=payload)
        self.assertEqual(put_res.status_code, 200)
        data = put_res.json()
        self.assertIn("Successfully updated", data["message"])

        # 3. Validation failure for polygon with only 2 points
        invalid_payload = {
            "zones": [
                {
                    "id": "invalid-api-zone",
                    "name": "Invalid Two Points",
                    "zone_type": "TRACK_INTRUSION",
                    "points": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.9, "y": 0.1}
                    ],
                    "is_demo": False,
                    "color": "#10b981"
                }
            ]
        }
        bad_res = self.client.put("/api/zones", json=invalid_payload)
        # Should fail with 422 (Pydantic min_length validation) or 400
        self.assertIn(bad_res.status_code, (400, 422))

        # 4. Restore original zones
        self.client.put("/api/zones", json={"zones": zones})

    def test_barrier_state_toggle(self):
        for state in ["OPEN", "CLOSED", "UNKNOWN"]:
            res = self.client.put(f"/api/barrier-state?state={state}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["state"], state)

    def test_processing_status(self):
        res = self.client.get("/api/processing/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("current_frame", data)
        self.assertIn("total_frames", data)
        self.assertIn("active_alerts_count", data)

    def test_events_list(self):
        res = self.client.get("/api/events")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("events", data)
        self.assertIn("count", data)

    def test_videos_list(self):
        res = self.client.get("/api/videos")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("videos", data)

    def test_video_stream_not_found(self):
        res = self.client.get("/api/videos/stream?path=nonexistent_file.mp4")
        self.assertEqual(res.status_code, 404)

    def test_output_latest_video(self):
        res = self.client.get("/api/outputs/latest-video")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("available", data)

    def test_output_video_stream_not_found(self):
        res = self.client.get("/api/outputs/video/nonexistent_annotated.mp4")
        self.assertEqual(res.status_code, 404)

    def test_live_frame_endpoint(self):
        res = self.client.get("/api/processing/live-frame")
        self.assertIn(res.status_code, [200, 404])

if __name__ == "__main__":
    unittest.main()
