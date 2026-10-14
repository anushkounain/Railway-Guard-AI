"""
Object Tracking State Management & Trajectory Tracking.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class TrackStateManager:
    def __init__(self, max_history: int = 30, gap_tolerance: int = 5):
        self.max_history = max_history
        self.gap_tolerance = gap_tolerance
        # track_id -> list of {"frame": int, "point": (x, y), "box": [...], "timestamp": float}
        self.tracks: Dict[int, List[Dict[str, Any]]] = {}
        # track_id -> last seen frame number
        self.last_seen: Dict[int, int] = {}
        self.unassigned_counter = 1

    def update(self, detections: List[Dict[str, Any]], frame_idx: int, timestamp: float) -> List[Dict[str, Any]]:
        """
        Update tracking history with current frame detections.
        Assigns fallback pseudo-IDs if tracker dropped the ID.
        """
        updated_detections = []
        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                # Assign fallback pseudo ID based on spatial proximity if possible
                tid = self._find_nearest_track(det["bottom_center"], frame_idx)
                if tid is None:
                    tid = -(self.unassigned_counter)
                    self.unassigned_counter += 1
                det["track_id"] = tid

            # Record history
            if tid not in self.tracks:
                self.tracks[tid] = []
            
            self.tracks[tid].append({
                "frame": frame_idx,
                "point": det["bottom_center"],
                "box": det["box"],
                "timestamp": timestamp,
                "class": det["mapped_class"]
            })
            if len(self.tracks[tid]) > self.max_history:
                self.tracks[tid].pop(0)

            self.last_seen[tid] = frame_idx
            updated_detections.append(det)

        # Cleanup tracks older than gap_tolerance
        stale_ids = [tid for tid, last_f in self.last_seen.items() if frame_idx - last_f > self.gap_tolerance * 6]
        for tid in stale_ids:
            self.tracks.pop(tid, None)
            self.last_seen.pop(tid, None)

        return updated_detections

    def _find_nearest_track(self, point: List[float], current_frame: int, max_dist: float = 60.0) -> Optional[int]:
        """Find recent track whose last known point is within max_dist pixels."""
        best_tid = None
        min_dist = float("inf")
        for tid, history in self.tracks.items():
            if not history:
                continue
            last_entry = history[-1]
            if current_frame - last_entry["frame"] <= self.gap_tolerance:
                lp = last_entry["point"]
                dist = np.hypot(point[0] - lp[0], point[1] - lp[1])
                if dist < min_dist and dist <= max_dist:
                    min_dist = dist
                    best_tid = tid
        return best_tid

    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Retrieve recent bottom-center points for visualization trails."""
        if track_id in self.tracks:
            return [entry["point"] for entry in self.tracks[track_id]]
        return []

    def reset(self):
        """Reset all tracking states."""
        self.tracks.clear()
        self.last_seen.clear()
        self.unassigned_counter = 1
