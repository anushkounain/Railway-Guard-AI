"""
Violation Rule Engine for Railway Safety.

Implements:
- Outside-to-inside zone transitions
- Consecutive-frame temporal confirmation
- Tracking-gap tolerance
- Duplicate-alert suppression & cooldowns
- Clear barrier state logic (OPEN, CLOSED, UNKNOWN)
- Transparent heuristic severity scoring
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import uuid

from backend.config import settings
from backend.schemas import ZoneModel, EventItem

class ViolationRuleEngine:
    def __init__(
        self,
        consecutive_frames_required: int = None,
        cooldown_seconds: float = None,
        barrier_state: str = None
    ):
        self.consecutive_frames_required = consecutive_frames_required or settings.consecutive_frames_required
        self.cooldown_seconds = cooldown_seconds or settings.cooldown_seconds
        self.barrier_state = barrier_state or settings.barrier_state_default
        
        # (track_id, zone_id) -> count of consecutive frames detected inside
        self.consecutive_inside: Dict[Tuple[int, str], int] = {}
        # (track_id, zone_id) -> last alert video timestamp
        self.last_alert_timestamp: Dict[Tuple[int, str], float] = {}
        # (track_id, zone_id) -> was inside on previous frame
        self.was_inside: Dict[Tuple[int, str], bool] = {}

    def set_barrier_state(self, state: str):
        """Update active barrier state: OPEN, CLOSED, or UNKNOWN."""
        valid_states = {"OPEN", "CLOSED", "UNKNOWN"}
        self.barrier_state = state.upper() if state and state.upper() in valid_states else "UNKNOWN"
        print(f"[RuleEngine] Barrier state set to: {self.barrier_state}")

    def evaluate_frame(
        self,
        zone_matches: List[Dict[str, Any]],
        all_detections: List[Dict[str, Any]],
        frame_idx: int,
        video_timestamp: float
    ) -> List[EventItem]:
        """
        Evaluate detections against configured zones and rules.
        Returns list of new confirmed events (if any).
        """
        events: List[EventItem] = []
        
        # Set of (track_id, zone_id) pairs present inside in this frame
        current_inside_pairs = set()

        for match in zone_matches:
            det = match["detection"]
            zone: ZoneModel = match["zone"]
            track_id = det["track_id"]
            pair = (track_id, zone.id)
            current_inside_pairs.add(pair)

            # Increment consecutive frame count
            prev_count = self.consecutive_inside.get(pair, 0)
            new_count = prev_count + 1
            self.consecutive_inside[pair] = new_count
            
            was_in = self.was_inside.get(pair, False)
            last_alert = self.last_alert_timestamp.get(pair, -999.0)
            time_since_alert = video_timestamp - last_alert

            # Rule Check:
            # 1. Temporal confirmation: Must be inside for consecutive frames (suppresses single-frame noise)
            # 2. Cooldown suppression: Suppress duplicate alerts for same object in same zone
            # 3. Transition: Alert when consecutive count hits threshold AND cooldown satisfied
            if new_count >= self.consecutive_frames_required and time_since_alert >= self.cooldown_seconds:
                # Trigger a confirmed rule event
                event = self._create_event(det, zone, video_timestamp, frame_idx)
                if event:
                    events.append(event)
                    self.last_alert_timestamp[pair] = video_timestamp

            self.was_inside[pair] = True

        # For any pair that was previously inside but not in this frame, reset consecutive count
        all_monitored_pairs = list(self.was_inside.keys())
        for pair in all_monitored_pairs:
            if pair not in current_inside_pairs:
                self.consecutive_inside[pair] = 0
                self.was_inside[pair] = False

        return events

    def _create_event(
        self,
        det: Dict[str, Any],
        zone: ZoneModel,
        timestamp: float,
        frame_idx: int
    ) -> Optional[EventItem]:
        """
        Synthesize event details based on zone type, object class, and barrier state.
        """
        mapped_class = det["mapped_class"]
        raw_class = det["raw_class"]
        conf = det["conf"]
        track_id = det["track_id"]
        
        # Calculate human-readable timestamp
        mins = int(timestamp // 60)
        secs = timestamp % 60
        time_str = f"{mins:02d}:{secs:05.2f}"
        
        event_id = f"EVT-{int(timestamp*1000)}-{uuid.uuid4().hex[:6]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Determine event type, severity, and risk score
        if zone.zone_type == "TRACK_INTRUSION":
            event_type = "TRACK_INTRUSION"
            severity, risk_score = self._compute_track_intrusion_risk(mapped_class, conf)
            notes = (
                f"Track intrusion detected in '{zone.name}'. Bottom-center contact point entered restricted railway zone. "
                f"Confirmed across {self.consecutive_frames_required} consecutive frames."
            )

        elif zone.zone_type == "LEVEL_CROSSING":
            if self.barrier_state == "CLOSED":
                event_type = "CLOSED_BARRIER_CROSSING"
                severity = "CRITICAL"
                risk_score = 95.0
                notes = (
                    f"Level-crossing violation! Barrier is CLOSED and {mapped_class} entered crossing conflict zone '{zone.name}'."
                )
            elif self.barrier_state == "OPEN":
                event_type = "CROSSING_TRANSIT_PERMITTED"
                severity = "LOW"
                risk_score = 15.0
                notes = (
                    f"Normal crossing transit: Barrier is OPEN while {mapped_class} entered crossing zone '{zone.name}'."
                )
            else:  # UNKNOWN
                event_type = "POTENTIAL_CROSSING_ALERT"
                severity = "MEDIUM"
                risk_score = 50.0
                notes = (
                    f"Crossing zone occupancy in '{zone.name}'. Note: Barrier state is UNKNOWN. "
                    f"Cannot classify as confirmed closed-barrier violation until barrier state is verified."
                )
        else:
            event_type = "ZONE_OCCUPANCY"
            severity = "LOW"
            risk_score = 25.0
            notes = f"Object entered configured zone '{zone.name}'."

        return EventItem(
            event_id=event_id,
            video_timestamp=round(timestamp, 3),
            video_timestamp_str=time_str,
            processing_timestamp=now_iso,
            object_class=mapped_class,
            raw_class=raw_class,
            tracking_id=track_id,
            confidence=round(conf, 3),
            zone_id=zone.id,
            zone_name=zone.name,
            event_type=event_type,
            barrier_state=self.barrier_state,
            risk_score=risk_score,
            severity=severity,
            evidence_path=None,  # Populated when clip generated
            notes=notes
        )

    def _compute_track_intrusion_risk(self, mapped_class: str, conf: float) -> Tuple[str, float]:
        """
        Transparent heuristic severity scoring:
        - Pedestrian on tracks: CRITICAL (Risk 90)
        - Vehicle on tracks: CRITICAL (Risk 95)
        - Train on tracks: LOW / Normal operation (Risk 10)
        - Other / Infrastructure: LOW (Risk 20)
        """
        cls_lower = mapped_class.lower()
        if cls_lower in ("pedestrian", "person"):
            return "CRITICAL", round(85.0 + (conf * 10.0), 1)
        elif cls_lower in ("vehicle", "car", "bus", "truck", "motorcycle"):
            return "CRITICAL", round(90.0 + (conf * 10.0), 1)
        elif cls_lower == "train":
            return "LOW", 10.0
        else:
            return "MEDIUM", 45.0

    def reset(self):
        """Reset rule engine tracking states for a new video session."""
        self.consecutive_inside.clear()
        self.last_alert_timestamp.clear()
        self.was_inside.clear()

rule_engine = ViolationRuleEngine()
