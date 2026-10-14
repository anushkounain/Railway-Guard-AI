"""
Pydantic Schemas for T225 Railway Safety System.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ZonePoint(BaseModel):
    x: float = Field(..., description="X coordinate (normalized 0.0 to 1.0)")
    y: float = Field(..., description="Y coordinate (normalized 0.0 to 1.0)")

class ZoneModel(BaseModel):
    id: str = Field(..., description="Unique zone identifier")
    name: str = Field(..., description="Display name for the zone")
    zone_type: str = Field(default="TRACK_INTRUSION", description="TRACK_INTRUSION or LEVEL_CROSSING")
    points: List[ZonePoint] = Field(..., min_length=3, description="List of at least 3 polygon vertices")
    is_demo: bool = Field(default=False, description="Whether this zone is an unverified demo configuration")
    color: str = Field(default="#ef4444", description="Hex display color for frontend and overlays")

class ZoneListPayload(BaseModel):
    zones: List[ZoneModel]

class DetectionItem(BaseModel):
    box: List[float] = Field(..., description="[x1, y1, x2, y2] in pixels")
    conf: float
    cls_id: int
    raw_class: str
    mapped_class: str
    track_id: Optional[int] = None
    bottom_center: List[float] = Field(..., description="[x, y] in pixels")

class EventItem(BaseModel):
    event_id: str
    video_timestamp: float
    video_timestamp_str: str
    processing_timestamp: str
    object_class: str
    raw_class: str
    tracking_id: Optional[int] = None
    confidence: float
    zone_id: str
    zone_name: str
    event_type: str  # TRACK_INTRUSION, CLOSED_BARRIER_CROSSING, POTENTIAL_CROSSING_EVENT
    barrier_state: str  # OPEN, CLOSED, UNKNOWN
    risk_score: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    evidence_path: Optional[str] = None
    notes: str = ""

class StartProcessingPayload(BaseModel):
    video_path: Optional[str] = None
    conf_threshold: Optional[float] = 0.10
    barrier_state: Optional[str] = "UNKNOWN"
    detection_mode: Optional[str] = "hybrid"  # custom, coco, hybrid
    generate_evidence: Optional[bool] = True

class ProcessingStatusResponse(BaseModel):
    status: str  # IDLE, PROCESSING, PAUSED, COMPLETED, STOPPED, ERROR
    current_frame: int = 0
    total_frames: int = 0
    fps: float = 0.0
    progress_percent: float = 0.0
    active_alerts_count: int = 0
    total_detections_count: int = 0
    events_count: int = 0
    barrier_state: str = "UNKNOWN"
    error_message: Optional[str] = None
    output_video_path: Optional[str] = None
    detections_csv_path: Optional[str] = None

class ModelInfoResponse(BaseModel):
    model_path: str
    model_exists: bool
    model_type: str
    model_names: Dict[int, str]
    parameter_count: Optional[int] = None
    device: str
    cuda_available: bool
    classes_count: int
    file_size_mb: float
    detection_mode: str
    is_custom_checkpoint: bool
