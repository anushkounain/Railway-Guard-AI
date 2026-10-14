"""
Configuration and Environment Settings for T225 Railway Safety System.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVIDENCE_DIR = BASE_DIR / "evidence"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, EVIDENCE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default model path candidates
DEFAULT_DOWNLOAD_MODEL = r"C:\Users\91636\Downloads\best new.pt"
LOCAL_MODEL = MODELS_DIR / "best.pt"
FALLBACK_MODEL = "yolo11n.pt"

def resolve_model_path() -> str:
    """Resolve the model path from environment or expected local locations."""
    env_path = os.getenv("MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists(DEFAULT_DOWNLOAD_MODEL):
        return DEFAULT_DOWNLOAD_MODEL
    if os.path.exists(LOCAL_MODEL):
        return str(LOCAL_MODEL)
    return str(FALLBACK_MODEL)

class AppSettings(BaseModel):
    # Model
    model_path: str = Field(default_factory=resolve_model_path)
    secondary_model_path: str = os.getenv("SECONDARY_MODEL_PATH", "yolov8n.pt")
    detection_mode: str = os.getenv("DETECTION_MODE", "hybrid")  # "custom", "coco", or "hybrid"
    
    # Inference parameters
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.10"))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    inference_imgsz: int = int(os.getenv("INFERENCE_IMGSZ", "640"))
    device: str = os.getenv("DEVICE", "auto")
    
    # Tracking
    tracker_type: str = os.getenv("TRACKER_TYPE", "bytetrack.yaml")
    
    # Rule engine
    consecutive_frames_required: int = int(os.getenv("CONSECUTIVE_FRAMES", "3"))
    cooldown_seconds: float = float(os.getenv("COOLDOWN_SECONDS", "10.0"))
    tracking_gap_tolerance: int = int(os.getenv("TRACKING_GAP_TOLERANCE", "2"))
    barrier_state_default: str = os.getenv("BARRIER_STATE_DEFAULT", "UNKNOWN")  # OPEN, CLOSED, UNKNOWN
    
    # Evidence generation
    evidence_pre_seconds: float = float(os.getenv("EVIDENCE_PRE_SECONDS", "2.0"))
    evidence_post_seconds: float = float(os.getenv("EVIDENCE_POST_SECONDS", "3.0"))
    
    # Database
    db_path: str = str(DATA_DIR / "railway_safety.db")
    zones_path: str = str(DATA_DIR / "zones.json")

    # Class mappings
    class_mapping: dict = {
        "crossing": "railway_crossing",
        "track-signal-front": "railway_signal",
        "track-sign-front": "railway_sign",
        "person": "pedestrian",
        "car": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "motorcycle": "vehicle",
        "bicycle": "vehicle",
        "train": "train",
    }
    
    intruder_categories: list = ["pedestrian", "vehicle", "animal"]
    infrastructure_categories: list = ["railway_crossing", "railway_signal", "railway_sign"]

settings = AppSettings()
