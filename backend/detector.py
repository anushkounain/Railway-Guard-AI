"""
YOLO Object Detection & Model Loading Module for T225 Railway Safety.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import torch
from ultralytics import YOLO
import numpy as np

from backend.config import settings

class RailwayDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.model_path
        self.secondary_model_path = settings.secondary_model_path
        self.device = self._select_device()
        self.custom_model: Optional[YOLO] = None
        self.coco_model: Optional[YOLO] = None
        self.model_names: Dict[int, str] = {}
        self.is_custom_checkpoint: bool = False
        
        self.load_models()

    def _select_device(self) -> str:
        """Select CUDA if available and requested, otherwise CPU."""
        if settings.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif settings.device == "cpu":
            return "cpu"
        else:
            return "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self):
        """Safely load YOLO models with error handling."""
        print(f"[Detector] Loading primary model from: {self.model_path} (Device: {self.device})")
        
        if os.path.exists(self.model_path):
            try:
                self.custom_model = YOLO(self.model_path)
                self.model_names = dict(self.custom_model.names) if self.custom_model.names else {}
                self.is_custom_checkpoint = True
                print(f"[Detector] Primary model loaded successfully. Classes: {self.model_names}")
            except Exception as e:
                print(f"[Detector] Error loading primary model {self.model_path}: {e}")
                self.custom_model = None
        else:
            print(f"[Detector] Primary model path not found: {self.model_path}")
            
        # Also prepare COCO intruder model for hybrid/fallback mode
        try:
            print(f"[Detector] Initializing intruder detection model ({self.secondary_model_path})...")
            self.coco_model = YOLO(self.secondary_model_path)
            print(f"[Detector] Intruder model loaded successfully.")
        except Exception as e:
            print(f"[Detector] Notice: Could not load secondary model {self.secondary_model_path}: {e}")
            self.coco_model = None

    def get_model_info(self) -> Dict[str, Any]:
        """Return diagnostic information about loaded models."""
        exists = os.path.exists(self.model_path)
        file_size_mb = (os.path.getsize(self.model_path) / (1024 * 1024)) if exists else 0.0
        
        param_count = None
        if self.custom_model and hasattr(self.custom_model, "model") and self.custom_model.model is not None:
            try:
                param_count = sum(p.numel() for p in self.custom_model.model.parameters())
            except Exception:
                pass
                
        return {
            "model_path": self.model_path,
            "model_exists": exists,
            "model_type": str(type(self.custom_model)) if self.custom_model else "Not loaded",
            "model_names": self.model_names,
            "parameter_count": param_count,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "classes_count": len(self.model_names),
            "file_size_mb": round(file_size_mb, 2),
            "detection_mode": settings.detection_mode,
            "is_custom_checkpoint": self.is_custom_checkpoint
        }

    def detect_and_track(
        self,
        frame: np.ndarray,
        mode: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
        persist: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run detection and tracking on a single frame.
        Supports 'custom', 'coco', or 'hybrid' detection modes.
        """
        active_mode = mode or settings.detection_mode
        conf = conf_threshold if conf_threshold is not None else settings.conf_threshold
        sz = imgsz or settings.inference_imgsz
        
        detections: List[Dict[str, Any]] = []

        # 1. Custom checkpoint inference (e.g. crossing, signs, signals)
        if active_mode in ("custom", "hybrid") and self.custom_model is not None:
            try:
                # Run tracking with tracker config
                results = self.custom_model.track(
                    source=frame,
                    persist=persist,
                    tracker=settings.tracker_type,
                    conf=conf,
                    imgsz=sz,
                    device=self.device,
                    verbose=False
                )
                
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for b in boxes:
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        c_id = int(b.cls[0].item())
                        raw_cls = self.model_names.get(c_id, f"class_{c_id}")
                        score = float(b.conf[0].item())
                        track_id = int(b.id[0].item()) if b.id is not None else None
                        
                        # Bottom center for ground contact point
                        bc = [(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]]
                        mapped = settings.class_mapping.get(raw_cls, raw_cls)

                        detections.append({
                            "box": xyxy,
                            "conf": score,
                            "cls_id": c_id,
                            "raw_class": raw_cls,
                            "mapped_class": mapped,
                            "track_id": track_id,
                            "bottom_center": bc,
                            "source": "custom_checkpoint"
                        })
            except Exception as e:
                print(f"[Detector] Custom model inference warning: {e}")

        # 2. Intruder model inference (pedestrians, vehicles, etc.)
        if active_mode in ("coco", "hybrid") and self.coco_model is not None:
            try:
                # Use configured confidence threshold directly
                intruder_conf = conf
                coco_res = self.coco_model.track(
                    source=frame,
                    persist=persist,
                    tracker=settings.tracker_type,
                    conf=intruder_conf,
                    imgsz=sz,
                    device=self.device,
                    verbose=False
                )
                if coco_res and len(coco_res) > 0 and coco_res[0].boxes is not None:
                    boxes = coco_res[0].boxes
                    coco_names = self.coco_model.names
                    target_intruders = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "train"}
                    for b in boxes:
                        c_id = int(b.cls[0].item())
                        raw_cls = coco_names.get(c_id, f"class_{c_id}")
                        if raw_cls in target_intruders:
                            xyxy = b.xyxy[0].cpu().numpy().tolist()
                            score = float(b.conf[0].item())
                            # Offset track ID to avoid collisions with custom model track IDs
                            t_id = int(b.id[0].item()) if b.id is not None else None
                            track_id = (t_id + 10000) if t_id is not None else None
                            bc = [(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]]
                            mapped = settings.class_mapping.get(raw_cls, raw_cls)

                            detections.append({
                                "box": xyxy,
                                "conf": score,
                                "cls_id": c_id,
                                "raw_class": raw_cls,
                                "mapped_class": mapped,
                                "track_id": track_id,
                                "bottom_center": bc,
                                "source": "intruder_detector"
                            })
            except Exception as e:
                print(f"[Detector] COCO intruder inference warning: {e}")

        return detections

detector = RailwayDetector()
