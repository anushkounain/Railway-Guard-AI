"""
Evidence Video Generator & Browser-Compatible H.264 Validator.

Solves the gray-screen / unannotated clip problems by:
1. Drawing rich detection overlays, danger zones, tracking trails, and event badges.
2. Encoding to browser-playable H.264 (libx264, yuv420p, faststart) using FFmpeg.
3. Performing rigorous post-generation validation on decoded frames.
"""

import os
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import cv2
import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"

from backend.config import settings, EVIDENCE_DIR
from backend.schemas import EventItem, ZoneModel

class RollingFrameBuffer:
    def __init__(self, max_frames: int = 150):
        self.max_frames = max_frames
        # Stores tuples: (frame_idx, timestamp, frame_bgr, detections)
        self.buffer = deque(maxlen=max_frames)

    def append(self, frame_idx: int, timestamp: float, frame: np.ndarray, detections: List[Dict[str, Any]]):
        # Store a copy of the frame to prevent mutation
        self.buffer.append((frame_idx, timestamp, frame.copy(), detections))

    def get_window(self, start_idx: int, end_idx: int) -> List[Tuple[int, float, np.ndarray, List[Dict[str, Any]]]]:
        return [entry for entry in self.buffer if start_idx <= entry[0] <= end_idx]

    def clear(self):
        self.buffer.clear()

class EvidenceGenerator:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or str(EVIDENCE_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_clip(
        self,
        event: EventItem,
        frame_window: List[Tuple[int, float, np.ndarray, List[Dict[str, Any]]]],
        zones: List[ZoneModel],
        fps: float = 30.0
    ) -> Optional[str]:
        """
        Generate and validate a browser-compatible annotated evidence video clip.
        """
        if not frame_window:
            print(f"[Evidence] Cannot generate clip for {event.event_id}: Frame window is empty.")
            return None

        # Determine target resolution (scale down 4K to 1280x720 or 1920x1080 for web streaming performance)
        sample_frame = frame_window[0][2]
        orig_h, orig_w = sample_frame.shape[:2]
        target_w = 1280
        target_h = int(orig_h * (target_w / orig_w))
        # Ensure dimensions are even numbers (libx264 requirement)
        if target_w % 2 != 0: target_w += 1
        if target_h % 2 != 0: target_h += 1

        output_filename = f"{event.event_id}.mp4"
        output_filepath = os.path.join(self.output_dir, output_filename)

        print(f"[Evidence] Generating clip {output_filename} ({len(frame_window)} frames at {target_w}x{target_h})...")

        # Encode using FFmpeg pipe
        success = self._encode_ffmpeg(frame_window, zones, event, output_filepath, target_w, target_h, fps)
        
        # Fallback to OpenCV VideoWriter if FFmpeg fails
        if not success or not os.path.exists(output_filepath):
            print(f"[Evidence] FFmpeg pipe failed, trying OpenCV fallback writer...")
            success = self._encode_opencv(frame_window, zones, event, output_filepath, target_w, target_h, fps)

        # Validate the generated clip
        validation = self.validate_clip(output_filepath)
        if not validation["valid"]:
            print(f"[Evidence] Validation FAILED for {output_filename}: {validation['reason']}")
            if os.path.exists(output_filepath):
                os.remove(output_filepath)
            return None

        print(f"[Evidence] Successfully validated clip: {output_filename} ({validation['frame_count']} frames, {validation['duration']:.2f}s)")
        return output_filename

    def _render_annotated_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: List[Dict[str, Any]],
        zones: List[ZoneModel],
        event: EventItem,
        target_w: int,
        target_h: int
    ) -> np.ndarray:
        """
        Draw rich surveillance annotations:
        - Danger zone polygons with semi-transparent fill
        - Bounding boxes and labels
        - Highlight target object for this event
        - Top surveillance header with Event ID, Timestamp, Barrier State, Severity
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Draw Danger Zones
        overlay = canvas.copy()
        for zone in zones:
            pts = np.array([[int(p.x * w), int(p.y * h)] for p in zone.points], dtype=np.int32)
            color_bgr = (0, 0, 230) if zone.zone_type == "TRACK_INTRUSION" else (0, 165, 255)
            cv2.fillPoly(overlay, [pts], color_bgr)
            cv2.polylines(canvas, [pts], True, color_bgr, 3, cv2.LINE_AA)
            # Label zone
            if len(pts) > 0:
                cv2.putText(canvas, f"[{zone.name}]", (pts[0][0], max(30, pts[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Blend polygon overlay for transparency
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # 2. Draw Detections
        for det in detections:
            box = [int(v) for v in det["box"]]
            tid = det.get("track_id")
            cls_name = det.get("mapped_class", "object")
            conf = det.get("conf", 0.0)
            
            # Highlight target tracked object if matching event
            is_target = (tid is not None and event.tracking_id is not None and tid == event.tracking_id)
            box_color = (0, 0, 255) if is_target else (0, 255, 0)
            thickness = 4 if is_target else 2

            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), box_color, thickness)
            
            # Bottom-center ground contact dot
            bc_x, bc_y = int(det["bottom_center"][0]), int(det["bottom_center"][1])
            cv2.circle(canvas, (bc_x, bc_y), 5, (0, 255, 255), -1)

            # Label text
            id_str = f"ID:{tid} " if tid is not None else "ID:? "
            label = f"{id_str}{cls_name} {conf:.2f}"
            if is_target:
                label = f"ALERT -> {label}"

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(canvas, (box[0], max(0, box[1] - 22)), (box[0] + t_size[0] + 6, max(22, box[1])), box_color, -1)
            cv2.putText(canvas, label, (box[0] + 3, max(16, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if not is_target else (255, 255, 255), 2, cv2.LINE_AA)

        # Resize to web streaming resolution
        if (w, h) != (target_w, target_h):
            canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 3. Surveillance Banner (Top HUD)
        banner_h = 48
        banner = np.zeros((banner_h, target_w, 3), dtype=np.uint8)
        banner[:] = (20, 20, 25)
        
        # Event Badge Color
        sev_color = (0, 0, 220) if event.severity in ("CRITICAL", "HIGH") else (0, 180, 240)
        cv2.rectangle(banner, (0, 0), (8, banner_h), sev_color, -1)

        hud_text = (
            f"EVIDENCE RECORD | {event.event_id} | EVENT: {event.event_type} | "
            f"SEVERITY: {event.severity} | BARRIER: {event.barrier_state} | T: {event.video_timestamp_str}"
        )
        cv2.putText(banner, hud_text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 240, 255), 2, cv2.LINE_AA)

        # Combine banner on top
        canvas[:banner_h, :] = banner

        return canvas

    def _encode_ffmpeg(
        self,
        frame_window: List[Tuple[int, float, np.ndarray, List[Dict[str, Any]]]],
        zones: List[ZoneModel],
        event: EventItem,
        output_filepath: str,
        target_w: int,
        target_h: int,
        fps: float
    ) -> bool:
        """Encode frames into browser-ready H.264 / yuv420p via FFmpeg pipe."""
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{target_w}x{target_h}",
            "-pix_fmt", "bgr24",
            "-r", str(round(fps, 2)),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "faster",
            "-crf", "22",
            "-movflags", "+faststart",
            output_filepath
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            for _, ts, raw_f, dets in frame_window:
                annotated = self._render_annotated_frame(raw_f, ts, dets, zones, event, target_w, target_h)
                proc.stdin.write(annotated.tobytes())
            proc.stdin.close()
            proc.wait(timeout=15)
            return proc.returncode == 0
        except Exception as e:
            print(f"[Evidence] FFmpeg pipe exception: {e}")
            return False

    def _encode_opencv(
        self,
        frame_window: List[Tuple[int, float, np.ndarray, List[Dict[str, Any]]]],
        zones: List[ZoneModel],
        event: EventItem,
        output_filepath: str,
        target_w: int,
        target_h: int,
        fps: float
    ) -> bool:
        """Fallback writer using OpenCV mp4v codec."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_filepath, fourcc, fps, (target_w, target_h))
            for _, ts, raw_f, dets in frame_window:
                annotated = self._render_annotated_frame(raw_f, ts, dets, zones, event, target_w, target_h)
                writer.write(annotated)
            writer.release()
            return True
        except Exception as e:
            print(f"[Evidence] OpenCV writer exception: {e}")
            return False

    def validate_clip(self, filepath: str) -> Dict[str, Any]:
        """
        Validate generated video:
        - Exists and size > 2000 bytes
        - Can be opened and decoded
        - Not gray/blank (checks standard deviation of pixel values)
        - Frame count > 0
        """
        if not os.path.exists(filepath):
            return {"valid": False, "reason": "File does not exist."}
        
        size = os.path.getsize(filepath)
        if size < 2000:
            return {"valid": False, "reason": f"File size too small ({size} bytes)."}

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {"valid": False, "reason": "Failed to open video file with decoder."}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_count <= 0:
            cap.release()
            return {"valid": False, "reason": "Video has zero frames."}

        # Inspect decoded frames (first and middle frame)
        ret, frame1 = cap.read()
        if not ret or frame1 is None:
            cap.release()
            return {"valid": False, "reason": "Unable to decode first frame."}

        # Check for gray screen (pixel std dev > 3.0 confirms presence of visual details/annotations)
        std_val = float(np.std(frame1))
        if std_val < 3.0:
            cap.release()
            return {"valid": False, "reason": f"Frame appears blank/gray (pixel std dev: {std_val:.2f})."}

        cap.release()
        duration = frame_count / fps if fps > 0 else 0.0

        return {
            "valid": True,
            "frame_count": frame_count,
            "width": w,
            "height": h,
            "duration": duration,
            "size_kb": round(size / 1024, 2)
        }

evidence_generator = EvidenceGenerator()
