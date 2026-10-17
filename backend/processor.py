"""
Asynchronous Video Processing Pipeline for T225 Railway Safety.
Runs in a background thread to prevent blocking FastAPI request handlers.
"""

import os
import time
import threading
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

from backend.config import settings, OUTPUTS_DIR, EVIDENCE_DIR, DATA_DIR
from backend.schemas import ProcessingStatusResponse, EventItem, ZoneModel
from backend.detector import detector
from backend.tracker import TrackStateManager
from backend.zones import zone_manager
from backend.rules import rule_engine
from backend.evidence import RollingFrameBuffer, evidence_generator, FFMPEG_EXE
from backend.storage import storage_manager

class VideoProcessor:
    def __init__(self):
        self.status = "IDLE"
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0.0
        self.start_time = 0.0
        self.active_alerts_count = 0
        self.total_detections_count = 0
        self.events_count = 0
        self.error_message: Optional[str] = None
        self.output_video_path: Optional[str] = None
        self.detections_csv_path: Optional[str] = None
        
        self.stop_requested = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Buffer and state tracking
        self.frame_buffer = RollingFrameBuffer(max_frames=300)
        self.tracker_manager = TrackStateManager()
        self.latest_frame_jpeg: Optional[bytes] = None
        
        # Pending evidence requests: list of dicts: {"event": EventItem, "end_frame": int}
        self.pending_evidence: List[Dict[str, Any]] = []

    def get_latest_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_frame_jpeg

    def get_status(self) -> ProcessingStatusResponse:
        with self.lock:
            progress = (self.current_frame / self.total_frames * 100.0) if self.total_frames > 0 else 0.0
            return ProcessingStatusResponse(
                status=self.status,
                current_frame=self.current_frame,
                total_frames=self.total_frames,
                fps=round(self.fps, 1),
                progress_percent=round(progress, 1),
                active_alerts_count=self.active_alerts_count,
                total_detections_count=self.total_detections_count,
                events_count=self.events_count,
                barrier_state=rule_engine.barrier_state,
                error_message=self.error_message,
                output_video_path=self.output_video_path,
                detections_csv_path=str(storage_manager.detections_csv)
            )

    def start_processing(
        self,
        video_path: str,
        conf_threshold: float = 0.10,
        barrier_state: str = "UNKNOWN",
        detection_mode: str = "hybrid",
        generate_evidence: bool = True
    ) -> bool:
        with self.lock:
            if self.status == "PROCESSING":
                return False  # Already processing
            
            self.status = "PROCESSING"
            self.stop_requested = False
            self.current_frame = 0
            self.total_frames = 0
            self.fps = 0.0
            self.active_alerts_count = 0
            self.total_detections_count = 0
            self.events_count = 0
            self.error_message = None
            self.output_video_path = None
            self.pending_evidence.clear()

        try:
            rule_engine.set_barrier_state(barrier_state)
            rule_engine.reset()
            self.tracker_manager.reset()
            self.frame_buffer.clear()

            # Launch background worker thread
            self.thread = threading.Thread(
                target=self._worker,
                args=(video_path, conf_threshold, detection_mode, generate_evidence),
                daemon=True
            )
            self.thread.start()
            return True
        except Exception as e:
            with self.lock:
                self.status = "ERROR"
                self.error_message = str(e)
            return False

    def stop_processing(self) -> bool:
        with self.lock:
            if self.status != "PROCESSING":
                return False
            self.stop_requested = True
        return True

    def _worker(self, video_path: str, conf_threshold: float, detection_mode: str, generate_evidence: bool):
        print(f"[Processor] Starting video processing on: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with self.lock:
                self.status = "ERROR"
                self.error_message = f"Failed to open video file: {video_path}"
            return

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with self.lock:
            self.total_frames = total_f
            self.fps = video_fps

        out_w, out_h = 1280, int(orig_h * (1280 / orig_w))
        if out_w % 2 != 0: out_w += 1
        if out_h % 2 != 0: out_h += 1

        out_name = f"annotated_output_{int(time.time())}.mp4"
        out_path = str(OUTPUTS_DIR / out_name)
        
        # Prepare output video writer using OpenCV mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, video_fps, (out_w, out_h))

        pre_frames = int(video_fps * settings.evidence_pre_seconds)
        post_frames = int(video_fps * settings.evidence_post_seconds)
        buf_size = max(300, int(video_fps * (settings.evidence_pre_seconds + settings.evidence_post_seconds + 5.0)))
        self.frame_buffer = RollingFrameBuffer(max_frames=buf_size)
        
        frame_idx = 0
        detections_batch = []
        batch_timer = time.time()

        try:
            while cap.isOpened():
                if self.stop_requested:
                    with self.lock:
                        self.status = "STOPPED"
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                video_timestamp = frame_idx / video_fps

                # 1. Run Detection & Tracking
                raw_detections = detector.detect_and_track(
                    frame=frame,
                    mode=detection_mode,
                    conf_threshold=conf_threshold,
                    persist=True
                )

                # 2. Update tracking state & ID consistency
                tracked_detections = self.tracker_manager.update(raw_detections, frame_idx, video_timestamp)

                # 3. Store frame in rolling buffer for evidence clips
                if generate_evidence:
                    self.frame_buffer.append(frame_idx, video_timestamp, frame, tracked_detections)

                # 4. Check polygon zones
                zones = zone_manager.zones
                zone_matches = zone_manager.check_detections(tracked_detections, orig_w, orig_h)

                # 5. Evaluate Rule Engine for violations
                new_events = rule_engine.evaluate_frame(zone_matches, tracked_detections, frame_idx, video_timestamp)
                
                # 6. Handle New Events
                for event in new_events:
                    storage_manager.save_event(event)
                    with self.lock:
                        self.events_count += 1
                        self.active_alerts_count += 1

                    if generate_evidence:
                        self.pending_evidence.append({
                            "event": event,
                            "start_frame": max(1, frame_idx - pre_frames),
                            "end_frame": frame_idx + post_frames
                        })

                # 7. Check Pending Evidence Generation
                if generate_evidence:
                    completed_evidence = []
                    for req in self.pending_evidence:
                        if frame_idx >= req["end_frame"] or frame_idx >= total_f:
                            # Time to generate evidence clip!
                            window = self.frame_buffer.get_window(req["start_frame"], req["end_frame"])
                            clip_file = evidence_generator.generate_clip(req["event"], window, zones, fps=video_fps)
                            if clip_file:
                                storage_manager.update_event_evidence(req["event"].event_id, clip_file)
                            completed_evidence.append(req)
                    
                    for req in completed_evidence:
                        self.pending_evidence.remove(req)

                # 8. Render Annotated Frame for Full Output Video
                annotated = self._render_full_frame(frame, tracked_detections, zones, video_timestamp, out_w, out_h)
                writer.write(annotated)

                # Cache frame for live surveillance feed in browser
                if frame_idx % 2 == 0 or frame_idx == 1:
                    ret_enc, enc_bytes = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ret_enc:
                        with self.lock:
                            self.latest_frame_jpeg = enc_bytes.tobytes()

                # 9. Record detections batch
                for d in tracked_detections:
                    detections_batch.append({
                        "frame": frame_idx,
                        "timestamp": round(video_timestamp, 3),
                        "track_id": d.get("track_id"),
                        "raw_class": d.get("raw_class"),
                        "mapped_class": d.get("mapped_class"),
                        "conf": round(d.get("conf", 0.0), 3),
                        "x1": round(d["box"][0], 1),
                        "y1": round(d["box"][1], 1),
                        "x2": round(d["box"][2], 1),
                        "y2": round(d["box"][3], 1),
                        "source": d.get("source")
                    })

                # Flush detections CSV in batches
                if len(detections_batch) >= 100 or (time.time() - batch_timer > 3.0):
                    storage_manager.record_detections_batch(detections_batch)
                    with self.lock:
                        self.total_detections_count += len(detections_batch)
                    detections_batch.clear()
                    batch_timer = time.time()

                with self.lock:
                    self.current_frame = frame_idx

            # Finalize any remaining evidence requests at video end
            if generate_evidence and self.pending_evidence:
                for req in self.pending_evidence:
                    window = self.frame_buffer.get_window(req["start_frame"], frame_idx)
                    clip_file = evidence_generator.generate_clip(req["event"], window, zones, fps=video_fps)
                    if clip_file:
                        storage_manager.update_event_evidence(req["event"].event_id, clip_file)
                self.pending_evidence.clear()

            # Flush remaining detections
            if detections_batch:
                storage_manager.record_detections_batch(detections_batch)
                with self.lock:
                    self.total_detections_count += len(detections_batch)

            pass

        except Exception as e:
            print(f"[Processor] Exception during processing: {e}")
            with self.lock:
                self.status = "ERROR"
                self.error_message = str(e)
        finally:
            cap.release()
            writer.release()
            
            # Convert OpenCV output video to web-ready H.264 faststart
            final_h264_name = f"detection_{out_name}"
            final_h264_path = str(OUTPUTS_DIR / final_h264_name)
            print(f"[Processor] Converting detection video to web-ready H.264: {final_h264_name}...")
            converted = self._convert_to_h264(out_path, final_h264_path)

            if converted and os.path.exists(final_h264_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                output_final = final_h264_name
            else:
                output_final = out_name

            with self.lock:
                if self.status != "STOPPED" and self.status != "ERROR":
                    self.status = "COMPLETED"
                self.output_video_path = output_final
                self.latest_frame_jpeg = None

            print(f"[Processor] Video processing finished. Output: {output_final} (Status: {self.status})")

    def _convert_to_h264(self, input_path: str, output_path: str) -> bool:
        """Encode raw output video into browser-playable H.264 via FFmpeg."""
        import subprocess
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "22",
            "-movflags", "+faststart",
            output_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
            return res.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            print(f"[Processor] H.264 conversion warning: {e}")
            return False

    def _render_full_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        zones: List[ZoneModel],
        timestamp: float,
        out_w: int,
        out_h: int
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Overlay Zones
        overlay = canvas.copy()
        for zone in zones:
            pts = np.array([[int(p.x * w), int(p.y * h)] for p in zone.points], dtype=np.int32)
            color_bgr = (0, 0, 220) if zone.zone_type == "TRACK_INTRUSION" else (0, 160, 255)
            cv2.fillPoly(overlay, [pts], color_bgr)
            cv2.polylines(canvas, [pts], True, color_bgr, 3, cv2.LINE_AA)
            if len(pts) > 0:
                cv2.putText(canvas, zone.name, (pts[0][0], max(35, pts[0][1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # 2. Detections & Tracking Trails
        for det in detections:
            box = [int(v) for v in det["box"]]
            tid = det.get("track_id")
            cls_name = det.get("mapped_class", "object")
            conf = det.get("conf", 0.0)

            # Draw tracking trail if available
            if tid is not None:
                trail = self.tracker_manager.get_track_history(tid)
                if len(trail) > 1:
                    for k in range(1, len(trail)):
                        pt1 = (int(trail[k-1][0]), int(trail[k-1][1]))
                        pt2 = (int(trail[k][0]), int(trail[k][1]))
                        cv2.line(canvas, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

            # Box and label
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            bc_x, bc_y = int(det["bottom_center"][0]), int(det["bottom_center"][1])
            cv2.circle(canvas, (bc_x, bc_y), 5, (0, 255, 255), -1)

            id_str = f"ID:{tid} " if tid is not None else ""
            label = f"{id_str}{cls_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(canvas, (box[0], max(0, box[1] - 22)), (box[0] + t_size[0] + 6, max(22, box[1])), (0, 255, 0), -1)
            cv2.putText(canvas, label, (box[0] + 3, max(16, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Resize to target output
        if (w, h) != (out_w, out_h):
            canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # 3. Top HUD Banner
        mins = int(timestamp // 60)
        secs = timestamp % 60
        hud_text = f"T225 RAILWAY MONITOR | TIME: {mins:02d}:{secs:05.2f} | BARRIER: {rule_engine.barrier_state} | ALERTS: {self.active_alerts_count}"
        cv2.putText(canvas, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

        return canvas

processor = VideoProcessor()
