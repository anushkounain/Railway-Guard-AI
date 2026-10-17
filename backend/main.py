"""
FastAPI Application Entry Point for T225 Railway Safety System.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Response, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import cv2

from backend.config import (
    settings,
    FRONTEND_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
    EVIDENCE_DIR,
    MODELS_DIR,
    DEFAULT_DOWNLOAD_MODEL
)
from backend.evidence import FFMPEG_EXE
from backend.schemas import (
    ZoneModel,
    ZoneListPayload,
    StartProcessingPayload,
    ProcessingStatusResponse,
    ModelInfoResponse,
    EventItem
)
from backend.detector import detector
from backend.zones import zone_manager
from backend.rules import rule_engine
from backend.storage import storage_manager
from backend.processor import processor

app = FastAPI(
    title="T225 Railway Level-Crossing & Track Intrusion Safety System",
    description="End-to-end computer-vision surveillance, violation detection, and evidence vault.",
    version="1.0.0"
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# System & Model Endpoints
# -------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "T225-Railway-Safety",
        "processor_status": processor.status,
        "barrier_state": rule_engine.barrier_state
    }

@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info():
    info = detector.get_model_info()
    return ModelInfoResponse(**info)

# -------------------------------------------------------------
# Video Management Endpoints
# -------------------------------------------------------------

def stream_file_range(file_path: Path, request: Request):
    """Serve video files with HTTP 206 Partial Content via Starlette's native FileResponse."""
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found.")
    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )

@app.get("/api/videos")
def list_videos():
    """List available videos from data folder (newest first) and Downloads."""
    videos = []
    # Check data/ folder first (sorted newest first)
    data_files = sorted(list(DATA_DIR.glob("*.mp4")), key=lambda x: x.stat().st_mtime, reverse=True)
    for f in data_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        videos.append({
            "name": f.name,
            "path": str(f),
            "stream_url": f"/api/videos/stream?name={f.name}",
            "size_mb": round(size_mb, 1),
            "source": "Uploaded"
        })

    # Check Downloads folder for user video
    dl_video = r"C:\Users\91636\Downloads\18828999-uhd_4096_2160_60fps.mp4"
    if os.path.exists(dl_video) and not any(v["name"] == Path(dl_video).name for v in videos):
        size_mb = os.path.getsize(dl_video) / (1024 * 1024)
        videos.append({
            "name": "18828999-uhd_4096_2160_60fps.mp4",
            "path": dl_video,
            "stream_url": f"/api/videos/stream?path={dl_video}",
            "size_mb": round(size_mb, 1),
            "source": "Downloads"
        })
    return {"videos": videos}

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a new video file securely and ensure web-compatible progressive faststart."""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Please upload MP4, AVI, MOV, or MKV."
        )

    # Sanitize filename
    safe_name = Path(file.filename).name
    target_path = DATA_DIR / safe_name
    temp_target = DATA_DIR / f"temp_{safe_name}"

    try:
        with open(temp_target, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Optimize with FFmpeg for progressive browser playback (+faststart)
    try:
        res = subprocess.run([
            FFMPEG_EXE, "-y", "-i", str(temp_target),
            "-c", "copy", "-movflags", "+faststart",
            str(target_path)
        ], capture_output=True, timeout=60)
        if res.returncode == 0 and target_path.is_file():
            temp_target.unlink(missing_ok=True)
        else:
            temp_target.replace(target_path)
    except Exception:
        if temp_target.is_file() and not target_path.is_file():
            temp_target.replace(target_path)

    # Inspect video specs
    cap = cv2.VideoCapture(str(target_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0.0
    cap.release()

    return {
        "message": "Video uploaded successfully",
        "filename": safe_name,
        "path": str(target_path),
        "stream_url": f"/api/videos/stream?name={safe_name}",
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "total_frames": frames,
        "duration_sec": round(duration, 2)
    }

@app.api_route("/api/videos/stream", methods=["GET", "HEAD"])
def stream_video(request: Request, path: Optional[str] = None, name: Optional[str] = None):
    """Stream raw or uploaded surveillance video with Range headers for HTML5 video seeking."""
    target_path = None
    if name:
        target_path = DATA_DIR / Path(name).name
    elif path:
        p = Path(path)
        if p.is_file():
            target_path = p
        else:
            cand1 = DATA_DIR / p.name
            cand2 = Path(r"C:\Users\91636\Downloads") / p.name
            if cand1.is_file():
                target_path = cand1
            elif cand2.is_file():
                target_path = cand2

    if name or path:
        if not target_path or not target_path.is_file():
            raise HTTPException(status_code=404, detail="Requested video file not found.")
    else:
        mp4s = list(DATA_DIR.glob("*.mp4"))
        if mp4s:
            target_path = mp4s[0]
        else:
            dl = Path(r"C:\Users\91636\Downloads\18828999-uhd_4096_2160_60fps.mp4")
            if dl.is_file():
                target_path = dl
            else:
                raise HTTPException(status_code=404, detail="No video file found to stream.")

    return stream_file_range(target_path, request)

@app.get("/api/processing/live-frame")
def get_live_frame():
    """Retrieve the latest real-time frame with detection overlays during active video processing."""
    frame_bytes = processor.get_latest_frame()
    if not frame_bytes:
        raise HTTPException(status_code=404, detail="No live frame currently available.")
    return Response(content=frame_bytes, media_type="image/jpeg")

@app.get("/api/video-frame")
def get_video_frame(video_path: Optional[str] = None, frame_number: int = 0):
    """Extract a single frame image from a video for zone configuration previews."""
    target_path = video_path or r"C:\Users\91636\Downloads\18828999-uhd_4096_2160_60fps.mp4"
    if not os.path.exists(target_path):
        # Fallback to any file in data
        mp4s = list(DATA_DIR.glob("*.mp4"))
        if mp4s:
            target_path = str(mp4s[0])
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found at {target_path}")

    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot decode video.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Failed to read requested frame.")

    # Resize preview frame for quick frontend loading (1280px wide)
    orig_h, orig_w = frame.shape[:2]
    preview_w = 1280
    preview_h = int(orig_h * (preview_w / orig_w))
    resized = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

    ret, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        raise HTTPException(status_code=500, detail="Frame encoding failed.")

    return Response(content=encoded.tobytes(), media_type="image/jpeg")

# -------------------------------------------------------------
# Video Processing Control Endpoints
# -------------------------------------------------------------

@app.post("/api/processing/start")
def start_processing(payload: StartProcessingPayload):
    # Resolve video path
    vpath = payload.video_path
    if not vpath or not os.path.exists(vpath):
        # Default to downloaded test video
        dl_video = r"C:\Users\91636\Downloads\18828999-uhd_4096_2160_60fps.mp4"
        if os.path.exists(dl_video):
            vpath = dl_video
        else:
            mp4s = list(DATA_DIR.glob("*.mp4"))
            if mp4s:
                vpath = str(mp4s[0])
            else:
                raise HTTPException(status_code=400, detail="No video specified and default video not found.")

    started = processor.start_processing(
        video_path=vpath,
        conf_threshold=payload.conf_threshold or 0.10,
        barrier_state=payload.barrier_state or "UNKNOWN",
        detection_mode=payload.detection_mode or "hybrid",
        generate_evidence=payload.generate_evidence if payload.generate_evidence is not None else True
    )

    if not started:
        raise HTTPException(status_code=409, detail="A video processing task is already running.")

    return {"message": "Processing started successfully", "video_path": vpath}

@app.post("/api/processing/stop")
def stop_processing():
    stopped = processor.stop_processing()
    return {"message": "Processing stop requested" if stopped else "No active processing job"}

@app.get("/api/processing/status", response_model=ProcessingStatusResponse)
def get_processing_status():
    return processor.get_status()

# -------------------------------------------------------------
# Zone Configuration Endpoints
# -------------------------------------------------------------

@app.get("/api/zones", response_model=List[ZoneModel])
def get_zones():
    return zone_manager.zones

@app.put("/api/zones")
def update_zones(payload: ZoneListPayload):
    try:
        zone_manager.save_zones(payload.zones)
        return {"message": f"Successfully updated {len(payload.zones)} zones", "zones": zone_manager.zones}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update zones: {e}")

# -------------------------------------------------------------
# Barrier State Controls
# -------------------------------------------------------------

@app.put("/api/barrier-state")
def set_barrier_state(state: str = Query(..., pattern="^(OPEN|CLOSED|UNKNOWN)$")):
    rule_engine.set_barrier_state(state)
    return {"message": f"Barrier state updated to {rule_engine.barrier_state}", "state": rule_engine.barrier_state}

# -------------------------------------------------------------
# Event & Evidence Retrieval Endpoints
# -------------------------------------------------------------

@app.get("/api/events")
def get_events(
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    object_class: Optional[str] = None,
    limit: int = 100
):
    events = storage_manager.get_events(
        event_type=event_type,
        severity=severity,
        object_class=object_class,
        limit=limit
    )
    return {"events": events, "count": len(events)}

@app.get("/api/events/{event_id}")
def get_event(event_id: str):
    event = storage_manager.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@app.get("/api/evidence")
def list_evidence():
    """List all generated evidence video files with size and timestamp."""
    clips = []
    for f in EVIDENCE_DIR.glob("*.mp4"):
        stat = f.stat()
        clips.append({
            "filename": f.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "created_at": stat.st_mtime
        })
    clips.sort(key=lambda x: x["created_at"], reverse=True)
    return {"evidence_clips": clips, "count": len(clips)}

@app.api_route("/api/evidence/{filename}", methods=["GET", "HEAD"])
def stream_evidence_clip(filename: str, request: Request):
    """Stream evidence video clip with safe path validation and Range seeking."""
    safe_name = Path(filename).name
    file_path = EVIDENCE_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Evidence clip not found")
    return stream_file_range(file_path, request)

@app.api_route("/api/outputs/video/{filename}", methods=["GET", "HEAD"])
def stream_output_video(filename: str, request: Request):
    """Stream browser-compatible H.264 annotated output video with Range seeking."""
    safe_name = Path(filename).name
    file_path = OUTPUTS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found.")
    return stream_file_range(file_path, request)

@app.get("/api/outputs/latest-video")
def get_latest_output_video():
    """Retrieve the latest annotated detection video path and metadata."""
    if processor.output_video_path:
        p = OUTPUTS_DIR / processor.output_video_path
        if p.exists():
            return {
                "available": True,
                "filename": processor.output_video_path,
                "url": f"/api/outputs/video/{processor.output_video_path}",
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2)
            }
    outputs = sorted(list(OUTPUTS_DIR.glob("*.mp4")), key=lambda x: x.stat().st_mtime, reverse=True)
    if outputs:
        latest = outputs[0]
        return {
            "available": True,
            "filename": latest.name,
            "url": f"/api/outputs/video/{latest.name}",
            "size_mb": round(latest.stat().st_size / (1024 * 1024), 2)
        }
    return {"available": False, "filename": None, "url": None}

@app.get("/api/outputs/csv")
def download_detections_csv():
    if not storage_manager.detections_csv.exists():
        raise HTTPException(status_code=404, detail="No detections CSV generated yet.")
    return FileResponse(storage_manager.detections_csv, media_type="text/csv", filename="detections.csv")

@app.get("/api/outputs/events-csv")
def download_events_csv():
    if not storage_manager.events_csv.exists():
        raise HTTPException(status_code=404, detail="No events CSV generated yet.")
    return FileResponse(storage_manager.events_csv, media_type="text/csv", filename="events.csv")

# -------------------------------------------------------------
# Static Frontend Serving
# -------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
