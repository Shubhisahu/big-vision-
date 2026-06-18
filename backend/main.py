"""
main.py — FastAPI backend
─────────────────────────
FIX #6 — Added SlowAPI rate limiting to prevent CPU abuse on public deployments.
          Also includes CORS, lifespan model loading, and WebSocket streaming.

Endpoints
─────────
  GET  /health          — model status + uptime
  GET  /metrics         — session + request counters
  POST /infer           — single image → JSON detections + annotated base64
  POST /infer-frame     — webcam frame (stateful ByteTrack per session)
  POST /infer-video     — upload MP4 → returns tracked MP4 download URL
  GET  /videos/{file}   — serves processed video files
  WS   /ws/{session_id} — real-time WebSocket (alternative to polling /infer-frame)

Run
───
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
# Guarantee the backend directory is in the Python path so imports always work
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2              # type: ignore
import numpy as np
from fastapi import (
    FastAPI, File, HTTPException, Query, Request,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── FIX #6: Rate limiting ─────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
from slowapi.errors import RateLimitExceeded               # type: ignore
from slowapi.util import get_remote_address                # type: ignore

from inference import (
    bytes_to_frame, detect_and_annotate, frame_to_b64_jpeg,
    load_model, model_info, CONF_THRESH, IOU_THRESH,
)
from tracker import (
    active_sessions, get_or_create_session, remove_session,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
VIDEO_DIR  = REPO_ROOT / "backend" / "processed_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# ─── Global counters ──────────────────────────────────────────────────────────
_counters = {"images": 0, "videos": 0, "ws_frames": 0}
_start_time = time.time()

# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading RF-DETR model ...")
    load_model()
    print("[STARTUP] Model ready [OK]")
    yield
    print("[SHUTDOWN] Cleanup complete.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShelfSight — Retail Object Detector",
    description=(
        "RF-DETR fine-tuned on a retail-shelf dataset "
        "(bottle / box / can / bag) with ByteTrack multi-object tracking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Lock this down to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")


# ─── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
@limiter.limit("30/minute")
async def health(request: Request):
    return {
        "status":      "ok",
        "uptime_s":    round(time.time() - _start_time, 1),
        "model":       model_info(),
        "ws_sessions": len(active_sessions()),
    }


@app.get("/metrics", tags=["System"])
@limiter.limit("30/minute")
async def metrics(request: Request):
    return {
        **_counters,
        "active_sessions": active_sessions(),
        "uptime_s": round(time.time() - _start_time, 1),
    }


# ─── Image inference ──────────────────────────────────────────────────────────
@app.post("/infer", tags=["Inference"])
@limiter.limit("10/minute")          # FIX #6: 10 image inferences per IP per minute
async def infer(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(default=CONF_THRESH, ge=0.01, le=1.0),
    iou:  float = Query(default=IOU_THRESH,  ge=0.01, le=1.0),
):
    """Single image → JSON detections + base64-encoded annotated image."""
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:   # 20 MB guard
        raise HTTPException(413, "Image too large (max 20 MB).")

    frame = bytes_to_frame(raw)
    if frame is None:
        raise HTTPException(400, "Could not decode image. Send a valid JPG/PNG.")

    result, annotated = await asyncio.get_event_loop().run_in_executor(
        None, detect_and_annotate, frame, conf, iou,
    )
    _counters["images"] += 1

    response = result.to_dict()
    if annotated is not None:
        response["annotated_image"] = frame_to_b64_jpeg(annotated)
    return JSONResponse(content=response)


# ─── Webcam frame inference (stateful tracking) ───────────────────────────────
@app.post("/infer-frame", tags=["Inference"])
@limiter.limit("600/minute")         # webcam sends ~5 req/s — allow generous burst
async def infer_frame(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Query(default="default"),
    conf: float = Query(default=CONF_THRESH),
    iou:  float = Query(default=IOU_THRESH),
):
    """
    Webcam frame → JSON detections with stable ByteTrack IDs.
    Each session_id maintains independent tracker state.
    """
    raw = await file.read()
    frame = bytes_to_frame(raw)
    if frame is None:
        raise HTTPException(400, "Could not decode frame.")

    tracker = get_or_create_session(session_id, conf=conf, iou=iou)
    tracked = await asyncio.get_event_loop().run_in_executor(
        None, tracker.update, frame,
    )
    _counters["ws_frames"] += 1
    return JSONResponse(content={
        "session_id":  session_id,
        "count":       len(tracked),
        "detections":  [t.to_dict() for t in tracked],
    })


# ─── Video inference ──────────────────────────────────────────────────────────
@app.post("/infer-video", tags=["Inference"])
@limiter.limit("3/minute")           # FIX #6: video processing is expensive
async def infer_video(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(default=CONF_THRESH),
    iou:  float = Query(default=IOU_THRESH),
):
    """Upload MP4 → process with ByteTrack → return annotated video URL."""
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, f"Unsupported video format: {suffix}")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    if os.path.getsize(tmp_path) > 200 * 1024 * 1024:
        os.unlink(tmp_path)
        raise HTTPException(413, "Video too large (max 200 MB).")

    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = VIDEO_DIR / out_name

    try:
        stats = await asyncio.get_event_loop().run_in_executor(
            None, _process_video_blocking, tmp_path, str(out_path), conf, iou,
        )
    finally:
        os.unlink(tmp_path)

    _counters["videos"] += 1
    return JSONResponse(content={
        "status":    "complete",
        "video_url": f"/videos/{out_name}",
        "stats":     stats,
    })


def _process_video_blocking(src: str, dst: str, conf: float, iou: float) -> dict:
    """Run in thread executor so the event loop stays responsive."""
    from rfdetr import RFDETRBase  # type: ignore
    import supervision as sv       # type: ignore
    from tracker import _coerce_to_sv_detections  # reuse shim

    weights = Path(__file__).resolve().parent / "weights" / "best_checkpoint.pth"
    model   = RFDETRBase(pretrain_weights=str(weights) if weights.exists() else None)
    tracker = sv.ByteTrack()
    box_ann = sv.BoxAnnotator()
    lbl_ann = sv.LabelAnnotator(text_scale=0.5)

    cap     = cv2.VideoCapture(src)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h    = int(cap.get(3)), int(cap.get(4))
    writer  = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n_frames = 0
    n_dets   = 0
    CLASS_NAMES = ["Retail-shelf-detector", "bag", "bottle", "box", "can"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw  = model.predict(frame, threshold=conf)
        dets = _coerce_to_sv_detections(raw)
        dets = tracker.update_with_detections(dets)
        n_dets += len(dets)

        labels = [
            f"{CLASS_NAMES[cid]} #{tid}"
            for cid, tid in zip(dets.class_id or [], dets.tracker_id or [])
        ]
        annotated = box_ann.annotate(frame.copy(), dets)
        annotated = lbl_ann.annotate(annotated, dets, labels=labels)
        writer.write(annotated)
        n_frames += 1

    cap.release()
    writer.release()
    return {"frames": n_frames, "total_detections": n_dets}


# ─── WebSocket streaming ──────────────────────────────────────────────────────
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-time detection + tracking via WebSocket.
    Client sends raw JPEG bytes; server replies with JSON detections.
    """
    await websocket.accept()
    tracker = get_or_create_session(session_id)
    frame_idx = 0
    print(f"[WS] Session {session_id} connected.")
    try:
        while True:
            raw = await websocket.receive_bytes()
            frame = bytes_to_frame(raw)
            if frame is None:
                await websocket.send_json({"error": "bad frame"})
                continue
            tracked = await asyncio.get_event_loop().run_in_executor(
                None, tracker.update, frame,
            )
            await websocket.send_json({
                "frame_idx":  frame_idx,
                "count":      len(tracked),
                "detections": [t.to_dict() for t in tracked],
            })
            frame_idx += 1
            _counters["ws_frames"] += 1
    except WebSocketDisconnect:
        print(f"[WS] Session {session_id} disconnected after {frame_idx} frames.")
    finally:
        remove_session(session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
