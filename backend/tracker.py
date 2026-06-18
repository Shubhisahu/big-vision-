"""
tracker.py
──────────
Per-session ByteTrack state manager for the FastAPI backend.

Exports (used by main.py)
──────────────────────────
  get_or_create_session(session_id, conf, iou) → SessionTracker
  remove_session(session_id)
  active_sessions() → dict
  _coerce_to_sv_detections(raw) → sv.Detections   (shared shim)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import supervision as sv  # type: ignore

from inference import (
    CONF_THRESH, IOU_THRESH, CLASS_NAMES, CLASS_COLOURS_HEX,
    _run_inference,
)


# ─── Detection wrapper (re-exported for main.py) ──────────────────────────────
@dataclass
class TrackedDetection:
    class_id:   int
    class_name: str
    confidence: float
    track_id:   int
    x1: int; y1: int; x2: int; y2: int
    colour_hex: str = "#FFFFFF"

    def to_dict(self):
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "track_id":   self.track_id,
            "bbox": {
                "x1": self.x1, "y1": self.y1,
                "x2": self.x2, "y2": self.y2,
                "cx": (self.x1 + self.x2) // 2,
                "cy": (self.y1 + self.y2) // 2,
            },
            "colour_hex": self.colour_hex,
        }


# ─── Return-type shim ─────────────────────────────────────────────────────────
def _coerce_to_sv_detections(raw) -> sv.Detections:
    """
    Normalise model.predict() output to supervision.Detections.
    Handles all known rfdetr return formats.
    """
    def _np(x):
        if x is None:
            return None
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(x)

    if isinstance(raw, sv.Detections):
        return raw

    if isinstance(raw, dict):
        boxes  = _np(raw.get("boxes")  or raw.get("xyxy"))
        scores = _np(raw.get("scores") or raw.get("confidence"))
        labels = _np(raw.get("labels") or raw.get("class_id"))
    elif isinstance(raw, (list, tuple)) and len(raw) == 3:
        boxes, scores, labels = [_np(v) for v in raw]
    else:
        return sv.Detections.empty()

    if boxes is None or len(boxes) == 0:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=boxes.reshape(-1, 4).astype(float),
        confidence=scores.flatten().astype(float) if scores is not None else None,
        class_id=labels.flatten().astype(int)     if labels is not None else None,
    )


# ─── Per-session tracker ──────────────────────────────────────────────────────
class SessionTracker:
    """Wraps a supervision ByteTrack instance for one webcam/WS session."""

    def __init__(self, conf: float = CONF_THRESH, iou: float = IOU_THRESH):
        self.conf    = conf
        self.iou     = iou
        self.tracker = sv.ByteTrack()

    def update(self, frame: np.ndarray) -> list[TrackedDetection]:
        """Run inference + tracking on one frame. Returns tracked detections."""
        raw_dets = _run_inference(frame, threshold=self.conf)

        # Convert raw dicts → sv.Detections
        if raw_dets:
            boxes  = np.array([[d["box"][0], d["box"][1], d["box"][2], d["box"][3]] for d in raw_dets], dtype=float)
            scores = np.array([d["score"] for d in raw_dets], dtype=float)
            labels = np.array([d["label"] for d in raw_dets], dtype=int)
            sv_dets = sv.Detections(xyxy=boxes, confidence=scores, class_id=labels)
        else:
            sv_dets = sv.Detections.empty()

        # ByteTrack update
        tracked = self.tracker.update_with_detections(sv_dets)

        results: list[TrackedDetection] = []
        if tracked.tracker_id is None:
            return results

        for i in range(len(tracked)):
            cid  = int(tracked.class_id[i])  if tracked.class_id  is not None else 0
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            tid  = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[i]]
            cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"

            results.append(TrackedDetection(
                class_id=cid, class_name=cname,
                confidence=conf, track_id=tid,
                x1=x1, y1=y1, x2=x2, y2=y2,
                colour_hex=CLASS_COLOURS_HEX.get(cname, "#ffffff"),
            ))

        return results


# ─── Session registry ─────────────────────────────────────────────────────────
_sessions: dict[str, SessionTracker] = {}
_lock = threading.Lock()


def get_or_create_session(
    session_id: str,
    conf: float = CONF_THRESH,
    iou:  float = IOU_THRESH,
) -> SessionTracker:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = SessionTracker(conf=conf, iou=iou)
        return _sessions[session_id]


def remove_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def active_sessions() -> dict:
    with _lock:
        return {sid: {"conf": t.conf, "iou": t.iou} for sid, t in _sessions.items()}
