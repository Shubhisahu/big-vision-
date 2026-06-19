"""
inference.py
────────────
FIX #7 — Intel Arc GPU support via three inference backends, tried in order:
  1. ONNX Runtime + OpenVINO execution provider  → fastest on Arc (~30-60ms)
  2. Intel Extension for PyTorch (IPEX) XPU      → Arc GPU PyTorch path
  3. CPU via PyTorch / rfdetr                     → always works (~100-200ms)

The backend is auto-detected at startup and logged.  You never need to change
this file when moving between machines.

Exports (for the FastAPI app)
─────────────────────────────
  load_model()                → call once at startup
  model_info() -> dict        → name, backend, classes, thresholds
  detect_and_annotate(frame, conf, iou) -> (DetectionResult, annotated_frame)
  bytes_to_frame(raw) -> np.ndarray | None
  frame_to_b64_jpeg(frame) -> str
"""

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2          # type: ignore
import numpy as np
import io
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
WEIGHTS_PTH = BACKEND_DIR / "weights" / "best_checkpoint.pth"
WEIGHTS_ONX = BACKEND_DIR / "weights" / "retail_detector.onnx"

CONF_THRESH = float(os.getenv("CONF_THRESH", "0.35"))
IOU_THRESH  = float(os.getenv("IOU_THRESH",  "0.45"))

CLASS_NAMES = ["Retail-shelf-detector", "bag", "bottle", "box", "can"]
CLASS_COLOURS_BGR = {
    0: (128, 128, 128),  # Retail-shelf-detector
    1: (200,   0, 255),  # bag
    2: (0,   200, 255),  # bottle
    3: (0,   255, 100),  # box
    4: (255, 160,   0),  # can
}
CLASS_COLOURS_HEX = {
    "Retail-shelf-detector": "#808080",
    "bag":    "#C800FF",
    "bottle": "#FFC800",
    "box":    "#64FF64",
    "can":    "#00A0FF",
}

# ─── State ────────────────────────────────────────────────────────────────────
_backend_name: str = "unloaded"
_model = None   # rfdetr model or None
_ort_session = None  # ONNX Runtime session or None


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int

    def to_dict(self):
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "cx": (self.x1+self.x2)//2, "cy": (self.y1+self.y2)//2}


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       BBox
    track_id:   Optional[int] = None
    colour_hex: str = "#FFFFFF"

    def to_dict(self):
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox":       self.bbox.to_dict(),
            "track_id":   self.track_id,
            "colour_hex": self.colour_hex,
        }


@dataclass
class DetectionResult:
    detections:   list[Detection] = field(default_factory=list)
    frame_w:      int = 0
    frame_h:      int = 0
    inference_ms: float = 0.0

    @property
    def count(self): return len(self.detections)

    def counts_by_class(self):
        c: dict[str, int] = {}
        for d in self.detections:
            c[d.class_name] = c.get(d.class_name, 0) + 1
        return c

    def to_dict(self):
        return {
            "count":        self.count,
            "by_class":     self.counts_by_class(),
            "inference_ms": round(self.inference_ms, 2),
            "frame_w":      self.frame_w,
            "frame_h":      self.frame_h,
            "detections":   [d.to_dict() for d in self.detections],
        }


# ─── Backend detection + loading ──────────────────────────────────────────────

def _try_openvino_onnx() -> bool:
    """
    FIX #7 path A — ONNX Runtime with OpenVINO EP.
    Fastest on Intel Arc; works on any Intel iGPU/dGPU/CPU.
    Install: pip install onnxruntime-openvino
    """
    global _ort_session, _backend_name
    if not WEIGHTS_ONX.exists():
        return False
    try:
        import onnxruntime as ort  # type: ignore
        available_eps = ort.get_available_providers()
        preferred = []
        if "OpenVINOExecutionProvider" in available_eps:
            preferred.append("OpenVINOExecutionProvider")
        if "CPUExecutionProvider" in available_eps:
            preferred.append("CPUExecutionProvider")

        _ort_session = ort.InferenceSession(
            str(WEIGHTS_ONX),
            providers=preferred,
        )
        active_ep = _ort_session.get_providers()[0]
        _backend_name = f"ONNX+{active_ep.replace('ExecutionProvider','')}"
        print(f"[OK]   ONNX Runtime backend: {_backend_name}")
        print(f"       Model: {WEIGHTS_ONX}")
        return True
    except Exception as e:
        print(f"[WARN] ONNX/OpenVINO load failed: {e}")
        return False


def _try_ipex_xpu() -> bool:
    """
    FIX #7 path B — Intel Extension for PyTorch (IPEX) on Intel Arc XPU.
    Install: pip install intel-extension-for-pytorch
    """
    global _model, _backend_name
    if not WEIGHTS_PTH.exists():
        return False
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore
        import torch
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return False
        from rfdetr import RFDETRBase  # type: ignore
        model = RFDETRBase(pretrain_weights=str(WEIGHTS_PTH))
        # Move to XPU and optimise
        model = ipex.optimize(model)
        _model = model
        _backend_name = "IPEX-XPU (Intel Arc)"
        print(f"[OK]   IPEX XPU backend active: {_backend_name}")
        return True
    except Exception as e:
        print(f"[WARN] IPEX XPU not available: {e}")
        return False


def _load_cpu_rfdetr() -> None:
    """FIX #7 path C — plain CPU inference. Always works."""
    global _model, _backend_name
    if WEIGHTS_PTH.exists():
        from rfdetr import RFDETRBase  # type: ignore
        _model = RFDETRBase(pretrain_weights=str(WEIGHTS_PTH))
        _backend_name = "CPU (rfdetr)"
        print(f"[OK]   CPU backend: rfdetr | weights: {WEIGHTS_PTH}")
    else:
        print(f"[WARN] No weights found at {WEIGHTS_PTH} or {WEIGHTS_ONX}")
        print("       Place best_checkpoint.pth in backend/weights/")
        _backend_name = "no_model"


def load_model() -> None:
    """
    Auto-detect the best available inference backend in priority order:
      1. ONNX + OpenVINO  (fastest on Intel Arc, no PyTorch needed)
      2. IPEX XPU         (Arc GPU via PyTorch)
      3. CPU rfdetr       (always works)
    """
    print("[INFO] Auto-detecting inference backend …")
    if _try_openvino_onnx():
        return
    if _try_ipex_xpu():
        return
    _load_cpu_rfdetr()

    # Warm-up pass
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        _run_inference(dummy, threshold=0.5)
        print("[OK]   Warm-up pass complete.")
    except Exception as e:
        print(f"[WARN] Warm-up failed: {e}")


def model_info() -> dict:
    return {
        "backend":     _backend_name,
        "onnx_path":   str(WEIGHTS_ONX) if WEIGHTS_ONX.exists() else None,
        "pth_path":    str(WEIGHTS_PTH) if WEIGHTS_PTH.exists() else None,
        "classes":     CLASS_NAMES,
        "conf_thresh": CONF_THRESH,
        "iou_thresh":  IOU_THRESH,
    }


# ─── Core inference ───────────────────────────────────────────────────────────

def _run_inference(frame: np.ndarray, threshold: float) -> list[dict]:
    """
    Dispatch to the active backend.  Returns a list of raw detection dicts:
    [{"box": [x1,y1,x2,y2], "score": float, "label": int}, ...]
    """
    if _ort_session is not None:
        return _infer_onnx(frame, threshold)
    if _model is not None:
        return _infer_rfdetr(frame, threshold)
    raise RuntimeError("No model loaded. Call load_model() first.")


def _infer_onnx(frame: np.ndarray, threshold: float) -> list[dict]:
    """ONNX Runtime inference path."""
    # Preprocess: BGR→RGB, resize to 640, normalise
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (640, 640)).astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None]  # NCHW

    ort_inputs = {_ort_session.get_inputs()[0].name: inp}
    outs = _ort_session.run(None, ort_inputs)

    # Output format depends on the export; try common layouts
    # Layout A: [boxes (N,4), scores (N,), labels (N,)]
    # Layout B: single tensor (N, 6) with [x1,y1,x2,y2,score,label]
    h, w = frame.shape[:2]
    results = []

    if len(outs) >= 3:
        boxes, scores, labels = outs[0], outs[1], outs[2]
        for box, score, label in zip(boxes, scores, labels):
            if float(score) < threshold:
                continue
            x1, y1, x2, y2 = box.flatten()[:4]
            # Scale from 640→original
            results.append({
                "box":   [int(x1*w/640), int(y1*h/640), int(x2*w/640), int(y2*h/640)],
                "score": float(score),
                "label": int(label),
            })
    elif len(outs) == 1:
        for det in outs[0].reshape(-1, 6):
            score = float(det[4])
            if score < threshold:
                continue
            x1, y1, x2, y2 = det[:4]
            results.append({
                "box":   [int(x1*w/640), int(y1*h/640), int(x2*w/640), int(y2*h/640)],
                "score": score,
                "label": int(det[5]),
            })
    return results


def _infer_rfdetr(frame: np.ndarray, threshold: float) -> list[dict]:
    """rfdetr model.predict() path — handles return-type variation."""
    raw = _model.predict(frame, threshold=threshold)
    return _sv_detections_to_dicts(raw, frame.shape)


def _sv_detections_to_dicts(raw, shape) -> list[dict]:
    """Convert any rfdetr output to a normalised list of dicts."""
    import supervision as sv  # type: ignore

    def _np(x):
        if x is None: return None
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(x)

    if isinstance(raw, sv.Detections):
        dets = raw
    elif isinstance(raw, dict):
        boxes  = _np(raw.get("boxes") or raw.get("xyxy"))
        scores = _np(raw.get("scores") or raw.get("confidence"))
        labels = _np(raw.get("labels") or raw.get("class_id"))
        if boxes is None or len(boxes) == 0:
            return []
        dets = sv.Detections(
            xyxy=boxes.reshape(-1, 4).astype(float),
            confidence=scores.flatten().astype(float) if scores is not None else None,
            class_id=labels.flatten().astype(int) if labels is not None else None,
        )
    elif isinstance(raw, (list, tuple)) and len(raw) == 3:
        boxes, scores, labels = [_np(v) for v in raw]
        if boxes is None or len(boxes) == 0:
            return []
        dets = sv.Detections(
            xyxy=boxes.reshape(-1, 4).astype(float),
            confidence=scores.flatten().astype(float) if scores is not None else None,
            class_id=labels.flatten().astype(int) if labels is not None else None,
        )
    else:
        return []

    results = []
    for i in range(len(dets)):
        box  = dets.xyxy[i]
        conf = float(dets.confidence[i]) if dets.confidence is not None else 1.0
        cid  = int(dets.class_id[i])    if dets.class_id   is not None else 0
        results.append({
            "box":   [int(v) for v in box],
            "score": conf,
            "label": cid,
        })
    return results


# ─── Public helpers ───────────────────────────────────────────────────────────

def detect_and_annotate(
    frame: np.ndarray,
    conf: float = CONF_THRESH,
    iou:  float = IOU_THRESH,
) -> tuple["DetectionResult", Optional[np.ndarray]]:
    h, w = frame.shape[:2]
    t0 = time.perf_counter()
    raw_dets = _run_inference(frame, threshold=conf)
    elapsed  = (time.perf_counter() - t0) * 1000

    detections = []
    for d in raw_dets:
        x1, y1, x2, y2 = d["box"]
        cid   = d["label"]
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        detections.append(Detection(
            class_id=cid, class_name=cname,
            confidence=d["score"],
            bbox=BBox(x1, y1, x2, y2),
            colour_hex=CLASS_COLOURS_HEX.get(cname, "#fff"),
        ))

    result = DetectionResult(
        detections=detections, frame_w=w, frame_h=h, inference_ms=elapsed,
    )
    annotated = _draw_boxes(frame.copy(), detections)
    return result, annotated


def _draw_boxes(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for det in detections:
        b      = det.bbox
        colour = CLASS_COLOURS_BGR.get(det.class_id, (128, 128, 128))
        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), colour, 2)
        label  = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(frame, (b.x1, b.y1 - th - 10), (b.x1 + tw + 6, b.y1), colour, -1)
        cv2.putText(frame, label, (b.x1 + 3, b.y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def bytes_to_frame(raw: bytes) -> Optional[np.ndarray]:
    # Standard OpenCV decoding for JPEG, PNG, etc.
    # Frontend guarantees HEIC files are pre-converted to JPEG before reaching us.
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def frame_to_b64_jpeg(frame: np.ndarray, quality: int = 82) -> str:
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode()
