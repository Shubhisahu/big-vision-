"""
track_video.py
--------------
ByteTrack via supervision — wired to RF-DETR.

FIX #3 — model.predict() return type varies across rfdetr versions.
          The `_coerce_to_sv_detections()` shim handles all known return formats
          before passing to tracker.update_with_detections().

Usage
─────
    python tracking/track_video.py --source test_video.mp4
    python tracking/track_video.py --source 0             # webcam
    python tracking/track_video.py --source test_video.mp4 --show
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2              # type: ignore
import numpy as np
import supervision as sv  # type: ignore

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
WEIGHTS     = REPO_ROOT / "backend" / "weights" / "best_checkpoint.pth"
OUT_DIR     = REPO_ROOT / "tracking"
OUT_VIDEO   = OUT_DIR / "output_tracked.mp4"
STATS_OUT   = OUT_DIR / "track_stats.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Retail-shelf-detector", "bag", "bottle", "box", "can"]

# Supervision colour palette — one colour per class
COLOURS = sv.ColorPalette.from_hex(["#808080", "#E879F9", "#FFC800", "#4ADE80", "#60A5FA"])


# ─── Return-type shim (FIX #3) ────────────────────────────────────────────────
def _coerce_to_sv_detections(raw) -> sv.Detections:
    """
    Normalise model.predict() output to supervision.Detections regardless of
    the rfdetr version.

    Known formats:
      v0.1.x → returns sv.Detections directly  ✓
      dict   → {'boxes': Tensor, 'scores': Tensor, 'labels': Tensor}
      tuple  → (boxes_xyxy, scores, labels)
    """
    if isinstance(raw, sv.Detections):
        return raw

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

    if isinstance(raw, dict):
        boxes  = _np(raw.get("boxes")  or raw.get("xyxy"))
        scores = _np(raw.get("scores") or raw.get("confidence"))
        labels = _np(raw.get("labels") or raw.get("class_id"))
    elif isinstance(raw, (list, tuple)) and len(raw) == 3:
        boxes, scores, labels = [_np(v) for v in raw]
    else:
        raise TypeError(
            f"Cannot convert type {type(raw)} to sv.Detections.\n"
            f"Inspect with: raw = model.predict(frame, threshold=0.35); print(type(raw), raw)"
        )

    if boxes is None or len(boxes) == 0:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=boxes.reshape(-1, 4).astype(float),
        confidence=scores.flatten().astype(float) if scores is not None else None,
        class_id=labels.flatten().astype(int) if labels is not None else None,
    )


# ─── Main tracking loop ───────────────────────────────────────────────────────
def run_tracking(
    source: str,
    weights: Path = WEIGHTS,
    threshold: float = 0.35,
    track_thresh: float = 0.35,   # ByteTrack: min confidence to start a track
    match_thresh: float = 0.8,    # ByteTrack: IoU threshold for association
    track_buffer: int  = 30,      # frames to keep a lost track alive
    show: bool = False,
    out_path: Path = OUT_VIDEO,
) -> dict:
    from rfdetr import RFDETRBase  # type: ignore

    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Place best_checkpoint.pth in the weights/ directory."
        )

    print(f"[INFO] Loading model: {weights}")
    model = RFDETRBase(pretrain_weights=str(weights))

    # ── Supervision annotators ────────────────────────────────────────────────
    tracker   = sv.ByteTrack(
        track_thresh=track_thresh,
        match_thresh=match_thresh,
        track_buffer=track_buffer,
    )
    box_ann   = sv.BoxAnnotator(color=COLOURS)
    label_ann = sv.LabelAnnotator(color=COLOURS, text_scale=0.5)
    trace_ann = sv.TraceAnnotator(color=COLOURS, trace_length=30,
                                  thickness=2, position=sv.Position.BOTTOM_CENTER)

    # ── Open source ───────────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # ── Per-run stats ─────────────────────────────────────────────────────────
    all_track_ids: set[int] = set()
    class_tallies: dict[str, set[int]] = defaultdict(set)   # class → set of track IDs
    id_switches = 0
    prev_id_map: dict = {}      # object_region → last track_id (rough switch heuristic)
    frame_idx = 0
    t0 = time.time()

    print(f"[INFO] Source: {source}  |  {width}×{height} @ {fps:.1f} FPS")
    print(f"       ByteTrack: track_thresh={track_thresh}, match_thresh={match_thresh}, "
          f"track_buffer={track_buffer}")
    print(f"       Output: {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detect ────────────────────────────────────────────────────────────
        raw  = model.predict(frame, threshold=threshold)
        dets = _coerce_to_sv_detections(raw)

        # ── Track ─────────────────────────────────────────────────────────────
        dets = tracker.update_with_detections(dets)

        # ── Update stats ──────────────────────────────────────────────────────
        if dets.tracker_id is not None:
            for tid, cid in zip(dets.tracker_id, dets.class_id or []):
                all_track_ids.add(tid)
                cls = CLASS_NAMES[cid] if cid is not None and cid < len(CLASS_NAMES) else "unknown"
                class_tallies[cls].add(tid)

        # ── Build labels ──────────────────────────────────────────────────────
        labels = []
        if dets.tracker_id is not None:
            for tid, cid, conf in zip(
                dets.tracker_id,
                dets.class_id or [None]*len(dets),
                dets.confidence or [0]*len(dets),
            ):
                cls  = CLASS_NAMES[cid] if cid is not None and cid < len(CLASS_NAMES) else "?"
                labels.append(f"{cls} #{tid}  {conf:.2f}")

        # ── Annotate ──────────────────────────────────────────────────────────
        annotated = frame.copy()
        annotated = trace_ann.annotate(annotated, dets)
        annotated = box_ann.annotate(annotated, dets)
        annotated = label_ann.annotate(annotated, dets, labels=labels)

        # HUD
        elapsed = time.time() - t0
        live_fps = (frame_idx + 1) / max(elapsed, 1e-6)
        hud = f"Frame {frame_idx:05d} | IDs: {len(all_track_ids)} | FPS: {live_fps:.1f}"
        cv2.rectangle(annotated, (0, 0), (len(hud) * 11, 28), (0, 0, 0), -1)
        cv2.putText(annotated, hud, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(annotated)
        if show:
            cv2.imshow("Retail Tracker", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  … {frame_idx} frames | unique IDs: {len(all_track_ids)}")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    total_time = time.time() - t0
    stats = {
        "frames":         frame_idx,
        "unique_ids":     len(all_track_ids),
        "avg_fps":        round(frame_idx / max(total_time, 1e-6), 2),
        "by_class_ids":   {k: len(v) for k, v in class_tallies.items()},
        "bytetrack_config": {
            "track_thresh": track_thresh,
            "match_thresh": match_thresh,
            "track_buffer": track_buffer,
        },
        "output_video":   str(out_path),
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2))

    print(f"\n✅ Tracking complete")
    print(f"   Frames     : {frame_idx}")
    print(f"   Unique IDs : {len(all_track_ids)}")
    print(f"   Avg FPS    : {stats['avg_fps']}")
    print(f"   By class   : {stats['by_class_ids']}")
    print(f"\n   TIP: Count ID switches by watching the output video and noting")
    print(f"   when an object's track ID changes after brief occlusion.")
    print(f"   Tune --track-thresh / --match-thresh if switches are frequent.")
    return stats


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ByteTrack + RF-DETR retail tracker")
    parser.add_argument("--source",       default="0",          help="Video file or webcam index")
    parser.add_argument("--weights",      type=Path, default=WEIGHTS)
    parser.add_argument("--threshold",    type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--track-thresh", type=float, default=0.35, help="ByteTrack track_thresh")
    parser.add_argument("--match-thresh", type=float, default=0.80, help="ByteTrack match_thresh (IoU)")
    parser.add_argument("--track-buffer", type=int,   default=30,   help="Frames to keep lost track alive")
    parser.add_argument("--show",         action="store_true",   help="Display window during processing")
    parser.add_argument("--out",          type=Path, default=OUT_VIDEO, help="Output MP4 path")
    args = parser.parse_args()

    run_tracking(
        source=args.source,
        weights=args.weights,
        threshold=args.threshold,
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        show=args.show,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
