"""
evaluate.py
-----------
FIX #2 — Complete evaluation with the inference loop that was missing from the
original plan. Also fixes FIX #2b: uses threshold=0.0 when computing mAP so
the full precision-recall curve is captured.

Usage
─────
    python evaluation/evaluate.py
    python evaluation/evaluate.py --weights backend/weights/best_checkpoint.pth
    python evaluation/evaluate.py --split val   # quick check on val set
"""

import argparse
import json
import os
import time
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "backend" / "weights"
DATA_DIR    = REPO_ROOT / "data"
EVAL_DIR    = REPO_ROOT / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Retail-shelf-detector", "bag", "bottle", "box", "can"]


def load_model(weights_path: Path):
    from rfdetr import RFDETRBase  # type: ignore
    if not weights_path.exists():
        fallback = REPO_ROOT / "backend" / "weights" / weights_path.name
        if fallback.exists():
            weights_path = fallback
        else:
            raise FileNotFoundError(
                f"Weights not found: {weights_path} or {fallback}\n"
                "Download best_checkpoint.pth from your Kaggle output panel and "
                "place it in the weights/ directory."
            )
    print(f"[INFO] Loading model from: {weights_path}")
    model = RFDETRBase(pretrain_weights=str(weights_path))
    print("[INFO] Optimizing model for inference...")
    model.optimize_for_inference()
    print("[OK]   Model loaded and optimized.")
    return model


def run_inference_loop(model, ds, threshold: float = 0.0):
    """
    FIX #2 — This is the loop that was absent from the original plan.

    threshold=0.0 is intentional for mAP computation:
    mAP is computed across the full P-R curve by varying the score threshold.
    A high threshold collapses recall to near-zero at the low end of the curve,
    which makes mAP look artificially worse than it is.
    """
    import supervision as sv  # type: ignore

    predictions: list[sv.Detections] = []
    targets:     list[sv.Detections] = []
    latencies:   list[float]         = []

    print(f"\n[INFO] Running inference over {len(ds)} test images "
          f"(threshold={threshold}) ...")

    for i, (path, image, annotation) in enumerate(ds):
        t0 = time.perf_counter()

        # ── model.predict() return-type shim (FIX #3 applied here) ──────────
        raw = model.predict(image, threshold=threshold)
        pred = _coerce_to_sv_detections(raw)

        latencies.append((time.perf_counter() - t0) * 1000)
        predictions.append(pred)
        targets.append(annotation)

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(ds)} done")

    return predictions, targets, latencies


def _coerce_to_sv_detections(raw) -> "sv.Detections":
    """
    FIX #3 — model.predict() return type varies across rfdetr versions.
    This shim normalises any return format into a supervision.Detections object.

    Tested against:
      - rfdetr 0.1.x → returns sv.Detections natively ✓
      - rfdetr dict  → returns {'boxes': ..., 'scores': ..., 'labels': ...}
      - custom tuple → (boxes_xyxy, scores, labels)
    """
    import supervision as sv  # type: ignore
    import numpy as np

    if isinstance(raw, sv.Detections):
        return raw                                   # already correct

    if isinstance(raw, dict):
        boxes  = _to_numpy(raw.get('boxes')  or raw.get('xyxy'))
        scores = _to_numpy(raw.get('scores') or raw.get('confidence'))
        labels = _to_numpy(raw.get('labels') or raw.get('class_id'))
        if boxes is None or len(boxes) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=boxes.reshape(-1, 4).astype(float),
            confidence=scores.flatten().astype(float),
            class_id=labels.flatten().astype(int),
        )

    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        boxes, scores, labels = raw
        boxes  = _to_numpy(boxes)
        scores = _to_numpy(scores)
        labels = _to_numpy(labels)
        if boxes is None or len(boxes) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=boxes.reshape(-1, 4).astype(float),
            confidence=scores.flatten().astype(float),
            class_id=labels.flatten().astype(int),
        )

    raise TypeError(
        f"Cannot convert model.predict() output of type {type(raw)} "
        "to supervision.Detections.\n"
        "Inspect the raw output with: print(type(raw), raw)"
    )


def _to_numpy(x):
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    import numpy as np
    return np.asarray(x)


def compute_metrics(predictions, targets, ds):
    import supervision as sv  # type: ignore
    import numpy as np

    print("\n[INFO] Computing mAP ...")
    mean_ap = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )

    print("\n+---------------------------------------+")
    print("|      Detection Metrics - Test Set     |")
    print("+---------------+-----------------------+")
    print(f"| mAP@50        |  {mean_ap.map50:.4f}              |")
    print(f"| mAP@50-95     |  {mean_ap.map50_95:.4f}              |")
    print("+---------------+-----------------------+")

    results = {
        "mAP50":    round(float(mean_ap.map50), 4),
        "mAP50_95": round(float(mean_ap.map50_95), 4),
    }

    # Per-class breakdown (if available)
    if hasattr(mean_ap, 'per_class_ap50'):
        print("\nPer-class AP@50:")
        per_class = {}
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            ap = float(mean_ap.per_class_ap50[cls_id]) if cls_id < len(mean_ap.per_class_ap50) else 0.0
            bar = "=" * int(ap * 30)
            print(f"  {cls_name:<25s}: {ap:.4f}  {bar}")
            per_class[cls_name] = round(ap, 4)
        results["per_class_ap50"] = per_class

    return results


def compute_latency(latencies):
    import numpy as np
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    print(f"\n[INFO] Latency over {len(latencies)} images:")
    print(f"  p50 : {p50:.0f} ms")
    print(f"  p95 : {p95:.0f} ms")
    print(f"  mean: {np.mean(latencies):.0f} ms")
    return {"p50_ms": round(p50, 1), "p95_ms": round(p95, 1)}


def plot_confusion_matrix(predictions, targets, ds, save_path: Path):
    import supervision as sv  # type: ignore

    print("\n[INFO] Generating confusion matrix ...")
    cm = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=CLASS_NAMES,
    )
    cm.plot(save_path=str(save_path))
    print(f"[OK]   Confusion matrix saved -> {save_path}")
    return cm


def find_hard_failures(test_images, predictions, targets, save_dir: Path,
                       conf_threshold: float = 0.6):
    """
    FIX #2 (failure analysis) — Find high-confidence wrong predictions.
    These are more instructive than low-confidence misses.

    For each high-conf wrong prediction: saves a side-by-side annotated crop
    showing GT box (green) and predicted box (red).
    """
    import cv2  # type: ignore
    import numpy as np
    import supervision as sv  # type: ignore

    save_dir.mkdir(parents=True, exist_ok=True)
    failures_found = 0

    for i, (path, gt, pred) in enumerate(zip(test_images, targets, predictions)):
        if pred.class_id is None or gt.class_id is None:
            continue
        if len(pred) == 0 or len(gt) == 0:
            continue

        # Filter predictions by confidence
        if pred.confidence is not None:
            high_conf_mask = pred.confidence > conf_threshold
            pred_xyxy = pred.xyxy[high_conf_mask]
            pred_cls = pred.class_id[high_conf_mask]
            pred_conf = pred.confidence[high_conf_mask]
        else:
            pred_xyxy = pred.xyxy
            pred_cls = pred.class_id
            pred_conf = np.ones(len(pred))

        if len(pred_xyxy) == 0:
            continue

        gt_xyxy = gt.xyxy
        gt_cls = gt.class_id

        hard_wrong_indices = []

        # Find IoU between high-conf preds and GTs
        for j, p_box in enumerate(pred_xyxy):
            xA = np.maximum(p_box[0], gt_xyxy[:, 0])
            yA = np.maximum(p_box[1], gt_xyxy[:, 1])
            xB = np.minimum(p_box[2], gt_xyxy[:, 2])
            yB = np.minimum(p_box[3], gt_xyxy[:, 3])

            interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
            boxAArea = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
            boxBArea = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])

            ious = interArea / (boxAArea + boxBArea - interArea + 1e-6)
            
            if len(ious) > 0:
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]
                
                # Failure if it matches a box but wrong class, OR if it doesn't match any box (IoU < 0.5)
                if max_iou > 0.5:
                    if pred_cls[j] != gt_cls[max_iou_idx]:
                        hard_wrong_indices.append(j)
                else:
                    hard_wrong_indices.append(j)
            else:
                hard_wrong_indices.append(j)

        if not hard_wrong_indices:
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        out = img.copy()

        # Draw GT boxes in green
        for box, cid in zip(gt_xyxy, gt_cls):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(out, f"GT:{CLASS_NAMES[cid]}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

        # Draw wrong predictions in red
        for j in hard_wrong_indices:
            box = pred_xyxy[j]
            cid = pred_cls[j]
            conf = pred_conf[j]
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 2)
            cv2.putText(out, f"PRED:{CLASS_NAMES[cid]} {conf:.2f}",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

        out_path = save_dir / f"failure_{i:04d}.jpg"
        cv2.imwrite(str(out_path), out)
        failures_found += 1
        if failures_found >= 20:   # cap at 20 examples
            break

    print(f"[OK]   {failures_found} failure images saved -> {save_dir}")
    return failures_found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path,
                        default=WEIGHTS_DIR / "best_checkpoint.pth")
    parser.add_argument("--split",   default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    import supervision as sv  # type: ignore

    model = load_model(args.weights)

    ann_path = DATA_DIR / args.split / "_annotations.coco.json"
    img_dir  = DATA_DIR / args.split / "images"
    if not img_dir.exists():
        img_dir = DATA_DIR / args.split
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")

    ds = sv.DetectionDataset.from_coco(
        images_directory_path=str(img_dir),
        annotations_path=str(ann_path),
    )
    print(f"[INFO] Loaded {len(ds)} images from '{args.split}' split.")

    # ── Inference loop (threshold=0.0 for mAP) ──────────────────────────────
    predictions, targets, latencies = run_inference_loop(model, ds, threshold=0.0)

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(predictions, targets, ds)
    lat     = compute_latency(latencies)

    # ── Save all results ──────────────────────────────────────────────────────
    output = {**metrics, "latency": lat, "split": args.split}
    out_path = EVAL_DIR / f"metrics_{args.split}.json"
    out_path.write_text(json.dumps(output, indent=2))
    
    # Rename previous artifacts inside script execution if needed, but since we define them above:
    # Actually, let's redefine the cm_path and hard_failures paths based on the split.
    cm_path = EVAL_DIR / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(predictions, targets, ds, cm_path)

    # ── Failure analysis ─────────────────────────────────────────────────────
    test_paths = [Path(p) for p, _, _ in ds]
    find_hard_failures(
        test_paths, predictions, targets,
        save_dir=EVAL_DIR / f"hard_failures_{args.split}",
        conf_threshold=0.6,
    )

    print(f"\n[OK] All results saved to: {EVAL_DIR}")
    print(f"   metrics:           {out_path}")
    print(f"   confusion_matrix:  {cm_path}")
    print(f"   hard failures:     {EVAL_DIR / f'hard_failures_{args.split}'}/")


if __name__ == "__main__":
    main()
