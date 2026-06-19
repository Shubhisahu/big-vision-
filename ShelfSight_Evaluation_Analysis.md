# ShelfSight Evaluation Analysis

## 1. Overview

ShelfSight is a retail shelf object detector built on **RF-DETR (DINOv2-B backbone)**, trained for 60 epochs on a Roboflow-augmented dataset (`shubhi272/retail-object-detector`) using a Kaggle GPU, with an effective batch size of 32 (batch size 4 × gradient accumulation 8). The model detects four classes: **bag, bottle, box, can**. This analysis reviews the actual evaluation outputs — `metrics.json`, `metrics_train.json`, `metrics_val.json`, and the three confusion matrix images — against the documented pipeline and architecture choices.

Two of the three uploaded confusion matrices (the un-suffixed `confusion_matrix.png` and `confusion_matrix_train.png`) are pixel-identical and both represent the **training split**. The third (`confusion_matrix_val.png`) is the **validation split**, at roughly 1/20th the sample count.

## 2. Headline Metrics

| Metric | Train | Val | Gap |
|---|---|---|---|
| mAP@50 | 0.9526 | 0.8499 | −10.3 pts |
| mAP@50-95 | 0.7554 | 0.6416 | −11.4 pts |
| Latency p50 (ms) | ~862–900 | 1417.2 | +~58% |
| Latency p95 (ms) | ~2028–2192 | 5192.3 | +~2.4× |

**This is the central finding of the evaluation: a sizable train→val gap on both accuracy and latency.** A ~10-point mAP@50 drop and an even larger mAP@50-95 drop (which penalizes loose/imprecise boxes more heavily) indicates the model is not generalizing as cleanly as the training numbers suggest. The latency gap is unusual — inference speed shouldn't normally differ much between splits unless validation images differ systematically in resolution, object density, or clutter, or unless the timing run included cold-start / thermal effects. Given the documentation's note that free-tier CPU inference runs around 1.4s/frame, the val p50 of 1.4s is consistent with CPU-only evaluation, while the train numbers may have benefited from caching or a warmed-up runtime.

## 3. Per-Class Breakdown

Derived from the confusion matrices (rows = ground truth, columns = predictions; "FN" = missed detections, "FP" row = spurious detections with no matching ground truth object):

### Training split

| Class | Support | Recall | Precision | Notes |
|---|---|---|---|---|
| bag | 6,783 | 89.4% | 82.8% | 692 missed; small bleed into "bottle" (28) |
| bottle | 19,394 | 86.2% | 85.3% | Largest class; 2,576 missed (largest FN count); 102 misread as "bag" |
| box | 4,025 | 94.8% | 92.2% | Cleanest class besides "can" |
| can | 1,070 | 98.2% | 96.2% | Best-performing class, smallest class |

### Validation split

| Class | Support | Recall | Precision | Notes |
|---|---|---|---|---|
| bag | 474 | 89.2% | 73.6% | Recall holds up; precision drops (more FP bleed) |
| bottle | 872 | **50.6%** | 82.9% | Recall collapses — over 400 missed detections |
| box | 833 | 89.2% | 90.2% | Modest, consistent drop from train |
| can | 84 | 100% | 96.6% | Holds up, but tiny sample size limits confidence |

**The standout failure is "bottle" recall on validation: it falls from 86.2% (train) to 50.6% (val) — a 35-point collapse**, while precision for bottle barely moves (85.3% → 82.9%). In other words, when the model predicts "bottle" it's usually right, but on unseen data it is **missing roughly half of all bottles outright** (427 of 872 ground-truth bottles go undetected, vs. only 4 confused with "bag"). This is the single metric most worth investigating before any deployment decision — it's not a class-confusion problem (the model isn't mistaking bottles for other classes), it's a **detection/recall problem**, meaning the model frequently fails to propose a box for a bottle at all.

Possible contributing factors, given the documented pipeline:
- **Bottle is the majority class** (19,394 of ~31,272 training instances, ~62%) — yet it has the *worst* train recall of the four classes even before the val drop, suggesting it's an intrinsically harder class (likely due to visual variety — sizes, transparency, labels — and occlusion in packed shelves) rather than a data-volume problem.
- The Roboflow augmentation set (mosaic/cutout for occlusion, brightness/exposure for lighting) is exactly aimed at this failure mode, which implies either the val set contains harder real-world conditions than the augmentations cover, or the augmentation strength/coverage needs to increase specifically for bottles.
- Bag and box, by contrast, generalize their recall almost perfectly (89.4%→89.2%, 94.8%→89.2%) — only precision erodes on validation, which is the expected, "normal" amount of train/val gap. Bottle is the outlier.

## 4. False Positives

The "FP" row (detections with no matching ground truth) is proportionally larger on training data for bottle (2,847 of 19,591 predicted-bottle boxes, ~14.5%) than on validation (91 of 532, ~17.1%) — roughly comparable rates. Bag shows the same pattern (1,157/7,322 ≈ 15.8% train vs. 148/575 ≈ 25.7% val), meaning bag precision degrades more on unseen data than bottle precision does. So the model's two failure modes split by class: **bottle struggles with recall (missed detections), bag struggles more with precision (spurious detections)** on validation data.

## 5. Relating This Back to the Documented Design Choices

- **RF-DETR over YOLO** was chosen partly for better performance in "dense, cluttered scenes" by avoiding NMS-related box suppression. The bottle recall collapse on validation suggests this benefit isn't fully showing up for the most visually dense class in this dataset — bottles are typically packed tightest on real shelves, so this is worth a closer qualitative look (e.g., visualizing missed-bottle cases) rather than assuming the architecture choice alone solves clutter.
- **60 epochs** was selected as a "sweet spot" against under/overfitting. The ~10–11 point mAP gap and the bottle-specific recall collapse are consistent with at least mild overfitting on the bottle class specifically, even if the aggregate epoch count is reasonable. This points toward investigating class-balanced sampling or stronger bottle-specific augmentation rather than simply reducing total epochs.
- **Effective batch size of 32** (4 × 8 accumulation) was used to fit Kaggle's GPU memory. This is unlikely to explain the per-class recall gap, which looks like a data/augmentation coverage issue rather than an optimization stability issue.
- **Free-tier CPU latency** (~1.4s/frame, documented as a known limitation) lines up closely with the validation p50 of 1417.2 ms, reinforcing that the val metrics were likely captured under realistic/production-like (CPU) conditions, while train metrics may reflect a faster runtime — worth confirming, since comparing accuracy across two different hardware/timing conditions can be misleading if not controlled for.
