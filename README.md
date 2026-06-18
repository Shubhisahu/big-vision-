# ShelfSight — Retail Object Detector

RF-DETR fine-tuned on a retail-shelf dataset (bottle · box · can · bag) with ByteTrack multi-object tracking and a FastAPI + React web interface.

---

## Project Structure

```
BIG VISION/
  backend/
    weights/          ← place best_checkpoint.pth here after training
    main.py           ← FastAPI server
    inference.py      ← multi-backend inference (ONNX/IPEX/CPU)
    requirements.txt
  frontend/
    src/
      App.jsx
      components/
        UploadPane.jsx   ← drag & drop image inference
        ResultPane.jsx   ← detection results display
        WebcamPane.jsx   ← live webcam detection
    package.json
  evaluation/
    evaluate.py       ← mAP, confusion matrix, failure analysis
  tracking/
    track_video.py    ← ByteTrack video inference CLI
  training/
    retail_detector.ipynb  ← Kaggle RF-DETR fine-tuning notebook
  data/
    test/             ← copy test split here for local evaluation
      images/
      _annotations.coco.json
  README.md
```

---

## Quick Start

### 1. Clone and install backend
```bash
cd backend
pip install -r requirements.txt
```

### 2. Add weights
Download `best_checkpoint.pth` from your Kaggle output panel and place it in:
```
backend/weights/best_checkpoint.pth
```

### 3. Start the backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model status + uptime |
| GET | `/metrics` | Request counters |
| POST | `/infer` | Single image → detections + annotated image |
| POST | `/infer-frame` | Webcam frame → tracked detections |
| POST | `/infer-video` | Upload MP4 → tracked video URL |
| WS | `/ws/{session_id}` | Real-time WebSocket streaming |

---

## Training

Training can be run on Kaggle or Google Colab with Tesla T4 GPUs.

- **Kaggle:** Open `training/retail_detector.ipynb` and run all cells.
- **Google Colab:** Open `training/colab_retail_detector.ipynb`, upload the dataset folder to your Google Drive, and follow the instructions in the notebook to run the training.

**Dataset:** `shubhi272/retail-object-detector` on Kaggle  
**Model:** RF-DETR Base (DINOv2-B backbone, 31.9M params)  
**Classes:** bottle, box, can, bag  
**Epochs:** 60 | **Batch:** 4 × grad_accum 8 = effective 32

---

## Evaluation

After training, copy test images locally then run:
```bash
python evaluation/evaluate.py --weights backend/weights/best_checkpoint.pth
```
Outputs: `mAP@50`, `mAP@50-95`, confusion matrix, hard failure images.

---

## Tracking

```bash
# Process a video file
python tracking/track_video.py --source video.mp4

# Live webcam
python tracking/track_video.py --source 0 --show
```

---

## Inference Backends (auto-detected)

| Priority | Backend | Speed | When used |
|----------|---------|-------|-----------|
| 1st | ONNX + OpenVINO | ~30ms | Intel Arc GPU / iGPU |
| 2nd | IPEX XPU | ~40ms | Intel Arc via PyTorch |
| 3rd | CPU rfdetr | ~150ms | Always available |

---

## Classes & Colours

| Class | Colour |
|-------|--------|
| bottle | 🟡 #FFC800 |
| box | 🟢 #64FF64 |
| can | 🔵 #00A0FF |
| bag | 🟣 #C800FF |
