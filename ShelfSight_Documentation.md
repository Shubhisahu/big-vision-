# ShelfSight — Retail Object Detector: Complete Process Documentation

This document provides a clean, detailed write-up of the end-to-end process for building the ShelfSight project. It covers why specific tools (like Roboflow and Kaggle) were used, what parameters mean, and how the entire codebase fits together.

---

## 1. Dataset Preparation & Roboflow

### Why Roboflow?
We use Roboflow because managing computer vision datasets manually is extremely tedious. Roboflow handles image hosting, collaborative annotation (drawing bounding boxes for `bottle`, `box`, `can`, `bag`), versioning, and formatting. Crucially, it handles **Preprocessing** and **Augmentations**.

### How Roboflow Parameters Affect the Model

When you generate a dataset version in Roboflow, you apply Preprocessing and Augmentation steps. Here is how they impact the final model:

**1. Preprocessing Parameters (Ensuring Consistency)**
*   **Auto-Orient:** Strips EXIF data so images don't appear sideways to the model. *Effect: Prevents the model from learning incorrect spatial orientations.*
*   **Resize (e.g., to 640x640):** Neural networks require fixed input sizes. *Effect: Standardizes training, but if the original image was huge, small objects lose detail. We must balance resolution with GPU memory limits.*

**2. Augmentation Parameters (Preventing Overfitting)**
Augmentations artificially multiply your dataset size by creating modified copies of your images. 
*   **Rotation / Flip:** Flips images horizontally or rotates them. *Effect: Teaches the model that a bottle on its side is still a bottle. The model learns rotation-invariant features.*
*   **Brightness / Contrast / Exposure:** Randomly darkens or brightens images. *Effect: Makes the model robust against different lighting conditions in retail stores (e.g., dark shelves vs. brightly lit aisles).*
*   **Noise / Blur:** Adds artificial static or blur. *Effect: Simulates cheap webcams or motion blur, forcing the model to rely on shapes rather than sharp textures.*
*   **Mosaic / Cutout:** Combines multiple images into one or cuts out black squares. *Effect: Forces the model to identify objects even if they are partially occluded by other items on the shelf.*

**Conclusion on Roboflow:** By using heavy augmentations, we prevent the model from simply memorizing our training images (overfitting) and force it to generalize to unseen real-world scenarios.

---

## 2. Model Training & Kaggle

### Why Kaggle?
Training modern Vision Transformers requires significant compute power. Kaggle provides **free access to high-end GPUs** (like the Nvidia Tesla T4 or P100) and allows us to easily attach large datasets (`shubhi272/retail-object-detector`) directly to our notebook without downloading gigabytes of data locally.

### Why RF-DETR instead of YOLO?
We used the **RF-DETR (Real-Time Detection Transformer)** model with a DINOv2-B backbone rather than traditional YOLO architectures for several key reasons:

1. **It Doesn't Need NMS (Non-Maximum Suppression):** YOLO initially guesses hundreds of overlapping bounding boxes and uses NMS to delete duplicates. NMS is slow and requires manual threshold tuning. RF-DETR uses bipartite matching to directly output one unique box per object, vastly simplifying the pipeline.
2. **Better at Dense, Cluttered Scenes:** In retail, bottles and cans are packed tightly. YOLO's NMS can accidentally delete valid bounding boxes if objects are too close together. Because RF-DETR outputs exact, unique predictions without suppression, it handles dense and overlapping objects much better.
3. **Global Context (Understanding the "Big Picture"):** YOLO uses CNNs which look at images in small local patches. RF-DETR uses Transformers which look at the *entire* image at once via self-attention. It understands that a blurry shape is likely a bottle because it sees the surrounding context of the shelf.
4. **The "Real-Time" Breakthrough:** Historically, Transformer models were highly accurate but too slow for video. RF-DETR optimized the architecture to run just as fast as YOLO, providing state-of-the-art accuracy at real-time speeds.

### How Training Parameters Affect the Output
Inside the `training/retail_detector.ipynb` file, several hyperparameters govern the training process:

*   **Epochs (60):** An epoch is one full pass of the dataset through the neural network. 
    *   *Effect:* Too few epochs (e.g., 5) leads to *underfitting* (the model hasn't learned enough). Too many (e.g., 300) leads to *overfitting* (the model memorizes the data and fails in the real world). 60 is a sweet spot for convergence.
*   **Batch Size (4) & Gradient Accumulation (8):** 
    *   *Effect:* Batch size is how many images the model looks at before updating its internal weights. High batch sizes stabilize learning. However, transformers are memory-hungry and a large batch would crash the Kaggle GPU (Out of Memory). We use a batch size of 4, but accumulate the gradients over 8 steps. This gives us an **effective batch size of 32 (4 x 8)**, providing stable learning without crashing the GPU.
*   **Weights (`best_checkpoint.pth`):** During training, the model saves checkpoints. The one with the lowest validation loss is saved as the final output.

---

## 3. Project Files & Architecture Explained

Once the model is trained, it needs to be integrated into an application. Here is how every piece of the puzzle fits together.

### The Backend (`/backend`)
*   **`main.py`:** A FastAPI web server. It listens for HTTP requests (like an image upload from a user) or WebSockets (for live webcam streaming).
*   **`inference.py`:** The brain of the API. It loads the `best_checkpoint.pth` generated by Kaggle. It is optimized to automatically select the best hardware available (Intel ONNX/IPEX for speed, or CPU fallback).
*   *Why we used this:* FastAPI is incredibly fast and asynchronous, making it perfect for handling heavy machine learning payloads concurrently.

### The Frontend (`/frontend`)
*   **`src/components/UploadPane.jsx`, `WebcamPane.jsx`, etc.:** A React web interface. 
*   *Why we used this:* React allows us to build a dynamic single-page application. When a user uploads an image, the frontend sends it to the FastAPI backend, waits for the bounding box coordinates, and then dynamically draws the boxes (🟡 bottle, 🟢 box, 🔵 can, 🟣 bag) over the image on the screen.

### Tracking (`/tracking/track_video.py`)
*   **ByteTrack Integration:** Standard object detection treats every frame of a video as completely independent. This causes bounding boxes to "flicker", and the system doesn't know if the bottle in frame 1 is the same bottle in frame 2.
*   *Why we used this:* ByteTrack assigns a unique ID to every detected object and tracks it via its bounding box trajectory across frames. This is vital for retail analytics (e.g., counting *how many unique* bottles pass by the camera).

### Evaluation (`/evaluation/evaluate.py`)
*   *Why we used this:* Once the model is deployed locally, we need to know how good it actually is. This script calculates the **mAP (Mean Average Precision)**. It generates confusion matrices to tell us exactly where the model struggles (e.g., "Is it confusing a box for a bag?"). This tells us what kind of data we need to add to Roboflow for the next version.

---

## Summary of the Pipeline
1.  **Roboflow** standardizes and augments raw retail images so the model doesn't overfit.
2.  **Kaggle** provides the heavy GPU compute needed to train the cutting-edge **RF-DETR** architecture over 60 epochs.
3.  The output weights are placed into the **FastAPI Backend**, which serves predictions.
4.  The **React Frontend** provides a beautiful user interface for interacting with the AI.
5.  **ByteTrack** ensures objects in video feeds are tracked smoothly over time without flickering.

---

## 4. Comprehensive Technology & Tool List

Here is a complete list of all the technologies, frameworks, tools, and algorithms used in the ShelfSight project, broken down by their role in the pipeline:

### 🧠 Models & Algorithms
*   **RF-DETR (Real-Time Detection Transformer)**: The primary object detection model used to identify the items (bottles, boxes, cans, bags).
*   **DINOv2-B**: The pre-trained Vision Transformer backbone used by the RF-DETR model to extract visual features.
*   **ByteTrack**: The tracking algorithm used to assign persistent IDs to objects across video frames.

### 🐍 Backend Python Packages
*   `rfdetr (>=0.1.0)`: The core library for the detection transformer.
*   `torch (>=2.1.0)`: PyTorch, the deep learning tensor library used to run the model.
*   `supervision (>=0.22.0)`: Used for robust computer vision utilities.
*   `opencv-python-headless (>=4.9.0)`: OpenCV, used for image and video manipulation.
*   `numpy (>=1.24.0)`: Used for heavy mathematical operations.
*   `Pillow (>=10.0.0)`: Python Imaging Library, used for basic image reading/writing.
*   `albumentations (>=1.3.0)`: Used for applying image augmentations.
*   `fastapi (>=0.115.0)`: The high-performance API framework.
*   `uvicorn[standard] (>=0.30.0)`: The web server that runs FastAPI.
*   `python-multipart (>=0.0.9)`: Required to accept image uploads.
*   `aiofiles (>=23.0.0)`: Asynchronous file saving/reading.
*   `websockets (>=12.0)`: Used to stream live webcam frames.
*   `slowapi (>=0.1.9)`: Used to rate-limit API endpoints.
*   `roboflow (>=1.1.0)`: Used to automate dataset downloading.
*   `pyyaml (>=6.0)`: Parses configuration files.
*   `onnxruntime-openvino`, `onnx`, `onnxruntime`: Used to run the model at maximum speed (~30ms) on Intel hardware.
*   `intel-extension-for-pytorch`: IPEX framework for Intel Arc acceleration.

### ⚛️ Frontend NPM Packages
*   `react (^18.3.1)`: The core UI library.
*   `react-dom (^18.3.1)`: Binds React to the browser DOM.
*   `react-router-dom (^7.17.0)`: Handles navigation within the application.
*   `vite (^5.4.2)`: The local development server and build bundler.
*   `@vitejs/plugin-react (^4.3.1)`: Allows Vite to understand React JSX.

### 📦 Deployment
*   **Docker**: Used to containerize the application for consistent deployment environments.
*   **Hugging Face Spaces**: Used to host and deploy the containerized FastAPI backend, providing a scalable and accessible server for the machine learning model.
*   **Vercel**: Used to deploy the React frontend, offering fast, edge-optimized web hosting and a seamless CI/CD pipeline linked directly to the repository.

**⚠️ Limitations of Free-Tier Deployment:**
*   **Cold Starts:** Free Hugging Face Spaces go to sleep after inactivity. The first user request can take 2-5 minutes to wake the server, often causing an initial timeout.
*   **Slow Inference (No GPU):** Free tiers only provide CPUs. RF-DETR inference takes ~1.4s per frame on a CPU, making the live webcam feature extremely choppy compared to local execution.
*   **WebSocket Restrictions:** Vercel's serverless architecture kills long-lived connections. The frontend must bypass Vercel and connect its WebSockets directly to the Hugging Face backend.
