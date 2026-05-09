"""
Face Mask Detection — Flask + OpenCV Backend  (CPU-Optimised Build)
====================================================================
Real-time webcam streaming server with EfficientNetB0 inference.

CPU Optimisations applied:
  • Frame skipping  — ML inference runs every INFERENCE_EVERY_N frames.
                      Cached bounding-box results are reused for in-between
                      frames so the video stream never stalls.
  • Async inference — A dedicated background thread owns the ML model.
                      The camera-read loop and the encode/yield loop are
                      never blocked by a slow predict() call.
  • Smaller cascade — Haar face detection works on a half-resolution copy
                      of the frame, then scales the boxes back up.
  • Reduced JPEG    — Stream quality set to 70 (was 80). Unnoticeable
                      visually but measurably faster to encode.
  • Lower resolution — Webcam captures at 480x360 instead of 640x480.

Run:
    python app.py
Then open http://localhost:5001 in your browser.
"""

import threading
import time
import collections

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ── Performance knobs ────────────────────────────────────────────────
INFERENCE_EVERY_N  = 3      # run ML every Nth frame (1 = every frame, slower)
JPEG_QUALITY       = 70     # stream compression (lower = faster encode)
CAM_WIDTH          = 480    # webcam capture width  (lower = less CPU)
CAM_HEIGHT         = 360    # webcam capture height

# ── Detection quality filters ────────────────────────────────────────
HAAR_MIN_NEIGHBORS = 9      # higher = stricter face detection (fewer false hits)
                            # 5 catches hands; 9 requires a real face structure
MIN_CONFIDENCE     = 0.65   # ignore predictions below 65% confidence
FACE_ASPECT_RATIO  = (0.6, 1.5)  # valid w/h range for a face (hands fall outside)

# ── Haar Cascade for face detection (ships with OpenCV) ──────────────
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# ── Load trained EfficientNetB0 mask detector ────────────────────────
MODEL_PATH   = "avijit_task/mask_model_EfficientNetB0.h5"
CLASS_LABELS = ["Mask", "No Mask"]

try:
    print(f"[INFO] Loading model from {MODEL_PATH} …")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print("\n" + "=" * 60)
    print("[ERROR] Could not load the face mask detection model.")
    print(f"[ERROR] Reason: {e}")
    print("[ERROR] --> Place mask_model_EfficientNetB0.h5 in avijit_task/")
    print("=" * 60 + "\n")
    model = None


# ─────────────────────────────────────────────────────────────────────
#  ASYNC INFERENCE ENGINE
#  The ML model lives in its own thread.  The camera loop deposits a
#  frame into `_pending_frame`; the inference thread picks it up,
#  runs predict(), and writes results into `_last_detections`.
#  All communication is lock-protected.
# ─────────────────────────────────────────────────────────────────────
class InferenceEngine:
    """Decouples slow ML inference from the fast camera-read loop."""

    def __init__(self):
        self._lock           = threading.Lock()
        self._pending_frame  = None   # frame waiting to be inferred
        self._last_detections = []    # [(x,y,w,h,label,confidence,color), …]
        self._running        = False
        self._thread         = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def submit(self, frame):
        """Hand a frame to the inference thread (non-blocking)."""
        with self._lock:
            self._pending_frame = frame.copy()

    def get_detections(self):
        """Return the most recent detection results (non-blocking)."""
        with self._lock:
            return list(self._last_detections)

    # ── Internal worker ──────────────────────────────────────────────
    def _worker(self):
        while self._running:
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None

            if frame is None:
                time.sleep(0.005)   # nothing to do — yield CPU briefly
                continue

            detections = self._run_inference(frame)
            with self._lock:
                self._last_detections = detections

    def _run_inference(self, frame):
        """Run face detection + mask classification on one frame."""
        detections = []

        # ── Scale frame down for faster Haar detection ───────────────
        scale      = 0.5
        small      = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces_small = face_cascade.detectMultiScale(
            gray_small,
            scaleFactor=1.1,
            minNeighbors=HAAR_MIN_NEIGHBORS,   # stricter — filters out hands
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (sx, sy, sw, sh) in faces_small:
            # Scale box back to original frame coordinates
            x, y, w, h = (int(v / scale) for v in (sx, sy, sw, sh))

            # ── Aspect ratio guard ───────────────────────────────────
            # Real faces are roughly square. Hands, arms etc. are not.
            aspect = w / h if h > 0 else 0
            if not (FACE_ASPECT_RATIO[0] <= aspect <= FACE_ASPECT_RATIO[1]):
                continue   # skip — not shaped like a face

            # Clamp to frame boundaries
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            roi     = frame[y1:y2, x1:x2]
            resized = cv2.resize(roi, (224, 224))
            arr     = resized.astype("float32") / 255.0
            arr     = np.expand_dims(arr, axis=0)   # (1,224,224,3)

            if model is not None:
                pred = model.predict(arr, verbose=0)[0][0]
            else:
                pred = 0.0

            if pred > 0.5:
                label, confidence, color = "No Mask", float(pred), (0, 70, 255)
            else:
                label, confidence, color = "Mask", 1.0 - float(pred), (0, 200, 100)

            # ── Confidence threshold guard ───────────────────────────
            # Skip uncertain predictions (e.g. hand partially covering face)
            if confidence < MIN_CONFIDENCE:
                continue   # model isn't sure — don't show anything

            detections.append((x, y, w, h, label, confidence, color))

        return detections


inference_engine = InferenceEngine()


# ─────────────────────────────────────────────────────────────────────
#  THREAD-SAFE CAMERA MANAGER
# ─────────────────────────────────────────────────────────────────────
class CameraManager:
    """Thread-safe singleton that controls the webcam lifecycle."""

    def __init__(self):
        self._cap        = None
        self._lock       = threading.Lock()
        self._streaming  = False

    @property
    def is_streaming(self):
        return self._streaming

    def start(self):
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                self._streaming = True
                return True
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            # Reduce internal buffer to minimise latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self._cap.isOpened():
                print("[ERROR] Cannot open webcam.")
                self._cap = None
                return False
            self._streaming = True
            inference_engine.start()
            return True

    def stop(self):
        with self._lock:
            self._streaming = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        inference_engine.stop()

    def read_frame(self):
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            return self._cap.read()


camera = CameraManager()


# ─────────────────────────────────────────────────────────────────────
#  DRAW DETECTIONS — overlay boxes on any frame
# ─────────────────────────────────────────────────────────────────────
def draw_detections(frame, detections):
    """Stamp cached detection results onto the current frame."""
    for (x, y, w, h, label, confidence, color) in detections:
        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        text = f"{label}  {confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), color, -1)
        cv2.putText(
            frame, text, (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

        # Confidence bar
        bar_w = int(w * confidence)
        cv2.rectangle(frame, (x, y + h + 4), (x + bar_w, y + h + 12), color, -1)
        cv2.rectangle(frame, (x, y + h + 4), (x + w,     y + h + 12), (80, 80, 80), 1)

    return frame


# ─────────────────────────────────────────────────────────────────────
#  VIDEO GENERATOR — yields JPEG frames for MJPEG streaming
# ─────────────────────────────────────────────────────────────────────
def generate_frames():
    """
    Camera loop:
      - Every INFERENCE_EVERY_N frames → submit to async inference engine.
      - Every frame               → draw cached detections and stream JPEG.
    The stream is never blocked by a slow predict() call.
    """
    frame_idx = 0

    while camera.is_streaming:
        success, frame = camera.read_frame()
        if not success:
            break

        # ── Submit to inference engine every N frames ────────────────
        if frame_idx % INFERENCE_EVERY_N == 0:
            inference_engine.submit(frame)

        frame_idx += 1

        # ── Overlay cached detections (always, even on skipped frames) ──
        detections = inference_engine.get_detections()
        frame = draw_detections(frame, detections)

        # ── Encode → JPEG ────────────────────────────────────────────
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# ─────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/start_feed", methods=["POST"])
def start_feed():
    success = camera.start()
    return jsonify({"status": "started" if success else "error"})


@app.route("/stop_feed", methods=["POST"])
def stop_feed():
    camera.stop()
    return jsonify({"status": "stopped"})


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Face Mask Detection Server  [CPU-Optimised]")
    print("  Open  http://localhost:5001  in your browser.\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
