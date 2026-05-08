"""
Face Mask Detection — Flask + OpenCV Backend
=============================================
Real-time webcam streaming server with EfficientNetB0 (TensorFlow / Keras)
mask-detection inference.

Run:
    python app.py
Then open http://localhost:5001 in your browser.
"""

import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ── Haar Cascade for face detection (ships with OpenCV) ──────────────
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# ── Load trained EfficientNetB0 mask detector ────────────────────────
# Sigmoid output: 0 → WithMask, 1 → WithoutMask (alphabetical class indices)
MODEL_PATH = "avijit_task/mask_model_EfficientNetB0.h5"
print(f"[INFO] Loading model from {MODEL_PATH} …")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")
CLASS_LABELS = ["Mask", "No Mask"]


# ─────────────────────────────────────────────────────────────────────
#  THREAD-SAFE CAMERA MANAGER
#  Handles start/stop so the webcam is properly acquired and released.
# ─────────────────────────────────────────────────────────────────────
class CameraManager:
    """Thread-safe singleton that controls the webcam lifecycle."""

    def __init__(self):
        self._cap = None
        self._lock = threading.Lock()
        self._streaming = False

    @property
    def is_streaming(self):
        return self._streaming

    def start(self):
        """Open the webcam and begin streaming."""
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                self._streaming = True
                return True
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self._cap.isOpened():
                print("[ERROR] Cannot open webcam. Check camera permissions.")
                self._cap = None
                return False
            self._streaming = True
            return True

    def stop(self):
        """Release the webcam and stop streaming."""
        with self._lock:
            self._streaming = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def read_frame(self):
        """Read a single frame. Returns (success, frame)."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            return self._cap.read()


camera = CameraManager()


# ─────────────────────────────────────────────────────────────────────
#  PROCESS FRAME — the core inference hook
# ─────────────────────────────────────────────────────────────────────
def process_frame(frame):
    """
    Detect faces in *frame*, run mask/no-mask classification on each ROI,
    draw annotated bounding boxes, and return the processed frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for (x, y, w, h) in faces:
        # ── Crop face ROI from colour frame ──────────────────────────
        roi_color = frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_color, (224, 224))

        # ── Preprocess to match training pipeline ────────────────────
        # The model was trained with ImageDataGenerator(rescale=1./255),
        # so we replicate that exact normalisation here.
        roi_array = roi_resized.astype("float32") / 255.0
        roi_array = np.expand_dims(roi_array, axis=0)  # (1, 224, 224, 3)

        # ── Run inference ────────────────────────────────────────────
        # Sigmoid output: score → 0 = WithMask, score → 1 = WithoutMask
        prediction = model.predict(roi_array, verbose=0)[0][0]

        if prediction > 0.5:
            label = "No Mask"
            confidence = float(prediction)
            color = (0, 70, 255)    # red
        else:
            label = "Mask"
            confidence = 1.0 - float(prediction)
            color = (0, 200, 100)   # green

        # ── Draw bounding box ────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # ── Label background ─────────────────────────────────────────
        text = f"{label}  {confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(
            frame, (x, y - th - 14), (x + tw + 10, y), color, -1
        )
        cv2.putText(
            frame, text, (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

        # ── Confidence bar ───────────────────────────────────────────
        bar_w = int(w * confidence)
        cv2.rectangle(
            frame, (x, y + h + 4), (x + bar_w, y + h + 12), color, -1
        )
        cv2.rectangle(
            frame, (x, y + h + 4), (x + w, y + h + 12), (80, 80, 80), 1
        )

    return frame


# ─────────────────────────────────────────────────────────────────────
#  VIDEO GENERATOR — yields JPEG frames for MJPEG streaming
# ─────────────────────────────────────────────────────────────────────
def generate_frames():
    """
    Read frames from the webcam via CameraManager, run process_frame()
    on each, and yield them as a multipart JPEG stream.
    Stops cleanly when the camera manager signals streaming has ended.
    """
    while camera.is_streaming:
        success, frame = camera.read_frame()
        if not success:
            break

        # ── Run detection / inference ────────────────────────────────
        frame = process_frame(frame)

        # ── Encode frame → JPEG ─────────────────────────────────────
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ─────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint consumed by the <img> tag."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/start_feed", methods=["POST"])
def start_feed():
    """Open the webcam and begin streaming."""
    success = camera.start()
    return jsonify({"status": "started" if success else "error"})


@app.route("/stop_feed", methods=["POST"])
def stop_feed():
    """Release the webcam and stop streaming."""
    camera.stop()
    return jsonify({"status": "stopped"})


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Face Mask Detection Server")
    print("  Open  http://localhost:5001  in your browser.\n")
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
