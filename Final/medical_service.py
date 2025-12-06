"""
medical_service.py

Flask server for Medical Diagnosis model.
APIs:
 - GET  /health           -> server health
 - POST /predict          -> single image prediction (multipart/form-data: file field 'image')
 - POST /predict_batch    -> multiple images zip upload or multiple files (optional)

Dependencies:
 pip install flask tensorflow pillow numpy flask-cors werkzeug
"""

import os
import io
import zipfile
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# ----------------------
# Config - update paths
# ----------------------
MODEL_PATH = r"C:/7th Sem Project/TarunProject/Result/custom_vgg19_cpu_optimized.keras"
LABELS_PATH = r"C:/7th Sem Project/TarunProject/Result/labels.json"
IMG_SIZE = (128, 128)
UPLOAD_DIR = "uploads_medical"
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------
# Load model + labels
# ----------------------
print("Loading medical model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

print("Model and labels loaded. Classes:", labels)

# ----------------------
# Helper functions
# ----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_image_pil(pil_img):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE, Image.BICUBIC)
    arr = np.asarray(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_from_path(img_path):
    pil_img = Image.open(img_path)
    arr = preprocess_image_pil(pil_img)
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    return {"label": labels[idx], "confidence": float(preds[0][idx])}

# ----------------------
# Flask App
# ----------------------
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "num_classes": len(labels)})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expect multipart/form-data with file field 'image'.
    Returns: JSON { label, confidence }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        try:
            res = predict_from_path(filepath)
            # include readable confidence as percentage
            res["confidence_percent"] = round(res["confidence"] * 100, 2)
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # optional: remove saved file
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Option A: accept a zip file containing images in root. Field name 'zipfile'
    Option B: accept multiple 'image' files in same request (multipart)
    Returns JSON array of results.
    """
    results = []
    # 1) if zipfile provided
    if "zipfile" in request.files:
        zf = request.files["zipfile"]
        if zf.filename == "":
            return jsonify({"error": "Empty zip file"}), 400
        # read zipfile in-memory
        try:
            zbytes = io.BytesIO(zf.read())
            with zipfile.ZipFile(zbytes, "r") as z:
                for name in z.namelist():
                    if not allowed_file(name):
                        continue
                    data = z.read(name)
                    pil = Image.open(io.BytesIO(data))
                    arr = preprocess_image_pil(pil)
                    preds = model.predict(arr, verbose=0)
                    idx = int(np.argmax(preds, axis=1)[0])
                    results.append({"file": name, "label": labels[idx], "confidence": float(preds[0][idx])})
            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # 2) or multiple image files sent as 'image'
    files = request.files.getlist("image")
    if not files:
        return jsonify({"error": "No files provided (use 'zipfile' or multiple 'image' fields)"}), 400
    for file in files:
        if not allowed_file(file.filename):
            continue
        fname = secure_filename(file.filename)
        fpath = os.path.join(UPLOAD_DIR, fname)
        file.save(fpath)
        try:
            r = predict_from_path(fpath)
            r["file"] = fname
            results.append(r)
        finally:
            if os.path.exists(fpath):
                os.remove(fpath)
    return jsonify({"results": results})

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    # Production: use gunicorn/uvicorn. For development:
    app.run(host="0.0.0.0", port=5001, debug=False)
