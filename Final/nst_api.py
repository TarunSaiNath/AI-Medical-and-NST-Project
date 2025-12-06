import os
import io
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf

# ----------------------------------------------------
# CONFIGURATION — SAFE ABSOLUTE PATHS (IMPORTANT FIX)
# ----------------------------------------------------

# Always get absolute folder of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_EXTRACTOR_PATH = r"C:/7th Sem Project/TarunProject/custom_vgg19_feature_extractor_fixed.keras"
IMG_SIZE = (128, 128)

# Correct fixed directories inside THIS folder:
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads_nst")
OUTPUT_DIR = os.path.join(BASE_DIR, "nst_output")

# Create directories safely
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# LOAD FEATURE EXTRACTOR MODEL
# ----------------------------------------------------
print("🔄 Loading Feature Extractor…")
extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH, compile=False)
extractor.trainable = False
print("✅ NST Feature Extractor Loaded!")


# ----------------------------------------------------
# IMAGE HELPERS
# ----------------------------------------------------
def load_img_tensor(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BICUBIC)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return tf.convert_to_tensor(arr, dtype=tf.float32)

def deprocess_img(tensor):
    x = tensor.numpy()
    if x.ndim == 4:
        x = x[0]
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(x)

def gram_matrix(x):
    x = tf.reshape(x, [-1, x.shape[-1]])  # (HW, C)
    gram = tf.matmul(x, x, transpose_a=True)
    return gram / tf.cast(tf.shape(x)[0], tf.float32)


# ----------------------------------------------------
# NEURAL STYLE TRANSFER LOGIC
# ----------------------------------------------------
def run_style_transfer(content_tensor, style_tensor, iterations=150, style_weight=5e-3, content_weight=1.0):
    outputs = extractor(style_tensor)
    num_style_layers = len(outputs) - 1

    style_features = outputs[:num_style_layers]
    gram_style = [gram_matrix(f) for f in style_features]

    content_feature = extractor(content_tensor)[-1]

    generated = tf.Variable(content_tensor, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5.0, epsilon=1e-1)

    best_loss = float("inf")
    best_img = None

    for i in range(iterations):
        with tf.GradientTape() as tape:
            outputs = extractor(generated)
            gen_style = outputs[:num_style_layers]
            gen_content = outputs[-1]

            # style loss
            s_loss = tf.add_n([
                tf.reduce_mean(tf.square(gram_matrix(gen_style[j][0]) - gram_style[j]))
                for j in range(num_style_layers)
            ])
            s_loss *= style_weight / num_style_layers

            # content loss
            c_loss = tf.reduce_mean(tf.square(gen_content[0] - content_feature[0]))
            c_loss *= content_weight

            total_loss = s_loss + c_loss

        grads = tape.gradient(total_loss, generated)
        optimizer.apply_gradients([(grads, generated)])
        generated.assign(tf.clip_by_value(generated, 0.0, 1.0))

        if total_loss < best_loss:
            best_loss = total_loss
            best_img = deprocess_img(generated)

    return best_img


# ----------------------------------------------------
# FLASK APP
# ----------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "NST API Running ✔", "POST": "/stylize"})


@app.route("/stylize", methods=["POST"])
def stylize():
    # Validate inputs
    if "content" not in request.files:
        return jsonify({"error": "Missing content image"}), 400
    if "style" not in request.files:
        return jsonify({"error": "Missing style image"}), 400

    content_file = request.files["content"]
    style_file = request.files["style"]

    # Save uploads safely
    content_path = os.path.join(UPLOAD_DIR, "content.jpg")
    style_path = os.path.join(UPLOAD_DIR, "style.jpg")

    content_file.save(content_path)
    style_file.save(style_path)

    # Load tensors
    content_tensor = load_img_tensor(content_path)
    style_tensor = load_img_tensor(style_path)

    # Run NST
    output_img = run_style_transfer(content_tensor, style_tensor, iterations=150)

    # Save result
    output_path = os.path.join(OUTPUT_DIR, "output.jpg")
    output_img.save(output_path)

    # Return final stylized image
    return send_file(output_path, mimetype="image/jpeg")


# ----------------------------------------------------
# START SERVER
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=True)

