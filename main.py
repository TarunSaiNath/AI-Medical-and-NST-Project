from flask import Flask, request, render_template_string, send_file, jsonify
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import time

# ---------------------- CONFIG ----------------------
app = Flask(__name__)

# Healthcare Model
MODEL_PATH = "C:/7th Sem Project/TarunProject/Inference/latest_model.keras"
LABELS_PATH = "C:/7th Sem Project/TarunProject/Inference/labels.json"
IMG_HEIGHT, IMG_WIDTH = 299, 299

print("🔄 Loading Healthcare Model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# NST config
IMG_MAX_DIM = 512
TEMP_FOLDER = "./nst_output"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Style Transfer layers
content_layers = ['block4_conv2']
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# ---------------------- Functions ----------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array, verbose=0)
    index = np.argmax(predictions)
    return class_labels[index], float(predictions[0][index])


def load_img(path):
    img = Image.open(path).convert("RGB")       # Important Fix ✅
    long = max(img.size)
    scale = IMG_MAX_DIM / long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
    img = np.array(img, dtype=np.float32)
    return np.expand_dims(img, axis=0)


def load_and_process_img(path):
    return tf.keras.applications.vgg19.preprocess_input(load_img(path))


def deprocess_img(img):
    img = np.squeeze(img, 0)
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    return np.clip(img, 0, 255).astype('uint8')


def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    return tf.keras.Model(vgg.input, outputs)


def gram_matrix(tensor):
    x = tf.reshape(tensor, (-1, tensor.shape[-1]))
    return tf.matmul(x, x, transpose_a=True) / tf.cast(tf.shape(x)[0], tf.float32)


def run_style_transfer(content_path, style_path, iterations=50):
    model = get_model()
    style_image = load_and_process_img(style_path)
    content_image = load_and_process_img(content_path)
    generated = tf.Variable(content_image, dtype=tf.float32)

    outputs = model(style_image)
    style_features = [gram_matrix(layer) for layer in outputs[:num_style_layers]]

    outputs = model(content_image)
    content_features = outputs[num_style_layers:]

    opt = tf.keras.optimizers.Adam(learning_rate=5.0)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            outputs = model(generated)
            gen_style = outputs[:num_style_layers]
            gen_content = outputs[num_style_layers:]

            s_loss = tf.add_n([tf.reduce_mean((gram_matrix(gs)-style_features[i])**2)
                               for i, gs in enumerate(gen_style)])
            c_loss = tf.add_n([tf.reduce_mean((gc-content_features[i])**2)
                               for i, gc in enumerate(gen_content)])
            loss = s_loss*1e-2 + c_loss

        grad = tape.gradient(loss, generated)
        opt.apply_gradients([(grad, generated)])

    return deprocess_img(generated.numpy())


# ---------------------- API Endpoints ----------------------
@app.route("/")
def home():
    return render_template_string("""
    <h1 align=center style="margin-top:50px;">AI Project Dashboard</h1>
    <div style="text-align:center;margin-top:40px;">
        <a href="/nst-page"><button style="padding:12px 24px;">🎨 Neural Style Transfer</button></a><br><br>
        <a href="/health-page"><button style="padding:12px 24px;">🩺 Disease Classification</button></a>
    </div>
    """)


@app.route("/nst-page")
def nst_ui():
    return render_template_string("""
    <h2 align=center>Neural Style Transfer</h2>
    <form method="POST" action="/nst" enctype="multipart/form-data" style="text-align:center;">
        Content Image: <input type="file" name="content_image" required><br><br>
        Style Image: <input type="file" name="style_image" required><br><br>
        <button type="submit">Generate</button>
    </form>
    """)


@app.route("/health-page")
def health_ui():
    return render_template_string("""
    <h2 align=center>Disease Image Classification</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data" style="text-align:center;">
        Upload Image: <input type="file" name="image" required><br><br>
        <button type="submit">Analyze</button>
    </form>
    """)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    path = "temp.jpg"
    file.save(path)
    label, conf = predict_image(path)
    os.remove(path)
    return f"<center><h2>Prediction: {label} ({conf*100:.2f}%)</h2></center>"


@app.route("/nst", methods=["POST"])
def nst():
    content = request.files["content_image"]
    style = request.files["style_image"]
    cpath = TEMP_FOLDER + "/c.jpg"
    spath = TEMP_FOLDER + "/s.jpg"
    opath = TEMP_FOLDER + "/o.jpg"
    content.save(cpath)
    style.save(spath)
    output = run_style_transfer(cpath, spath)
    Image.fromarray(output).save(opath)
    return send_file(opath, mimetype="image/jpeg")


# ---------------------- RUN ----------------------
if __name__ == "__main__":
    app.run(debug=True, port=7000)
