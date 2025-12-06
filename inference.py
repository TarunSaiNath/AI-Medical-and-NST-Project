import tensorflow as tf
import numpy as np
import json
import sys
from tensorflow.keras.preprocessing import image
import pathlib

RESULT_DIR = pathlib.Path("./Result")
MODEL_PATH = RESULT_DIR / "latest_model.keras"
LABELS_PATH = RESULT_DIR / "labels.json"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

IMG_HEIGHT, IMG_WIDTH = 299, 299

if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = input("Enter image path: ").strip()

img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array, verbose=0)
predicted_index = np.argmax(predictions)
predicted_label = class_labels[predicted_index]
confidence = predictions[0][predicted_index]

print("\n🔍 Prediction Result:")
print(f"Predicted Label: {predicted_label}")
print(f"Confidence: {confidence * 100:.2f}%")

with open(RESULT_DIR / "last_inference.txt", "w") as f:
    f.write(f"Image: {img_path}\n")
    f.write(f"Predicted Label: {predicted_label}\n")
    f.write(f"Confidence: {confidence * 100:.2f}%\n")

print(f"\n✅ Result saved to: {RESULT_DIR}/last_inference.txt")
