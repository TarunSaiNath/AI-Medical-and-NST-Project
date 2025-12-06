import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pathlib


MODEL_PATH = 'C:/7th Sem Project/TarunProject/Result/custom_vgg19_cpu_optimized.keras'  # trained model
LABELS_PATH = 'C:/7th Sem Project/TarunProject/Result/labels.json'                      # class label map
IMG_SIZE = (128, 128)                                     # same as training


print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f) 

print(f"📁 Loaded {len(class_labels)} classes: {class_labels}")



def preprocess_image(img_path):
    """Load and preprocess an image for model inference."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # batch dimension
    img_array = img_array / 255.0                   # normalize
    return img_array

# ============================================================
# 🔮 Predict Function
# ============================================================

def predict_image(img_path, show=True):
    """Predict class of a single image and optionally visualize."""
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array, verbose=0)
    
    pred_idx = np.argmax(predictions)
    pred_label = class_labels[pred_idx]
    confidence = predictions[0][pred_idx] * 100

    print(f"\n🧾 Prediction for '{os.path.basename(img_path)}':")
    print(f"   ➤ Predicted Class : {pred_label}")
    print(f"   ➤ Confidence      : {confidence:.2f}%\n")

    if show:
        plt.imshow(image.load_img(img_path))
        plt.title(f"Prediction: {pred_label} ({confidence:.1f}%)")
        plt.axis('off')
        plt.show()

    return pred_label, confidence

# ============================================================
# 🗂️ Optional: Batch Prediction for Folder
# ============================================================

def predict_folder(folder_path):
    """Run inference on all images inside a folder."""
    folder = pathlib.Path(folder_path)
    img_files = list(folder.glob('*.*'))
    print(f"\n📁 Found {len(img_files)} images in {folder_path}")
    
    results = []
    for img_path in img_files:
        try:
            label, conf = predict_image(str(img_path), show=False)
            results.append((img_path.name, label, conf))
        except Exception as e:
            print(f"⚠️ Skipping {img_path.name}: {e}")
    
    print("\n📊 Summary:")
    for name, label, conf in results:
        print(f"{name:<30} | {label:<15} | {conf:>6.2f}%")
    
    return results

# ============================================================
# 🚀 Example Usage
# ============================================================

if __name__ == "__main__":
    # Single image prediction
    test_image_path = "C:/7th Sem Project/TarunProject/Dataset/train/skin_cancer_negative/ISIC_0028583.jpg"  # 🔁 Replace with your image path
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print(f"⚠️ Test image not found at: {test_image_path}")

    # Batch prediction example (optional)
    # predict_folder("./test_samples/")
