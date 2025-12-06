import os
import warnings
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, regularizers

# ============================================================
# 🧠 Custom VGG19-like Architecture (optimized for CPU)
# ============================================================

def conv_block(x, filters, convs=2, kernel_size=(3,3), name_prefix="block"):
    """VGG-like block: Conv2D → BN → ReLU × convs → MaxPool"""
    for i in range(convs):
        x = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{name_prefix}_conv{i+1}_{filters}"
        )(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn{i+1}_{filters}")(x)
        x = layers.Activation('relu', name=f"{name_prefix}_relu{i+1}_{filters}")(x)
    x = layers.MaxPooling2D((2,2), strides=(2,2), name=f"{name_prefix}_pool")(x)
    return x


def build_custom_vgg19(input_shape=(128,128,3), num_classes=4, dropout_rate=0.5):
    """Enhanced CPU-optimized VGG19-like model."""
    inputs = layers.Input(shape=input_shape, name="input_image")

    # VGG-style convolutional blocks
    x = conv_block(inputs, 64, 2, name_prefix="block1")
    x = conv_block(x, 128, 2, name_prefix="block2")
    x = conv_block(x, 192, 3, name_prefix="block3")
    x = conv_block(x, 256, 3, name_prefix="block4")
    x = conv_block(x, 256, 3, name_prefix="block5")

    # Replace Flatten with GlobalAveragePooling
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense Head with regularization
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name="classifier")(x)
    model = models.Model(inputs, outputs, name="CustomVGG19_CPU_Optimized")
    return model

# ============================================================
# ⚙️ Environment Setup
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

# GPU memory growth (optional)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU Memory Growth Error:", e)

# ============================================================
# 📁 Dataset Setup
# ============================================================

data_dir = pathlib.Path('./Dataset')
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'

img_height, img_width = 128, 128
batch_size = 8
epochs = 100

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\nDetected {num_classes} classes: {list(train_generator.class_indices.keys())}")
print("Training images:", len(train_generator.filenames))
print("Validation images:", len(val_generator.filenames))

train_steps = max(1, train_generator.samples // batch_size)
val_steps = max(1, val_generator.samples // batch_size)

# ============================================================
# ⚖️ Compute Class Weights (helps with imbalance)
# ============================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("\nClass Weights:", class_weights)

# ============================================================
# 🧩 Build & Compile Model
# ============================================================

model = build_custom_vgg19(
    input_shape=(img_height, img_width, 3),
    num_classes=num_classes,
    dropout_rate=0.5
)
model.summary()

optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 🚀 Callbacks for Better Accuracy
# ============================================================

results_folder = './Result'
os.makedirs(results_folder, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(results_folder, 'best_model.keras'),
                                       monitor='val_accuracy',
                                       save_best_only=True,
                                       verbose=1)
]

# ============================================================
# 🏋️‍♂️ Train Model
# ============================================================

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# ============================================================
# 📊 Evaluate and Save Metrics
# ============================================================

val_generator.reset()
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

y_pred = model.predict(val_generator, steps=val_steps + 1, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_true[:len(y_pred_classes)], y_pred_classes)
prec = precision_score(y_true[:len(y_pred_classes)], y_pred_classes, average='weighted', zero_division=0)
rec = recall_score(y_true[:len(y_pred_classes)], y_pred_classes, average='weighted', zero_division=0)
f1 = f1_score(y_true[:len(y_pred_classes)], y_pred_classes, average='weighted', zero_division=0)

with open(f'{results_folder}/metrics_summary.txt', 'w') as f:
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall:    {rec:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")

# ============================================================
# 📈 Plot Accuracy and Loss
# ============================================================

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'{results_folder}/training_metrics.png')
plt.close()

# ============================================================
# 💾 Save Final Model and Labels
# ============================================================

final_model_path = os.path.join(results_folder, "custom_vgg19_cpu_optimized.keras")
labels_path = os.path.join(results_folder, "labels.json")

model.save(final_model_path)
with open(labels_path, "w") as f:
    json.dump(class_labels, f, indent=2)

print(f"\n✅ Training complete.")
print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print(f"Model saved to: {final_model_path}")
print(f"Best model checkpoint saved to: {results_folder}/best_model.keras")
print(f"Metrics saved to: {results_folder}/metrics_summary.txt")
