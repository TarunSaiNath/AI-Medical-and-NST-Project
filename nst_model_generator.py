
#  Create and Save a Custom Feature Extractor for NST
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

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
    """Base CustomVGG19 model."""
    inputs = layers.Input(shape=input_shape, name="input_image")

    x = conv_block(inputs, 64, 2, name_prefix="block1")
    x = conv_block(x, 128, 2, name_prefix="block2")
    x = conv_block(x, 192, 3, name_prefix="block3")
    x = conv_block(x, 256, 3, name_prefix="block4")
    x = conv_block(x, 256, 3, name_prefix="block5")

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name="classifier")(x)
    model = models.Model(inputs, outputs, name="CustomVGG19_CPU_Optimized")
    return model

def build_feature_extractor(input_shape=(128,128,3)):
    """Builds NST-compatible feature extractor version."""
    base_model = build_custom_vgg19(input_shape=input_shape)

    # Use convolutional outputs for style and content features
    style_layers = [
        'block1_conv2_64',
        'block2_conv2_128',
        'block3_conv3_192',
        'block4_conv3_256',
        'block5_conv3_256'
    ]
    content_layer = 'block4_conv3_256'

    outputs = [base_model.get_layer(name).output for name in style_layers + [content_layer]]
    feature_extractor = models.Model(inputs=base_model.input, outputs=outputs, name="CustomVGG19_FeatureExtractor")

    return feature_extractor

# Build and save feature extractor
feature_model = build_feature_extractor()
feature_model.summary()
feature_model.save("C:/7th Sem Project/TarunProject/custom_vgg19_feature_extractor_fixed.keras")
print("\n✅ Feature extractor with convolutional outputs saved successfully!")
