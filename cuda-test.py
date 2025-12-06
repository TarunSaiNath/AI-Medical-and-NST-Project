import tensorflow as tf
import tf2onnx
import numpy
import google.protobuf

print("TensorFlow:", tf.__version__)
print("tf2onnx:", tf2onnx.__version__)
print("NumPy:", numpy.__version__)
print("protobuf:", google.protobuf.__version__)
print("CUDA available:", tf.config.list_physical_devices('GPU'))
