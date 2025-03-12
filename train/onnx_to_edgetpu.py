import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load mô hình ONNX
onnx_model = onnx.load("mobilenetv2.onnx")
tf_rep = prepare(onnx_model)

# Export sang TensorFlow
tf_rep.export_graph("mobilenetv2_tf")

# Chuyển sang TFLite với quantization
converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv2_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Cung cấp dữ liệu đại diện
def representative_dataset():
    for _ in range(100):
        yield [np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Lưu mô hình TFLite
with open("mobilenetv2_quantized.tflite", "wb") as f:
    f.write(tflite_model)