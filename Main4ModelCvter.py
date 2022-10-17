"""
tf2onnx: convert model from tensorflow-keras to onnx.
onnx2nvtrt: convert model from onnx to tensorrt.

These api version are worked in the Windows:
onnx=='1.12.0'
tf2onnx=='1.9.1'
onnxmltools=='1.9.1'
"""

from src import TFVars


if __name__ == "__main__":
    action = "onnx2trt" # "tf2onnx" or "onnx2trt".
    precision = "float16" # "float32", "float16" & "".
    #src_path = "./model/tf_keras/efficientbet_b0_imagenet_0_224_tf.h5"
    #dst_path = "./model/onnx/efficientbet_b0_imagenet.onnx"
    src_path = "./model/onnx/efficientbet_b0_imagenet.onnx"
    dst_path = "./model/tensorrt/efficientbet_b0_imagenet.trt"

    device_index = "0"
    device_memory_percent = 0.1


    TFVars.devices()
    TFVars.device_index = device_index
    TFVars.device_memory_percent = device_memory_percent

    from src import InitTFSetup
    from src import TensorRTProc

    print("Current using CUDA device name:", InitTFSetup.gpu_name)
    if action == "tf2onnx":
        TensorRTProc.tf2onnx(src_path, dst_path)
    elif action == "onnx2trt":
        TensorRTProc.onnx2trt(src_path, dst_path, precision)