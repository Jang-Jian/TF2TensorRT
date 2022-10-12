# TF2TensorRT

Convert tensorflow.keras model to NVIDIA TensorRT engine (*.trt).

## Main4ModelCvter.py

These parameters are presented in the *`Main4ModelCvter.py`*.

* action: A action for model transformation, which support tensorflow to onnx (*`tf2onnx`*) & onnx to tensorrt (*`onnx2trt`*).
* precision: This is used for *`onnx2trt`*, and it supports two modes, included *`float32`* & "float16".
* src_path: Inputed path for action used.
* dst_path: Outputed path for action used.
* device_index: GPU device index.
* device_memory_percent: Percent of gpu memory usage (range: 0~1). 

## Main4TRTInference.py

* nn_input_size: Assign nn input size, such as (224, 224, 3).
* trt_path: Saved path for tensorrt engine (*trt).
* tf_keras_path: Saved path for tensorflow.keras model.
* img_path: Saved path for image data.
* device_index: GPU device index.
* device_memory_percent: Percent of gpu memory usage (range: 0~1). 