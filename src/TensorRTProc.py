import os
import sys

import onnx
import onnxmltools
import numpy as np
import tensorrt as trt

import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit

from src import DataProc


def tf2onnx(tf_model_path: str, onnx_model_path: str):
    """
    convert model from tensorflow model to onnx.
    """
    tf_keras_model = tf.keras.models.load_model(tf_model_path)
    
    onnx_model = onnxmltools.convert_keras(tf_keras_model)

    # set batch size(dim_value) to 1 for inputs/outputs.
    for in_index in range(len(onnx_model.graph.input)):
        onnx_model.graph.input[in_index].type.tensor_type.shape.dim[0].dim_value = 1
    for out_index in range(len(onnx_model.graph.output)):
        onnx_model.graph.output[out_index].type.tensor_type.shape.dim[0].dim_value = 1
    #onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    #onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1

    #for a in tf_keras_model.output_names:
    #    print(a)

    #print(len(onnx_model.graph.input), len(onnx_model.graph.output))
    #print(onnx_model.graph.input[0].type.tensor_type.shape.dim)
    #print(onnx_model.graph.output[0].type.tensor_type.shape.dim)
    #print(tf_keras_model)
    #sys.exit(0)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_model_path)


def onnx2trt(onnx_model_path: str, trt_engine_path: str, trt_precision_type: str):
    """
    convert model from onnx to tensorrt engine.

    tensorrt 8: https://github.com/NVIDIA-AI-IOT/torch2trt/issues/557
    an exmaple code for build tensorrt model (line 401): https://github.com/NVIDIA/TensorRT/blob/main/demo/BERT/builder.py
    """
    precision_mode = None
    if trt_precision_type == "float32":
        pass
    elif trt_precision_type == "float16":
        precision_mode = trt.BuilderFlag.FP16
    else:
        print("onnx2trt() was not supported or implemented for ", trt_precision_type + ".")
        sys.exit(0)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         trt.Runtime(TRT_LOGGER) as runtime:
        config = builder.create_builder_config()
        config.max_workspace_size = 1<<28 # 256MiB
        if precision_mode != None:
            config.set_flag(precision_mode)

        # Parse model file
        with open(onnx_model_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
        print('Completed parsing of ONNX file')
        engine = None
        if trt.__version__ >= "8":
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
        else:
            engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(trt_engine_path, "wb") as f:
            f.write(engine.serialize())


class WrapHostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


class Classification(object):
    """
    a inference method for classification using the TensorRT engine.
    """
    def __init__(self, trt_path: str = 'saved_model.trt', input_size: (int, int, int) = (224, 224, 3), 
                 batch_size: int = 1):
        self.__engine = None
        self.__batch_size = batch_size
        if not os.path.exists(trt_path):
            print("Classification(): " + trt_path + " model is not existed.")

        #self.cuda_ctx = cuda.Device(0).make_context() # Use GPU:0
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

        with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.__engine = runtime.deserialize_cuda_engine(f.read())

        self.__context = self.__engine.create_execution_context()
        self.__inputs, self.__outputs, self.__bindings, self.__stream = self.__allocate_io_buffers()

        DataProc.model_warmup(self, input_size)

    def __allocate_io_buffers(self) -> tuple:
        """
        allocate device and host buffers.
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        if self.__batch_size > self.__engine.max_batch_size:
            self.__batch_size = self.__engine.max_batch_size

        for binding in self.__engine:
            size = trt.volume(self.__engine.get_binding_shape(binding)) * self.__engine.max_batch_size
            dtype = trt.nptype(self.__engine.get_binding_dtype(binding))

            # Allocate host and device buffers.
            # p.s. cuda.pagelocked_empty() has some bugs in the windows.
            host_mem = np.zeros(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.__engine.binding_is_input(binding):
                inputs.append(WrapHostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(WrapHostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __trt_engine_inference(self):
        """
        inference for tensorrt.
        """
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.__stream) for inp in self.__inputs]
        
        # Run inference.
        self.__context.execute_async(batch_size=self.__batch_size, bindings=self.__bindings, 
                                     stream_handle=self.__stream.handle)
        
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.__stream) for out in self.__outputs]
       
        # Synchronize the stream
        self.__stream.synchronize()
        
        # Return only the host outputs.
        return [out.host for out in self.__outputs]
    
    def inference(self, src: np.ndarray) -> (int, float):
        """
        inference using tensorrt engine.
        """
        #self.__inputs[0].host = tf.reshape(src, [-1])
        # https://forums.developer.nvidia.com/t/how-to-use-tensorrt-by-the-multi-threading-package-of-python/123085/8
        np.copyto(self.__inputs[0].host, src.ravel())
        #trt_outputs = None
        #with engine.create_execution_context() as context:
        trt_outs = self.__trt_engine_inference()

        result_np = trt_outs[0].reshape(1, -1)
        pred_index = np.argmax(result_np)
        pred_score = float(result_np[0][pred_index])
        
        return (pred_index, pred_score)