
from src import TFVars
from src import DataProc


if __name__ == "__main__":
    nn_input_size = (224, 224, 3)
    trt_path = "./model/tensorrt_mobilenet_v1/mobilenet_imagenet.trt"
    tf_keras_path = "./model/tf_keras_mobilenet_v1/mobilenet_imagenet_0_224_tf.h5"
    img_path = "./model/tiger_cat.jpg"
    img_tfdata = DataProc.im2tf(img_path, nn_input_size)

    device_index = "0"
    device_memory_percent = 0.1


    TFVars.devices()
    TFVars.device_index = device_index
    TFVars.device_memory_percent = device_memory_percent

    from src import InitTFSetup
    from src import TFProc
    from src import TensorRTProc
    
    tf_keras = TFProc.Classification(tf_keras_path, nn_input_size)
    trt_engine = TensorRTProc.Classification(trt_path, nn_input_size)
    
    tfk_max_index, tfk_max_score = tf_keras.inference(img_tfdata)
    trt_max_index, trt_max_score = trt_engine.inference(img_tfdata)

    print("tf.keras inference:", tfk_max_index, tfk_max_score)
    print("TensorRT inference:", trt_max_index, trt_max_score)