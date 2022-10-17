import time

from src import TFVars
from src import DataProc


if __name__ == "__main__":
    nn_input_size = (224, 224, 3)
    trt_path = "./model/tensorrt/efficientbet_b0_imagenet.trt"
    tf_keras_path = "./model/tf_keras/efficientbet_b0_imagenet_0_224_tf.h5"
    img_path = "./image/tiger_cat.jpg"
    imagenet_label_path = "./model/imagenet_label.json"
    
    device_index = "0"
    device_memory_percent = 0.1


    TFVars.devices()
    TFVars.device_index = device_index
    TFVars.device_memory_percent = device_memory_percent

    from src import InitTFSetup
    from src import TFProc
    from src import TensorRTProc
    
    imagenet_proc = DataProc.ImageNetProc(imagenet_label_path)
    img_tfdata = DataProc.im2tf(img_path, nn_input_size)
    tf_keras = TFProc.Classification(tf_keras_path, nn_input_size)
    trt_engine = TensorRTProc.Classification(trt_path, nn_input_size)
    
    time_1 = time.time()
    tfk_max_index, tfk_max_score = tf_keras.inference(img_tfdata)
    time_2 = time.time()
    trt_max_index, trt_max_score = trt_engine.inference(img_tfdata)
    time_3 = time.time()

    print("Source image " + str(nn_input_size) + ":", img_path)
    print("tf.keras inference (" + str(time_2 - time_1) + " second):",  imagenet_proc.label(tfk_max_index), tfk_max_score)
    print("TensorRT inference (" + str(time_3 - time_2) + " second):", imagenet_proc.label(trt_max_index), trt_max_score)