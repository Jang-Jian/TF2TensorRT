import os

from src import TFVars


device_index = TFVars.device_index
device_memory_percent = TFVars.device_memory_percent


# setup gpu environment.
os.environ["CUDA_VISIBLE_DEVICES"] = device_index

gpu_name = ""
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = device_memory_percent
    tf.compat.v1.Session(config=tf_config)
    
    details = tf.config.experimental.get_device_details(gpus[int(device_index)])
    gpu_name = details.get('device_name', 'Unknown GPU')