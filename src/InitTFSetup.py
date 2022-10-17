import os
import sys

from src import TFVars


device_index = TFVars.device_index
device_memory_percent = TFVars.device_memory_percent

gpu_name = ""
os.environ["CUDA_VISIBLE_DEVICES"] = device_index
import tensorflow as tf

# setup gpu environment.
if tf.test.is_gpu_available():
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = device_memory_percent
    tf.compat.v1.Session(config=tf_config)

    gpus = tf.config.list_physical_devices('GPU')
    details = tf.config.experimental.get_device_details(gpus[0])
    gpu_name = details.get('device_name', 'Unknown GPU')
else:
    print("It is not supported CUDA enviornment. Please check and revise it.")
    sys.exit(0)