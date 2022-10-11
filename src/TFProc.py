import numpy as np
import tensorflow as tf

from src import DataProc


class Classification(object):
    def __init__(self, tf_keras_path: str, input_size: (int, int, int) = (224, 224, 3)):
        self.__tf_keras_model = tf.keras.models.load_model(tf_keras_path, compile=False)
        DataProc.model_warmup(self, input_size)
    
    def inference(self, src: np.ndarray) -> (int, float):
        result_tesnor = self.__tf_keras_model(src)
        max_index = np.argmax(result_tesnor[0])
        max_score = float(result_tesnor[0][max_index])

        return (max_index, max_score)