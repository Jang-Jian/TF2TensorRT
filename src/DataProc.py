import cv2
import json
import numpy as np


def model_warmup(model: object, nn_input_size: (int, int, int)):
    """
    model warm up for first inference.
    """
    data = np.random.rand(1, nn_input_size[0], 
                          nn_input_size[1], nn_input_size[2])
    data = data.astype("float32")
    model.inference(data)

def im2tf(img_path: str, out_size: (int, int) = (224, 224), 
          dtype: np.dtype = np.float) -> np.ndarray:
    """
    read a image to format of tf input. 
    """
    img_data = cv2.imread(img_path)
    if len(out_size) > 1:
        img_data = cv2.resize(img_data, (out_size[0], out_size[1]))
    img_data = np.expand_dims(img_data, axis=0).astype(dtype)
    return img_data


class ImageNetProc(object):
    def __init__(self, json_path: str = "./model/imagenet_label.json") -> None:
        with open(json_path, newline='') as jsonfile:
            self.__data = json.load(jsonfile)

    def label(self, index: int) -> str:
        """
        get name of class for imagenet.
        """
        label_info = self.__data[str(index)]
        return label_info[1]