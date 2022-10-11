import tensorflow as tf


model_path = "./tf_mobilenet/mobilenet_imagenet_0_224_tf.h5"
mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=None,
                                                        alpha=1.0,
                                                        depth_multiplier=1,
                                                        dropout=0.001,
                                                        include_top=True,
                                                        weights='imagenet',
                                                        input_tensor=None,
                                                        pooling=None,
                                                        classes=1000,
                                                        classifier_activation='softmax')

tf.keras.models.save_model(mobilenet, model_path)