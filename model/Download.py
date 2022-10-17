import tensorflow as tf


model_path = "./efficientbet_b0_imagenet_0_224_tf.h5"
efficientbet_b0 = tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet',
                                                                    classes=1000,
                                                                    classifier_activation='softmax')
tf.keras.models.save_model(efficientbet_b0, model_path)