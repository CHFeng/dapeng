import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# DenseNet201 default image size is 224
# Xception default image size is 299
IMG_SIZE = 224
MODE_NAME = 'DenseNet201'
# set the GPU memory limit
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# load model
model = load_model('./{}_retrained_v2.h5'.format(MODE_NAME))
classes = ['ambulance', 'fire_engine', 'pickup_truck', 'police_car']


def detect_car_classified(frame):
    # preprocessing for detection model
    car_img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    car_img = car_img.astype("float") / 255.0
    car_img = img_to_array(car_img)
    car_img = np.expand_dims(car_img, axis=0)
    # apply detection on frame
    confArr = model.predict(car_img)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
    # get label with max accuracy
    idx = np.argmax(confArr)

    return classes[idx]