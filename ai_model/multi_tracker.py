import os
import tensorflow as tf
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2
from detect import Detect
from absl import app, flags, logging
from absl.flags import FLAGS
# tensoflow
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.saved_model import tag_constants
# local import
import core.utils as utils

flags.DEFINE_string("weights", "./checkpoints/yolov4-416",
                    "path to weights file")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
flags.DEFINE_boolean("info", True, "show detailed info of tracked objects")
flags.DEFINE_string("allow_classes", "person,car,truck,bus,motorbike",
                    "allowed classes")


def main(_argv):
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    saved_model_loaded = tf.saved_model.load(FLAGS.weights,
                                             tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    rtspUrl = "rtsp://user1:user10824@60.249.33.163:554/hosts/DESKTOP-F093S18/DeviceIpint.103/SourceEndpoint.video:0:0"
    rtspUrl2 = "rtsp://user1:user10824@60.249.33.163:554/hosts/DESKTOP-F093S18/DeviceIpint.101/SourceEndpoint.video:0:0"
    rtspUrl3 = "rtsp://user1:user10824@60.249.33.163:554/hosts/DESKTOP-F093S18/DeviceIpint.102/SourceEndpoint.video:0:0"
    cam = Detect(rtspUrl, infer, FLAGS)
    cam2 = Detect(rtspUrl2, infer, FLAGS)
    cam3 = Detect(rtspUrl3, infer, FLAGS)

    camIdx = 0
    while True:
        if camIdx == 0:
            img = cam.read()
        elif camIdx == 1:
            img = cam2.read()
        else:
            img = cam3.read()

        if img is not None:
            img = cv2.resize(img, (1280, 720))
            cv2.imshow('result', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("0"):
            camIdx = 0
        elif key == ord("1"):
            camIdx = 1
        elif key == ord("2"):
            camIdx = 2
    cam.release()
    cam2.release()
    cam3.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass