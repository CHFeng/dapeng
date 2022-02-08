import os
import tensorflow as tf
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2
import requests
import json
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
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
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

    camIds = []
    nvrConfig = {
        'account': 'root',
        'password': 'root',
        'host': '60.249.33.163'
    }
    # get NVR config
    try:
        url = "http://localhost:8000/nvr_config"
        result = requests.get(url)
        if result.status_code == requests.codes.ok:
            nvrConfig = json.loads(result.text)
        # get NVR video source
        url = "http://localhost:8000/check_nvr"
        result = requests.get(url)
        if result.status_code == requests.codes.ok:
            result = json.loads(result.text)
            for rtsp in result["resp"]:
                if rtsp["state"] == 'signal_restored' or rtsp[
                        "state"] == 'connected':
                    camIds.append(rtsp["origin"])
    except:
        # exit("Can't not get NVR config")
        # just for test
        print("Can't not get NVR config, use default value")
        camIds = [
            'DESKTOP-F093S18/DeviceIpint.101/SourceEndpoint.video:0:0',
            'DESKTOP-F093S18/DeviceIpint.102/SourceEndpoint.video:0:0',
            'DESKTOP-F093S18/DeviceIpint.103/SourceEndpoint.video:0:0',
            'DESKTOP-F093S18/DeviceIpint.104/SourceEndpoint.video:0:0',
            'DESKTOP-F093S18/DeviceIpint.105/SourceEndpoint.video:0:0',
            'DESKTOP-F093S18/DeviceIpint.106/SourceEndpoint.video:0:0',
            'DESKTOP-F093S18/DeviceIpint.107/SourceEndpoint.video:0:0',
            'DESKTOP-F093S18/DeviceIpint.108/SourceEndpoint.video:0:0',
        ]
    # read video source detect config from json file
    with open('detect_config.json', 'r') as f:
        detect_config = json.load(f)
    # create AI detect according camara video source
    cams = []
    for camId in camIds:
        rtspUrl = "rtsp://{}:{}@{}:554/hosts/{}".format(
            nvrConfig["account"], nvrConfig["password"], nvrConfig["host"],
            camId)
        detect_config[camId]["allow_classes"] = FLAGS.allow_classes
        cam = Detect(rtspUrl, camId, infer, detect_config[camId])
        cams.append(cam)

    camIdx = 0
    while True:
        if not FLAGS.dont_show:
            img = cams[camIdx].read()
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (1280, 720))
                cv2.imshow('result', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key >= ord("1") and key <= ord("8"):
                camIdx = key - ord("1")
                if camIdx >= len(camIds):
                    print("Only {} video sources".format(len(camIds)))
                    camIdx = len(camIds) - 1
        else:
            key = input('threads are running, you can press "q" to exit!\n')
            if key == 'q':
                break

    # release thread
    for cam in cams:
        cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass