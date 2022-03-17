# -*- coding: utf-8 -*-

import signal
import sys
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
from datetime import datetime as dt
# tensoflow
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.saved_model import tag_constants
# local import
import core.utils as utils

YOLOV4_WEIGHTS_PATH = os.path.join(os.getcwd(), "ai_model", "model_data/yolov4-416")
flags.DEFINE_string("weights", YOLOV4_WEIGHTS_PATH, "path to weights file")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
flags.DEFINE_string("output", None, "path to output video")

isPressCtrlC = False


# ctrl+c handler
def signal_handler(signal_num, frame):
    if signal_num == signal.SIGINT.value:
        global isPressCtrlC
        utils.utils.flush_print('To Close all threads now!')
        isPressCtrlC = True


def main(_argv):
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    camIds = []
    nvrConfig = {'account': 'root', 'password': 'root', 'host': '60.249.33.163'}
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
            for rtsp in result['resp']:
                if rtsp['state'] == 'signal_restored' or rtsp['state'] == 'connected':
                    camIds.append(rtsp['origin'])
    except:
        utils.flush_print("Can't not get NVR config from AI_WEB")
        sys.exit()
        # just for test
        # camIds = [
        #     'DESKTOP-F093S18/DeviceIpint.101/SourceEndpoint.video:0:0',
        #     'DESKTOP-F093S18/DeviceIpint.102/SourceEndpoint.video:0:0',
        #     'DESKTOP-F093S18/DeviceIpint.103/SourceEndpoint.video:0:0',
        #     'DESKTOP-F093S18/DeviceIpint.104/SourceEndpoint.video:0:0',
        #     'DESKTOP-F093S18/DeviceIpint.105/SourceEndpoint.video:0:0',
        #     'DESKTOP-F093S18/DeviceIpint.106/SourceEndpoint.video:0:0',
        #     'DESKTOP-F093S18/DeviceIpint.107/SourceEndpoint.video:0:0',
        #     'DESKTOP-F093S18/DeviceIpint.108/SourceEndpoint.video:0:0',
        # ]
    # read video source detect config from json file
    with open(os.path.join(os.getcwd(), "ai_model", "detect_config.json"), 'r') as f:
        detect_config = json.load(f)
    # create AI detect according camara video source
    cams = []
    for camId in camIds:
        rtspUrl = "rtsp://{}:{}@{}:554/hosts/{}".format(nvrConfig['account'], nvrConfig['password'], nvrConfig['host'], camId)
        try:
            cam = Detect(rtspUrl, camId, infer, detect_config[camId])
            cams.append(cam)
        except:
            utils.flush_print(rtspUrl + " can't be created!")
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        fps = 20
        codec = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (1280, 720))
    else:
        out = None

    camIdx = 0
    start_time = dt.now()
    utils.flush_print("Start detection procedure now!")
    while True:
        if isPressCtrlC: break
        if not FLAGS.dont_show:
            img = cams[camIdx].read()
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (1280, 720))
                cv2.imshow('result', img)
                # if output flag is set, save video file
                if FLAGS.output:
                    out.write(img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key >= ord("0") and key <= ord("9"):
                camIdx = key - ord("0")
                if camIdx >= len(cams):
                    utils.flush_print("Only {} video sources".format(len(cams)))
                    camIdx = len(cams) - 1
        else:
            # trigger read every 30 seconds to make sure thread is running
            try:
                diffTime = dt.now() - start_time
                if diffTime.seconds >= 30:
                    for cam in cams:
                        cam.read()
                    start_time = dt.now()
            except Exception as err:
                utils.flush_print("Something wrong on threads:" + str(err))
                break

    # release all threads
    for cam in cams:
        cam.release()
    # release cv2
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # register ctrl+c signal handler to close sub thread
        signal.signal(signal.SIGINT, signal_handler)
        # run main
        app.run(main)
    except SystemExit:
        pass