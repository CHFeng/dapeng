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
FETCH_CAMERA_URL = "http://{server_domain}/api/cameraPlan/list"
flags.DEFINE_string("weights", YOLOV4_WEIGHTS_PATH, "path to weights file")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
flags.DEFINE_string("output", None, "path to output video")

isPressCtrlC = False


def fetch_camera_models() -> list:
    '''
    向思納捷WEB查詢攝影機所屬辨識模組清單
    response data format:
    {
        "code": "0",
        "message": "",
        "data": [
            {
                "cameraName": "camera1",
                "models": [1, 2]
            }
        ]
    }
    '''
    try:
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
        result = requests.get(FETCH_CAMERA_URL, headers=headers)
        if result.status_code != requests.codes.ok:
            utils.flush_print("send fetch_camera_models Err:" + json.loads(result.text))
        else:
            result = json.loads(result.text)
            return result['data']
    except Exception as err:
        utils.flush_print("fetch_camera_models Err:" + str(err))
        return []


# ctrl+c handler
def signal_handler(signal_num, frame):
    if signal_num == signal.SIGINT.value:
        global isPressCtrlC
        utils.flush_print('To Close all threads now!')
        isPressCtrlC = True


def main(_argv):
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    camInfos = []
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
                    camInfos.append({'id': rtsp['origin'], 'cameraName': rtsp['friendlyNameShort']})
    except:
        utils.flush_print("Can't not get NVR config from AI_WEB")
        sys.exit()
        # just for test
        # camInfos = [
        #     {
        #         'id': 'DESKTOP-F093S18/DeviceIpint.101/SourceEndpoint.video:0:0',
        #         'cameraName': '攝影機1'
        #     },
        #     {
        #         'id': 'DESKTOP-F093S18/DeviceIpint.102/SourceEndpoint.video:0:0',
        #         'cameraName': '攝影機2'
        #     },
        #     {
        #         'id': 'DESKTOP-F093S18/DeviceIpint.103/SourceEndpoint.video:0:0',
        #         'cameraName': '攝影機3'
        #     },
        #     {
        #         'id': 'DESKTOP-F093S18/DeviceIpint.104/SourceEndpoint.video:0:0',
        #         'cameraName': '攝影機4'
        #     },
        #     {
        #         'id': 'DESKTOP-F093S18/DeviceIpint.105/SourceEndpoint.video:0:0',
        #         'cameraName': '攝影機5'
        #     },
        #     {
        #         'id': 'DESKTOP-F093S18/DeviceIpint.106/SourceEndpoint.video:0:0',
        #         'cameraName': '攝影機6'
        #     },
        #     {
        #         'id': 'DESKTOP-F093S18/DeviceIpint.107/SourceEndpoint.video:0:0',
        #         'cameraName': '攝影機7'
        #     },
        #     {
        #         'id': 'DESKTOP-F093S18/DeviceIpint.108/SourceEndpoint.video:0:0',
        #         'cameraName': '攝影機8'
        #     },
        # ]
    # 向思納捷WEB查詢攝影機所屬辨識模組清單
    cameraModes = fetch_camera_models()
    # read video source detect config from json file
    with open(os.path.join(os.getcwd(), "ai_model", "detect_config.json"), 'r') as f:
        detect_config = json.load(f)
    # create AI detect according camara video source
    cams = []
    for camInfo in camInfos:
        rtspUrl = "rtsp://{}:{}@{}:554/hosts/{}".format(nvrConfig['account'], nvrConfig['password'], nvrConfig['host'], camInfo['id'])
        # 比對思納捷回傳的攝影機模組型態,來調整detect_config中notify的數值是否修改
        for cameraMode in cameraModes:
            if cameraMode['cameraName'] == camInfo['cameraName']:
                # 如果該攝影機的modes包含了3,4,5則表示為停車場攝影機,需要主動通知思納捷
                if any(a in cameraMode['models'] for a in (3, 4, 5)):
                    for config in detect_config[camInfo['id']]['detect_configs']:
                        config['notify'] = True
        try:
            cam = Detect(rtspUrl, camInfo, infer, detect_config[camInfo['id']])
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