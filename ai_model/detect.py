# -*- coding: utf-8 -*-

import os
import cv2
import requests
import json
import time
import tensorflow as tf
import threading
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
# local import
from tools import generate_detections as gdet
from core.config import cfg
import core.utils as utils
# deep sort imports
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching
# cnn detect model
from cnn_model.cnn_classify import detect_car_classified

DETECTION_FRAME_RATE = 4
WEB_URL = "DAPENG"


def send_parking_event(cameraName, direction):
    '''
    通知思納捷主機有車輛進入或離開停車場的事件
    cameraName: 攝影機的名稱,相當於"friendlyNameShort"
    direction: "carIn" or "carOut"
    '''
    # send post request
    try:
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
        # 根據API文件設定body內容
        body = {'cameraName': cameraName, 'eventTime': dt.timestamp(dt.now()) * 1000}
        body[direction] = 1
        body = json.dumps(body)
        url = "http://{}/api/nvr/notify".format(WEB_URL)
        result = requests.post(url, data=body, headers=headers)
        if result.status_code != requests.codes.ok:
            utils.flush_print("send_parking_event Err:" + json.loads(result.text))
        utils.flush_print("Send parking event {}-{} to WEB successfully! ".format(cameraName, direction) + dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as err:
        utils.flush_print("send_parking_event Err:" + str(err))


def period_notify(cameraName, carIn, carOut, startTime):
    '''
    每分鐘回報攝影機的辨識結果(出與入)列表
    cameraName: 攝影機的名稱,相當於"friendlyNameShort"
    carIn: 入車數量
    carOut: 出車數量
    startTime: 統計開始時間,時間戳(long time to millisecond)
    '''
    # send post request
    try:
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
        # 根據API文件設定body內容
        body = {'cameraName': cameraName, 'carIn': carIn, 'carOut': carOut, 'startTime': startTime, 'statsType': 5}
        body = json.dumps(body)
        url = "http://{}/api/vehicle/inAndOutStats".format(WEB_URL)
        result = requests.post(url, data=body, headers=headers)
        if result.status_code != requests.codes.ok:
            utils.flush_print("period_notify Err:" + json.loads(result.text))
        utils.flush_print("Send period notify to WEB successfully! " + dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as err:
        utils.flush_print("period_notify Err:" + str(err))


def write_into_db(counter, camId, allowed_classes):
    '''
    將統計資料透過POST寫入DB
    counter type is dict and key is object type & direction.
    e.g. person-down or person-up
    '''
    CUSTOM_TYPE_LIST = {
        "truck": "TRUCK",
        "pickup_truck": "PICKUP_TRUCK",
        "bus": "BUS",
        "car": "AUTOCAR",
        "motorbike": "MOTORCYCLE",
        "bicycle": "BIKE",
        "ambulance": "AMBULANCE",
        "fire_engine": "FIRE_ENGINE",
        "police_car": "POLICE_CAR",
        "person": "PEOPLE",
        "parking_car": "PARKING_CAR"
    }
    records = []
    body = []
    # combine all types value into list except in & out value is 0
    for class_type in allowed_classes:
        type = CUSTOM_TYPE_LIST[class_type]
        data = {'class_type': type, 'inValue': 0, 'outValue': 0, 'inAvgSpeed': 0, 'outAvgSpeed': 0}
        key = class_type + "-up"
        if key in counter:
            data['inValue'] = counter[key]
        key = class_type + "-up-speed"
        if key in counter:
            data['inAvgSpeed'] = counter[key]
        key = class_type + "-down"
        if key in counter:
            data['outValue'] = counter[key]
        key = class_type + "-down-speed"
        if key in counter:
            data['outAvgSpeed'] = counter[key]
        if data['inValue'] > 0 or data['outValue'] > 0:
            records.append(data)
    # write every record into database
    for record in records:
        body.append({
            'camId': camId,
            'time': dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': record['class_type'],
            'inValue': record['inValue'],
            'outValue': record['outValue'],
            'inAvgSpeed': record['inAvgSpeed'],
            'outAvgSpeed': record['outAvgSpeed'],
        })
    # send post request
    try:
        url = "http://localhost:8000/records"
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
        body = json.dumps({'records': body})
        result = requests.post(url, data=body, headers=headers)
        if result.status_code != requests.codes.ok:
            utils.flush_print("send request Err:" + json.loads(result.text))
        utils.flush_print("write camId:{} counter into DB successfully! ".format(camId) + dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as err:
        utils.flush_print("write into DB Err:" + str(err))


def calculate_object_move_speed(x1, y1, x2, y2, frameCount):
    '''
    速度 = 距離 / 時間
    1 pixcel = 0.02m
    影像來源FPS=20,因此1個frame的時間長度為1/20
    但需要考量系統幾個frame才執行一次所以需要 * DETECTION_FRAME_RATE
    不確定原因需要*3後的數值才能與現實狀況相符
    '''
    distance = pow((x2 - x1), 2) + pow((y2 - y1), 2)
    distance = pow(distance, 0.5)
    speed = (distance * 0.02) / (frameCount * (1 / 20) * DETECTION_FRAME_RATE) * 3
    # utils.flush_print(x1, y1, x2, y2, distance, frameCount, speed)
    # convert speed from m/s to km/hr
    speed = int(speed / 1000 * 3600)
    # utils.flush_print("object Speed:{}".format(speed))
    # if speed over 120, it should be wrong
    return speed if speed < 120 else 0


def getCross(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1])


def isPointInMatrix(p1, p2, p3, p4, p):
    '''
    使用向量叉積公式判斷點P是否在凸四邊形內
    P1~P4為四邊形的座標且需要"順時針"帶入
    '''
    return getCross(p1, p2, p) * getCross(p3, p4, p) >= 0 and getCross(p2, p3, p) * getCross(p4, p1, p) >= 0


def yolov4(cam):
    '''
    使用yolov4偵測畫面中的物件
    '''
    # some threshold define
    NMS_MAX_OVERLAP = 1.0
    RESIZE = 416
    IOU_THRESHOLD = 0.45
    SCORE_THRESHOLD = 0.5

    input_size = RESIZE
    image_data = cv2.resize(cam.img_handle, (input_size, input_size))
    image_data = image_data / 255.0
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # run detections
    batch_data = tf.constant(image_data)
    pred_bbox = cam.infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    # combined_non_max_suppression
    (
        boxes,
        scores,
        classes,
        valid_detections,
    ) = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD,
    )
    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = cam.img_handle.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)

    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]
    # just for debug
    # cam.img_handle = utils.draw_bbox(cam.img_handle, pred_bbox, classes=cam.class_names)
    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = cam.class_names[class_indx]
        if class_name not in cam.allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    counter = len(names)
    # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)
    # encode yolo detections and feed to tracker
    features = cam.encoder(cam.img_handle, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, NMS_MAX_OVERLAP, scores)
    detections = [detections[i] for i in indices]

    return detections


def check_track_direction(cam, bbox, class_name, track_id, config):
    '''
    偵測物件移動的方向
    如果偵測區間設定為水平(horizontal),則物件往下移動(Y變大),則位移方向判定為down
    如果偵測區間設定為垂直(vertical),則物件往右移動(X變大),則位移方向判定為down
    '''
    frame = cam.img_handle
    detect_objs = cam.detect_objs
    # the detection area line
    points = np.array([config['leftUp'], config['leftDown'], config['rightDown'], config['rightUp']], np.int32)
    cv2.polylines(frame, pts=[points], isClosed=True, color=(255, 0, 0), thickness=3)
    # calcuate position of bbox and draw circle on
    x_cen = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
    y_cen = int(bbox[1] + (bbox[3] - bbox[1]) / 2)
    # show the center position of object on screen
    cv2.circle(frame, (x_cen, y_cen), 5, (255, 0, 0), -1)
    # check be tracked object on detection area
    checkDirection = False
    tracked_pos = 0
    if isPointInMatrix(config['leftUp'], config['rightUp'], config['rightDown'], config['leftDown'], [x_cen, y_cen]):
        checkDirection = True
        if config['direction'] == "horizontal":
            tracked_pos = y_cen
        else:
            tracked_pos = x_cen

    if checkDirection:
        existed = False
        for obj in detect_objs:
            if obj['id'] == track_id:
                existed = True
                obj['frameCount'] += 1
                if config['direction'] == "horizontal":
                    orig_pos = obj['y_orig']
                else:
                    orig_pos = obj['x_orig']
                diff = tracked_pos - orig_pos
                # check object direction if it is none
                if obj['direction'] == "none":
                    if diff >= config['threshold']:
                        obj['direction'] = "down"
                        orig_pos = tracked_pos
                    elif diff <= -config['threshold']:
                        obj['direction'] = "up"
                        orig_pos = tracked_pos
                    # the direction has been detected, calculate speed
                    if obj['direction'] != "none":
                        obj['speed'] = calculate_object_move_speed(obj['x_orig'], obj['y_orig'], x_cen, y_cen, obj['frameCount'])
                        # 只有原本判斷為car or truck會需要進一步分類
                        if class_name == "car" or class_name == "truck":
                            left, top, right, bottom = bbox
                            cut_img = cv2.cvtColor(frame[int(top):int(bottom), int(left):int(right)], cv2.COLOR_RGB2BGR)
                            class_name = detect_car_classified(cut_img, class_name)
                        # 通知思納捷主機有物件進入/離開停車場事件
                        if class_name == "car" and 'notify' in config and config['notify'] == True:
                            # 將class_type改為parking_car,方便後續資料的判定
                            obj['class'] = "parking_car"
                            # 判斷物件移動方向up是入停車場或者down是入停車場
                            eventType = ""
                            if 'intoParking' in config and config['intoParking'] == "down":
                                eventType = "carIn" if obj['direction'] == "down" else "carOut"
                            else:
                                eventType = "carIn" if obj['direction'] == "up" else "carOut"
                            send_parking_event(cam.camName, eventType)
                        # utils.flush_print("{}-ID:{} direction:{}".format(class_name, track_id, obj['direction']))
        # to append object into array if object doesn't existd
        if not existed:
            obj = {'class': class_name, 'id': track_id, 'y_orig': y_cen, 'x_orig': x_cen, 'direction': "none", 'frameCount': 0, 'speed': 0}
            detect_objs.append(obj)


def deep_sort(cam, detections):
    '''
    使用deep sort演算法判斷連續畫面中的物件
    '''
    # Call the tracker
    cam.tracker.predict()
    cam.tracker.update(detections)
    # initialize color map
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    # update tracks
    for track in cam.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        for config in cam.detect_configs:
            check_track_direction(cam, bbox, class_name, track.track_id, config)
        # draw bbox on screen, just for debug
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(
            cam.img_handle,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            2,
        )
        cv2.rectangle(
            cam.img_handle,
            (int(bbox[0]), int(bbox[1] - 30)),
            (
                int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                int(bbox[1]),
            ),
            color,
            -1,
        )
        cv2.putText(
            cam.img_handle,
            class_name + "-" + str(track.track_id),
            (int(bbox[0]), int(bbox[1] - 10)),
            0,
            0.75,
            (255, 255, 255),
            2,
        )


def counter_object(cam):
    # the font scale to show object counter result on frame
    FONT_SCALE = 2
    # the time interval(seconds) to write counter value into DB
    WRITE_DB_INTERVAL = 5 * 60
    # 停車場車輛進出辨識-週期回報時間
    PARKING_NOTIFY_INTERVAL = 1 * 60
    # define counter for every objects
    counter = {}
    for name in cam.allowed_classes:
        key_up = name + "-up"
        key_down = name + "-down"
        key_up_speed = name + "-up-speed"
        key_down_speed = name + "-down-speed"
        counter[key_up] = 0
        counter[key_down] = 0
        counter[key_up_speed] = 0
        counter[key_down_speed] = 0
    # record objects direction
    for obj in cam.detect_objs:
        if obj['direction'] == "none":
            continue
        key = obj['class'] + "-" + obj['direction']
        counter[key] += 1
        # calculate object average speed
        if obj['speed']:
            key = key + "-speed"
            counter[key] = (counter[key] + obj['speed']) // 2 if counter[key] > 0 else obj['speed']
    # show object direction counter value on screen
    idx = 0
    for key in counter:
        if counter[key] == 0:
            continue
        labelName = key
        # just for debug, show result on image
        cv2.putText(cam.img_handle, "{}:{}".format(labelName, counter[key]), (cam.width // 3, 35 + idx * 25 * FONT_SCALE), 0, FONT_SCALE, (255, 0, 0),
                    1)
        idx += 1
    # 停車場車輛進出辨識-週期回報,有PARKING型態的紀錄才回報
    diffTime = dt.now() - cam.lastNotifyTime
    if diffTime.seconds >= PARKING_NOTIFY_INTERVAL:
        # update last time stamp
        cam.lastNotifyTime = dt.now()
        if counter['parking_car-up'] or counter['parking_car-down']:
            period_notify(cam.camName, counter['parking_car-up'], counter['parking_car-down'], dt.timestamp(cam.lastWriteTime) * 1000)
    # wirte data into DB every time inteval
    diffTime = dt.now() - cam.lastWriteTime
    if diffTime.seconds >= WRITE_DB_INTERVAL:
        # update last time stamp
        cam.lastWriteTime = dt.now()
        write_into_db(counter, cam.camId, cam.allowed_classes)
        # reset counter
        cam.detect_objs = []


def calculate_fps(frame, fpsArr, start_time):
    fps = 1.0 / (time.time() - start_time)
    fpsArr.append(fps)
    if len(fpsArr) == 60:
        totalFps = 0
        for fps in fpsArr:
            totalFps += fps
        totalFps /= 60
        cv2.putText(frame, 'FPS: {:.2f}'.format(totalFps), (200, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (32, 32, 32), 4, cv2.LINE_AA)
        fpsArr.pop(0)


def sub_process(cam):
    """This 'sub_process' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    then execute object detector(yolov4) and tracker(deep sort)
    """
    fpsArr = []
    while cam.thread_running and cam.vid.isOpened():
        start_time = time.time()
        # ret, cam.img_handle = cam.vid.read()
        ret = cam.vid.grab()
        if not ret:
            utils.flush_print("Error grabbing frame from source! {}".format(cam.rtspUrl))
            cam.vid.release()
            cam.vid = cv2.VideoCapture(cam.rtspUrl)
            continue
        # get current frame count
        frameCount = int(cam.vid.get(cv2.CAP_PROP_POS_FRAMES))
        # only execute once every 4 frames
        if frameCount % DETECTION_FRAME_RATE == 0:
            ret, cam.img_handle = cam.vid.retrieve()
            # convert to RGB
            cam.img_handle = cv2.cvtColor(cam.img_handle, cv2.COLOR_BGR2RGB)
            # run detection by yolov4
            detections = yolov4(cam)
            deep_sort(cam, detections)
            counter_object(cam)
            cam.is_detected = True
            # calculate FPS
            calculate_fps(cam.img_handle, fpsArr, start_time)

    cam.thread_running = False


class Detect:
    def __init__(self, rtspUrl, camInfo, infer, args) -> None:
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        # initialize deep sort
        model_filename = os.path.join(os.getcwd(), "ai_model", "model_data/mars-small128.pb")
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker,物件消失5秒後移除追蹤, 物件只要出現一次就開始追蹤
        self.tracker = Tracker(metric, max_age=DETECTION_FRAME_RATE * 5, n_init=1)
        # initialize yolov4
        self.infer = infer
        # initialize rtsp url
        self.rtspUrl = rtspUrl
        # initialize vid
        self.vid = None
        self.img_handle = None
        self.width = 0
        self.height = 0
        self.camId = camInfo['id']
        self.camName = camInfo['cameraName']
        # read in all class names from config
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        if 'allow_classes' in args and args['allow_classes'] != None:
            self.allowed_classes = args['allow_classes'].split(",")
        else:
            self.allowed_classes = ["person", "car", "truck", "bus", "motorbike"]
        # 停車場車輛的類型
        self.allowed_classes.append("parking_car")
        # initialize detect config
        self.detect_configs = args['detect_configs']
        # define some flags
        self.frame_num = 0
        self.detect_objs = []
        self.lastWriteTime = dt.now()
        self.lastNotifyTime = dt.now()
        self.thread_running = False
        self.is_detected = False
        self.is_opened = False
        # try to open the camera
        self._open()

    def _open(self):
        """Open camera based on command line arguments."""
        if self.vid is not None:
            raise RuntimeError('camera is already opened!')
        utils.flush_print('RTSP url is: {}'.format(self.rtspUrl))
        openCount = 0
        while True:
            self.vid = cv2.VideoCapture(self.rtspUrl)
            self._start()
            openCount += 1
            # check cam is opened, else do it again
            if self.is_opened:
                break
            elif openCount > 2:
                # if open failed over than 2 times, raise exception
                raise RuntimeError('RTSP url can not be read!')
            else:
                self.vid.release()

    def isOpened(self):
        return self.is_opened

    def _start(self):
        if not self.vid.isOpened():
            utils.flush_print('Camera: starting while cap is not opened!')
            return

        # Try to grab the 1st image and determine width and height
        _, self.img_handle = self.vid.read()
        if self.img_handle is None:
            utils.flush_print('Camera: vid.read() returns no image!')
            self.is_opened = False
            return

        self.is_opened = True
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        utils.flush_print("The width:{} height:{}".format(self.width, self.height))
        for config in self.detect_configs:
            # 判斷偵測範圍右邊的點是否超過實際影像的寬
            if config['rightUp'][0] > self.width:
                config['rightUp'][0] = self.width
            if config['rightDown'][0] > self.width:
                config['rightDown'][0] = self.width
            # 判斷偵測範圍下邊的點是否超過實際影像的高
            if config['leftDown'][1] > self.height:
                config['leftDown'][1] = self.height
            if config['rightDown'][1] > self.height:
                config['rightDown'][1] = self.height
            # 計算物件位移方向的threshold
            if config['direction'] == "horizontal":
                # 水平方向的判定,計算上下兩邊的Y間隔距離,使用較小的數值/2為threshold
                y_diff_1 = abs(config['leftUp'][1] - config['leftDown'][1])
                y_diff_2 = abs(config['rightUp'][1] - config['rightDown'][1])
                if y_diff_1 < y_diff_2:
                    config['threshold'] = y_diff_1 // 2
                else:
                    config['threshold'] = y_diff_2 // 2
                # utils.flush_print(y_diff_1, y_diff_2, config['threshold'])
            elif config['direction'] == "vertical":
                # 垂直方向的判定,計算左右兩邊的X間隔距離,使用較小的數值/2為threshold
                x_diff_1 = abs(config['leftUp'][0] - config['rightUp'][0])
                x_diff_2 = abs(config['leftDown'][0] - config['rightDown'][0])
                if x_diff_1 < x_diff_2:
                    config['threshold'] = x_diff_1 // 2
                else:
                    config['threshold'] = x_diff_2 // 2
                # utils.flush_print(x_diff_1, x_diff_2, config['threshold'])

        assert not self.thread_running
        self.thread_running = True
        self.thread = threading.Thread(target=sub_process, args=(self, ))
        self.thread.start()

    def _stop(self):
        if self.thread_running:
            self.thread_running = False
            utils.flush_print('Wait until thread is terminated')
            self.thread.join()

    def read(self):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if self.thread_running:
            if not self.is_opened or not self.is_detected:
                return None
            self.is_detected = False

            return self.img_handle
        else:
            self.vid.release()
            err_msg = "RTSP url {} is failed".format(self.rtspUrl)
            raise RuntimeError(err_msg)

    def release(self):
        self._stop()
        try:
            self.vid.release()
        except:
            pass
        self.is_opened = False

    def __del__(self):
        self.release()
