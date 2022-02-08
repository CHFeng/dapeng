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
# # deep sort imports
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching


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
        "person": "PEOPLE"
    }
    records = []
    body = []
    # combine all types value into list except in & out value is 0
    for class_type in allowed_classes:
        type = CUSTOM_TYPE_LIST[class_type]
        data = {'class_type': type, 'inValue': 0, 'outValue': 0}
        key = class_type + "-up"
        if key in counter:
            data["inValue"] += counter[key]
        key = class_type + "-down"
        if key in counter:
            data["outValue"] += counter[key]
        if data['inValue'] > 0 or data['outValue'] > 0:
            records.append(data)
    # write every record into database
    for record in records:
        body.append({
            'camId': camId,
            'time': dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': record['class_type'],
            'inValue': record['inValue'],
            'outValue': record['outValue']
        })
    # send post request
    try:
        url = "http://localhost:8000/records"
        body = json.dumps({'records': body})
        result = requests.post(url, data=body)
        if result.status_code != requests.codes.ok:
            print("send request Err:" + json.loads(result.text))
    except Exception as err:
        print("write into DB Err:" + str(err))

    print("write camId:{} counter into DB successfully! ".format(camId) +
          dt.now().strftime("%Y-%m-%d %H:%M:%S"))


def yolov4(cam):
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
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
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
    # cam.img_handle = utils.draw_bbox(cam.img_handle,
    #                                  pred_bbox,
    #                                  classes=cam.class_names)
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
    detections = [
        Detection(bbox, score, class_name,
                  feature) for bbox, score, class_name, feature in zip(
                      bboxes, scores, names, features)
    ]
    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, NMS_MAX_OVERLAP,
                                                scores)
    detections = [detections[i] for i in indices]

    return detections


def check_track_direction(cam, bbox, class_name, track_id):
    # the detection area line
    line_pos_1 = cam.args.detect_pos - cam.args.detect_distance
    line_pos_2 = cam.args.detect_pos + cam.args.detect_distance
    # draw the detection area line on the screen
    if cam.args.flow_direction == "horizontal":
        # check detection area not over the screen
        if line_pos_1 > cam.height or line_pos_2 > cam.height:
            # print("the detection area:{}~{} over the screen:{}".format(
            #     line_pos_1, line_pos_2, cam.height))
            line_pos_1 = cam.height - cam.args.detect_distance // 2
            line_pos_2 = cam.height + cam.args.detect_distance // 2
        cv2.line(cam.img_handle, (cam.args.detect_pos_x, line_pos_1),
                 (cam.width, line_pos_1), (255, 0, 0), 2)
        cv2.line(cam.img_handle, (cam.args.detect_pos_x, line_pos_2),
                 (cam.width, line_pos_2), (255, 0, 0), 2)
    else:
        # check detection area not over the screen
        if line_pos_1 > cam.width or line_pos_2 > cam.width:
            # print("the detection area:{}~{} over the screen:{}".format(
            #     line_pos_1, line_pos_2, cam.width))
            line_pos_1 = cam.width - cam.args.detect_distance // 2
            line_pos_2 = cam.width + cam.args.detect_distance // 2
        cv2.line(cam.img_handle, (line_pos_1, cam.args.detect_pos_y),
                 (line_pos_1, cam.height), (255, 0, 0), 2)
        cv2.line(cam.img_handle, (line_pos_2, cam.args.detect_pos_y),
                 (line_pos_2, cam.height), (255, 0, 0), 2)

    # calcuate position of bbox and draw circle on
    x_cen = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
    y_cen = int(bbox[1] + (bbox[3] - bbox[1]) / 2)
    # just for debug show the center position of object on screen
    # cv2.circle(cam.img_handle, (x_cen, y_cen), 5, (255, 0, 0), -1)

    # check be tracked object on detection area
    tracked_pos = 0
    if cam.args.flow_direction == "horizontal":
        tracked_pos = y_cen
    else:
        tracked_pos = x_cen
    if tracked_pos > (cam.args.detect_pos -
                      cam.args.detect_distance) and tracked_pos < (
                          cam.args.detect_pos + cam.args.detect_distance):
        checkDirection = True
        # 當有設定cam.args.detect_pos_y or cam.args.detect_pos_x 需要物件位置大於設定值才計數
        if cam.args.detect_pos_y > 0 and y_cen < cam.args.detect_pos_y:
            checkDirection = False
        elif cam.args.detect_pos_x > 0 and x_cen < cam.args.detect_pos_y:
            checkDirection = False

        if checkDirection:
            existed = False
            for obj in cam.detect_objs:
                if obj['id'] == track_id:
                    existed = True
                    if cam.args.flow_direction == "horizontal":
                        orig_pos = obj['y_orig']
                    else:
                        orig_pos = obj['x_orig']
                    diff = tracked_pos - orig_pos
                    # check object direction if it is none
                    if obj['direction'] == "none":
                        if diff >= cam.args.object_speed:
                            obj['direction'] = "down"
                            orig_pos = tracked_pos
                        elif diff <= -cam.args.object_speed:
                            obj['direction'] = "up"
                            orig_pos = tracked_pos
            # to append object into array if object doesn't existd
            if not existed:
                obj = {
                    "class": class_name,
                    "id": track_id,
                    "y_orig": y_cen,
                    "x_orig": x_cen,
                    "direction": "none"
                }
                cam.detect_objs.append(obj)


def deep_sort(cam, detections):
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
        check_track_direction(cam, bbox, class_name, track.track_id)
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
                int(bbox[0]) +
                (len(class_name) + len(str(track.track_id))) * 17,
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
    # define counter for every objects
    counter = {}
    for name in cam.allowed_classes:
        key_up = name + "-up"
        key_down = name + "-down"
        counter[key_up] = 0
        counter[key_down] = 0
    # record objects direction
    for obj in cam.detect_objs:
        if obj['direction'] == "none":
            continue
        key = obj['class'] + "-" + obj['direction']
        counter[key] += 1
    # show object direction counter value on screen
    idx = 0
    for key in counter:
        if counter[key] == 0:
            continue
        labelName = key
        if cam.args.flow_direction == "vertical":
            if "up" in key:
                labelName = key.replace("up", "IN")
            elif "down" in key:
                labelName = key.replace("down", "OUT")
        # just for debug, show result on image
        cv2.putText(cam.img_handle, "{}:{}".format(labelName, counter[key]),
                    (cam.width // 3, 35 + idx * 25 * FONT_SCALE), 0,
                    FONT_SCALE, (255, 0, 0), 1)
        idx += 1
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
        cv2.putText(frame, 'FPS: {:.2f}'.format(totalFps), (200, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3.0, (32, 32, 32), 4, cv2.LINE_AA)
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
            print("Error grabbing frame from movie! {}".format(cam.rtspUrl))
            cam.vid.release()
            cam.vid = cv2.VideoCapture(cam.rtspUrl)
            continue
        # get current frame count
        frameCount = int(cam.vid.get(cv2.CAP_PROP_POS_FRAMES))
        # only execute once every 4 frames
        if frameCount % 4 == 0:
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
    def __init__(self, rtspUrl, camId, infer, args) -> None:
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        # initialize deep sort
        model_filename = "model_data/mars-small128.pb"
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        self.tracker = Tracker(metric, max_age=120)
        # initialize yolov4
        self.infer = infer
        # initialize rtsp url
        self.rtspUrl = rtspUrl
        # initialize args
        self.args = args
        # initialize vid
        self.vid = None
        self.width = 0
        self.height = 0
        self.camId = camId
        # read in all class names from config
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        if args.allow_classes:
            self.allowed_classes = args.allow_classes.split(",")
        else:
            # by default allow all classes in .names file
            self.allowed_classes = list(self.class_names.values())
        self.frame_num = 0
        self.detect_objs = []
        self.lastWriteTime = dt.now()
        self.thread_running = False
        self.is_detected = False
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        if self.vid is not None:
            raise RuntimeError('camera is already opened!')
        print('RTSP url is: {}'.format(self.rtspUrl))
        while True:
            self.vid = cv2.VideoCapture(self.rtspUrl)
            self._start()
            # check cam is opened, else do it again
            if self.is_opened:
                break
            else:
                self.vid.release()

    def isOpened(self):
        return self.is_opened

    def _start(self):
        if not self.vid.isOpened():
            print('Camera: starting while cap is not opened!')
            return

        # Try to grab the 1st image and determine width and height
        _, self.img_handle = self.vid.read()
        if self.img_handle is None:
            print('Camera: vid.read() returns no image!')
            self.is_opened = False
            return

        self.is_opened = True
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("The width:{} height:{}".format(self.width, self.height))
        # start the child thread if not using a video file source
        # i.e. rtsp, usb or onboard
        assert not self.thread_running
        self.thread_running = True
        self.thread = threading.Thread(target=sub_process, args=(self, ))
        self.thread.start()

    def _stop(self):
        if self.thread_running:
            self.thread_running = False
            print('Wait until Thread is terminating')
            self.thread.join()

    def read(self):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if not self.is_opened or not self.is_detected:
            return None
        self.is_detected = False

        return self.img_handle

    def release(self):
        self._stop()
        try:
            self.vid.release()
        except:
            pass
        self.is_opened = False

    def __del__(self):
        self.release()
