# -*- coding: utf-8 -*-

import os
import tensorflow as tf
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
import requests
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime as dt
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.saved_model import tag_constants
from absl.flags import FLAGS
from absl import app, flags, logging
# local import
from core.config import cfg
import core.utils as utils
from tools import generate_detections as gdet
# deep sort imports
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching

YOLOV4_WEIGHTS_PATH = os.path.join(os.getcwd(), "ai_model", "model_data/yolov4-416")
flags.DEFINE_string("weights", YOLOV4_WEIGHTS_PATH, "path to weights file")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
flags.DEFINE_boolean("info", True, "show detailed info of tracked objects")
# the setting of object flow direction
flags.DEFINE_string("direction", "vertical", "horizontal or vertical")
flags.DEFINE_list("leftUp", [2220, 700], "the detection point on left-up")
flags.DEFINE_list("leftDown", [2120, 1200], "the detection point on left-down")
flags.DEFINE_list("rightUp", [2480, 600], "the detection point on right-up")
flags.DEFINE_list("rightDown", [2180, 1200], "the detection point on right-down")
flags.DEFINE_boolean("frame_debug", False, "show frame one by one for debug")
flags.DEFINE_string("allow_classes", "person,car,truck,bus,motorbike", "allowed classes")
# NVR video source index
flags.DEFINE_integer("video_idx", "2", "the NVR video source index")
flags.DEFINE_boolean("cut_img", False, "cut object image when detected")
# the font scale to show object counter result on frame
FONT_SCALE = 2
# the time interval(seconds) to write counter value into DB
WRITE_DB_INTERVAL = 5 * 60


def getCross(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1])


def isPointInMatrix(p1, p2, p3, p4, p):
    '''
    ?????????????????????????????????P????????????????????????
    P1~P4??????????????????????????????"?????????"??????
    '''
    return getCross(p1, p2, p) * getCross(p3, p4, p) >= 0 and getCross(p2, p3, p) * getCross(p4, p1, p) >= 0


def write_into_db(counter, camId, allowed_classes):
    '''
    ?????????????????????POST??????DB
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
            print("send request Err:" + json.loads(result.text))
    except Exception as err:
        print("write into DB Err:" + str(err))

    print("write camId:{} counter into DB successfully! ".format(camId) + dt.now().strftime("%Y-%m-%d %H:%M:%S"))


def calculate_object_move_speed(x1, y1, x2, y2, frameCount):
    distance = pow((x2 - x1), 2) + pow((y2 - y1), 2)
    distance = pow(distance, 0.5)
    # 1 pixcel = 0.02m, FPS:20, ?????????????????????*3???????????????????????????
    speed = (distance * 0.02) / (frameCount / 20) * 3
    # print(x1, y1, x2, y2, distance, frameCount, speed)
    # convert speed from m/s to km/hr
    speed = int(speed / 1000 * 3600)
    # print("object Speed:{}".format(speed))
    # if speed over 120, it should be wrong
    return speed if speed < 120 else 0


def cut_detect_object(frame, bbox, object_type, index):
    try:
        now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        left, top, right, bottom = bbox
        cut_img = cv2.cvtColor(frame[int(top):int(bottom), int(left):int(right)], cv2.COLOR_RGB2BGR)
        filePath = os.path.join(os.getcwd(), "ai_model", "detection_images/{}/{}-{}.png".format(object_type, now, index))
        # print("top:{} left:{} right:{} bottom:{} filePath:{}".format(top, left, right, bottom, filePath))
        cv2.imwrite(filePath, cut_img)
    except Exception as err:
        print("cut_detect_object Err:" + str(err))


def main(_argv):
    # get NVR config
    try:
        url = "http://localhost:8000/nvr_config"
        result = requests.get(url)
        if result.status_code == requests.codes.ok:
            config = json.loads(result.text)
        # get NVR video source
        url = "http://localhost:8000/check_nvr"
        result = requests.get(url)
        if result.status_code == requests.codes.ok:
            result = json.loads(result.text)
            if result['resp'][FLAGS.video_idx]['state'] == 'signal_restored' or result['resp'][FLAGS.video_idx]['state'] == 'connected':
                camId = result['resp'][FLAGS.video_idx]['origin']
                rtspUrl = "rtsp://{}:{}@{}:554/hosts/{}".format(config['account'], config['password'], config['host'], camId)
    except:
        # exit("Can't not get NVR config")
        # just for test
        camId = "DESKTOP-F093S18/DeviceIpint.101/SourceEndpoint.video:0:0"
        rtspUrl = "rtsp://user1:user10824@60.249.33.163:554/hosts/" + camId
        print("Can't not get NVR config, use default rtsp url:{}".format(rtspUrl))
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = os.path.join(os.getcwd(), "ai_model", "model_data/mars-small128.pb")
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    print(rtspUrl)
    # begin video capture
    vid = cv2.VideoCapture(rtspUrl)

    # get width & height from video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("The width:{} height:{}".format(width, height))
    # ???????????????????????????threshold
    threshold = 0
    if FLAGS.direction == "horizontal":
        # ?????????????????????,?????????????????????Y????????????,?????????????????????/2???threshold
        y_diff_1 = abs(FLAGS.leftUp[1] - FLAGS.leftDown[1])
        y_diff_2 = abs(FLAGS.rightUp[1] - FLAGS.rightDown[1])
        if y_diff_1 < y_diff_2:
            threshold = y_diff_1 // 2
        else:
            threshold = y_diff_2 // 2
        print(y_diff_1, y_diff_2, threshold)
    else:
        # ?????????????????????,?????????????????????X????????????,?????????????????????/2???threshold
        x_diff_1 = abs(FLAGS.leftUp[0] - FLAGS.rightUp[0])
        x_diff_2 = abs(FLAGS.leftDown[0] - FLAGS.rightDown[0])
        if x_diff_1 < x_diff_2:
            threshold = x_diff_1 // 2
        else:
            threshold = x_diff_2 // 2
        print(x_diff_1, x_diff_2, threshold)

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    if FLAGS.allow_classes:
        allowed_classes = FLAGS.allow_classes.split(",")
    else:
        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

    frame_num = 0
    detect_objs = []
    lastWriteTime = dt.now()
    # when cut_img is true, disable info
    if FLAGS.cut_img:
        print("cut_img flag is enabled! disable info flag!")
        FLAGS.info = False
    # while video is running
    while vid.isOpened():
        start_time = time.time()
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print("Video has ended or failed, try a different video format!")
            vid.release()
            vid = cv2.VideoCapture(rtspUrl)
            continue
        frame_num += 1
        if FLAGS.info:
            print("Frame #: ", frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # run detections
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
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
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score,
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
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # the detection area line
        points = np.array([FLAGS.leftUp, FLAGS.leftDown, FLAGS.rightDown, FLAGS.rightUp], np.int32)
        cv2.polylines(frame, pts=[points], isClosed=True, color=(255, 0, 0), thickness=3)
        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        counter = len(names)
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            if FLAGS.info:
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1] - 30)),
                    (
                        int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                        int(bbox[1]),
                    ),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    class_name + "-" + str(track.track_id),
                    (int(bbox[0]), int(bbox[1] - 10)),
                    0,
                    0.75,
                    (255, 255, 255),
                    2,
                )
            # calcuate position of bbox and draw circle on
            x_cen = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
            y_cen = int(bbox[1] + (bbox[3] - bbox[1]) / 2)
            if FLAGS.info:
                cv2.circle(frame, (x_cen, y_cen), 5, (255, 0, 0), -1)
            # check be tracked object on detection area
            checkDirection = False
            tracked_pos = 0
            if isPointInMatrix(FLAGS.leftUp, FLAGS.rightUp, FLAGS.rightDown, FLAGS.leftDown, [x_cen, y_cen]):
                checkDirection = True
                if FLAGS.direction == "horizontal":
                    tracked_pos = y_cen
                else:
                    tracked_pos = x_cen
            if checkDirection:
                # print("Tracker In Area ID: {}, Class: {},  BBox Coords (x_cen, y_cen): {} W:{} H:{}".format(
                #     str(track.track_id), class_name, (x_cen, y_cen), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])))
                existed = False
                for obj in detect_objs:
                    if obj['id'] == track.track_id:
                        existed = True
                        obj['frameCount'] += 1
                        if FLAGS.direction == "horizontal":
                            orig_pos = obj['y_orig']
                        else:
                            orig_pos = obj['x_orig']
                        diff = tracked_pos - orig_pos
                        # check object direction if it is none
                        if obj['direction'] == "none":
                            if diff >= threshold:
                                obj['direction'] = "down"
                                orig_pos = tracked_pos
                            elif diff <= -threshold:
                                obj['direction'] = "up"
                                orig_pos = tracked_pos
                            # the direction has been detected, calculate speed
                            if obj['direction'] != "none":
                                obj['speed'] = calculate_object_move_speed(obj['x_orig'], obj['y_orig'], x_cen, y_cen, obj['frameCount'])
                                cut_detect_object(frame, bbox, class_name, track.track_id)
                # to append object into array if object doesn't existd
                if not existed:
                    obj = {
                        "class": class_name,
                        "id": track.track_id,
                        "y_orig": y_cen,
                        "x_orig": x_cen,
                        "direction": "none",
                        'frameCount': 0,
                        'speed': 0
                    }
                    detect_objs.append(obj)
            # if enable info flag then print details about each track
            # if FLAGS.info:
            #     print(
            #         "Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}"
            #         .format(str(track.track_id), class_name, (int(
            #             bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        # define counter for every objects
        counter = {}
        for name in allowed_classes:
            key_up = name + "-up"
            key_down = name + "-down"
            key_up_speed = name + "-up-speed"
            key_down_speed = name + "-down-speed"
            counter[key_up] = 0
            counter[key_down] = 0
            counter[key_up_speed] = 0
            counter[key_down_speed] = 0
        # record objects direction
        for obj in detect_objs:
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
            if FLAGS.direction == "vertical":
                if "up" in key:
                    labelName = key.replace("up", "IN")
                elif "down" in key:
                    labelName = key.replace("down", "OUT")
            cv2.putText(frame, "{}:{}".format(labelName, counter[key]), (width // 3, 35 + idx * 25 * FONT_SCALE), 0, FONT_SCALE, (255, 0, 0), 1)
            idx += 1
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        if FLAGS.info:
            print("FPS: %.2f" % fps)
        # show image on screen
        if not FLAGS.dont_show:
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # resize the ouput frame to 1280x720
            result = cv2.resize(result, (1280, 720))
            cv2.imshow("Output Video", result)
        # wirte data into DB every time inteval
        diffTime = dt.now() - lastWriteTime
        if diffTime.seconds >= WRITE_DB_INTERVAL:
            # update last time stamp
            lastWriteTime = dt.now()
            # write_into_db(counter, camId, allowed_classes)
            # reset counter
            detect_objs = []

        # check exit when press keyboard 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # keep wait unitl press 'n', just for debug
        while FLAGS.frame_debug:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                break
            elif key == ord("q"):
                return
    # destroy resource
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
