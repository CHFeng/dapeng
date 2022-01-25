import cv2
import threading
from datetime import datetime as dt
# local import
# from tools import generate_detections as gdet
# # deep sort imports
# from deep_sort.tracker import Tracker
# from deep_sort.detection import Detection
# from deep_sort import preprocessing, nn_matching


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running and cam.vid.isOpened():
        _, cam.img_handle = cam.vid.read()
        if cam.img_handle is None:
            print("rtsp streaming has ended or failed, try again!")
            cam.vid.release()
            cam.vid = cv2.VideoCapture(cam.rtspUrl)
            continue
    cam.thread_running = False


class Detect:
    def __init__(self, rtspUrl, infer, args) -> None:
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        # # initialize deep sort
        # model_filename = "model_data/mars-small128.pb"
        # self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # # calculate cosine distance metric
        # metric = nn_matching.NearestNeighborDistanceMetric(
        #     "cosine", max_cosine_distance, nn_budget)
        # # initialize tracker
        # self.tracker = Tracker(metric)
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
        self.frame_num = 0
        self.detect_objs = []
        self.lastWriteTime = dt.now()
        self.thread_running = False
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        if self.vid is not None:
            raise RuntimeError('camera is already opened!')
        print('RTSP url is: {}'.format(self.rtspUrl))
        self.vid = cv2.VideoCapture(self.rtspUrl)
        self._start()

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
        self.thread = threading.Thread(target=grab_img, args=(self, ))
        self.thread.start()

    def _stop(self):
        if self.thread_running:
            self.thread_running = False
            #self.thread.join()

    def read(self):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if not self.is_opened:
            return None

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
