"""camera.py

This code implements the Camera class, which encapsulates code to
handle IP CAM, USB webcam or the Jetson onboard camera.  In
addition, this Camera class is further extended to take a video
file or an image file as input.
"""
import cv2
import logging
import threading
import subprocess
from typing import DefaultDict
import numpy as np
# for capture streaming from youtube
import pafy

# The following flag ise used to control whether to use a GStreamer
# pipeline to open USB webcam source.  If set to False, we just open
# the webcam using cv2.VideoCapture(index) machinery. i.e. relying
# on cv2's built-in function to capture images from the webcam.
USB_GSTREAMER = True


def open_cam_rtsp(uri, width, height, latency):
    """Open an RTSP URI (IP CAM)."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
    elif 'avdec_h264' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        gst_str = ('rtspsrc location={} latency={} ! ' 'rtph264depay ! h264parse ! avdec_h264 ! ' 'videoconvert ! appsink').format(uri, latency)
    else:
        raise RuntimeError('H.264 decoder not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    """Open a USB webcam."""
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)


def open_cam_gstr(gstr, width, height):
    """Open camera using a GStreamer string.

    Example:
    gstr = 'v4l2src device=/dev/video0 ! video/x-raw, width=(int){width}, height=(int){height} ! videoconvert ! appsink'
    """
    gst_str = gstr.format(width=width, height=height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    """Open the Jetson onboard camera."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_youtube_uri(uri):
    '''
    gst-launch-1.0 souphttpsrc is-live=true location="$(youtube-dl --format "best[ext=mp4]" --get-url https://www.youtube.com/watch?v=sZXBFjepdeQ)" ! decodebin ! videoconvert ! autovideosink
    gst-launch-1.0 playbin uri="$(youtube-dl --format "best[ext=mp4]" --get-url https://www.youtube.com/watch?v=sZXBFjepdeQ)"
    '''
    # link = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1637765862/ei/hv6dYZfKK5Cv2roPk_6JoAI/ip/125.227.241.108/id/sZXBFjepdeQ.3/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/hls_chunk_host/rr1---sn-ipoxu-3iid.googlevideo.com/playlist_duration/30/manifest_duration/30/vprv/1/playlist_type/DVR/initcwndbps/6680/mh/qq/mm/44/mn/sn-ipoxu-3iid/ms/lva/mv/m/mvi/1/pl/24/dover/11/keepalive/yes/fexp/24001373,24007246/mt/1637744190/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,playlist_duration,manifest_duration,vprv,playlist_type/sig/AOq0QJ8wRQIhAJGNbthvsJm4rVQ6ebBT2FFE7hb3uVhS8q0SMrecV7UJAiAnZoFXsPaV1QJ0V1QO-dJ76tEjwwTGa82HxmnmC_FubA%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AG3C_xAwRQIhAJ9zmblMbBQgJAGZ_l3Ht3qz9H2MareiRqtHswWd93N7AiBp5eWAEpscWpvo-SgcuYAo80Zt6-eAO3KZFDBL3SywBg%3D%3D/playlist/index.m3u8"
    # gst_str = ('playbin uri="{}" ').format(link)
    # print(gst_str)
    print(uri)
    video = pafy.new(uri)
    best = video.getbest(preftype="mp4")
    return cv2.VideoCapture(best.url)

    # return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            #logging.warning('Camera: cap.read() returns None...')
            break
    cam.thread_running = False


class Camera():
    """Camera class which supports reading images from theses video sources:

    1. Image (jpg, png, etc.) file, repeating indefinitely
    2. Video file
    3. RTSP (IP CAM)
    4. USB webcam
    5. Jetson onboard camera
    6. youtube
    """
    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.video_file = ''
        self.video_looping = False
        self.thread_running = False
        self.img_handle = None
        self.copy_frame = False
        self.do_resize = False
        self.img_width = args.width
        self.img_height = args.height
        self.cap = None
        self.thread = None
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        if self.cap is not None:
            raise RuntimeError('camera is already opened!')
        a = self.args
        if a.video:
            logging.info('Camera: using a video file %s' % a.video)
            self.video_file = a.video
            self.cap = cv2.VideoCapture(a.video)
            self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._start()
        elif a.rtsp:
            logging.info('Camera: using RTSP stream %s' % a.rtsp)
            self.cap = open_cam_rtsp(a.rtsp, a.width, a.height, a.rtsp_latency)
            self._start()
        elif a.usb is not None:
            logging.info('Camera: using USB webcam /dev/video%d' % a.usb)
            self.cap = open_cam_usb(a.usb, a.width, a.height)
            self._start()
        elif a.gstr is not None:
            logging.info('Camera: using GStreamer string "%s"' % a.gstr)
            self.cap = open_cam_gstr(a.gstr, a.width, a.height)
            self._start()
        elif a.youtube is not None:
            self.cap = open_youtube_uri(a.youtube)
            self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._start()
        else:
            raise RuntimeError('no camera type specified!')

    def isOpened(self):
        return self.is_opened

    def _start(self):
        if not self.cap.isOpened():
            logging.warning('Camera: starting while cap is not opened!')
            return

        # Try to grab the 1st image and determine width and height
        _, self.img_handle = self.cap.read()
        if self.img_handle is None:
            logging.warning('Camera: cap.read() returns no image!')
            self.is_opened = False
            return

        self.is_opened = True
        if self.video_file:
            if not self.do_resize:
                self.img_height, self.img_width, _ = self.img_handle.shape
        else:
            self.img_height, self.img_width, _ = self.img_handle.shape
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
        if self.video_file:
            _, img = self.cap.read()
            if img is None:
                logging.info('Camera: reaching end of video file')
                if self.video_looping:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_file)
                _, img = self.cap.read()
            if img is not None and self.do_resize:
                img = cv2.resize(img, (self.img_width, self.img_height))
            return img
        elif self.cap == 'image':
            return np.copy(self.img_handle)
        else:
            if self.copy_frame:
                return self.img_handle.copy()
            else:
                return self.img_handle

    def release(self):
        self._stop()
        try:
            self.cap.release()
        except:
            pass
        self.is_opened = False

    def __del__(self):
        self.release()
