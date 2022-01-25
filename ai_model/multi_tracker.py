import cv2
from detect import Detect

rtspUrl = "rtsp://user1:user10824@60.249.33.163:554/hosts/DESKTOP-F093S18/DeviceIpint.103/SourceEndpoint.video:0:0"
rtspUrl2 = "rtsp://user1:user10824@60.249.33.163:554/hosts/DESKTOP-F093S18/DeviceIpint.101/SourceEndpoint.video:0:0"
cam = Detect(rtspUrl, None, None)
cam2 = Detect(rtspUrl2, None, None)

camIdx = 0
while True:
    if camIdx == 0:
        img = cam.read()
    else:
        img = cam2.read()

    img = cv2.resize(img, (1280, 720))
    cv2.imshow('result', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("0"):
        camIdx = 0
    elif key == ord("1"):
        camIdx = 1
cam.release()
cam2.release()
while True:
    if cam.isOpened() == False and cam2.isOpened() == False:
        break
    else:
        print("wait thread release")
cv2.destroyAllWindows()