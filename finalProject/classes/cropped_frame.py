import cv2
# TODO consider adding the locations too


class CroppedFrame:
    def __init__(self, frame_image):
        self.frame_image = frame_image
        self.gray_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        self.frame_keypoints = {}
        self.frame_des = {}
