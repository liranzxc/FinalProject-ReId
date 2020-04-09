# TODO consider adding the locations too


class CroppedFrame:
    def __init__(self, frame_image):
        self.frame_image = frame_image
        self.gray_image = None
        self.frame_keypoints = {}
        self.frame_des = {}
