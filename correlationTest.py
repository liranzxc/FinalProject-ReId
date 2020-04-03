"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint
import cv2
from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import drawTargetFinal
from finalProject.utils.images.imagesUtils import resizeImage
from finalProject.utils.keyPoints.AlgoritamKeyPoints import createDescriptorTarget
from finalProject.utils.matchers.Matchers import compare_between_two_description
from finalProject.utils.preprocessing.preprocess import readFromInputVideoFrames, framesExists, reduceNoise, \
    removeRemovalColor
from finalProject.utils.tracking.TrackingByYolo import source_detection_by_yolo, tracking_by_yolo
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    """# import images"""

    # init yolo
    yolo = Yolo()
    yolo.initYolo()
    pp = pprint.PrettyPrinter(indent=4)

    with open('./config.txt') as file_json:
        config = json.load(file_json)

        # source
        frameSource = readFromInputVideoFrames(config["source"])
        if not framesExists(frameSource):
            print("problem with source video input")
            exit(0)

        # pre processing reduce noise background
        if config["source"]["reduceNoise"]:
            frameSource = reduceNoise(frameSource)
        if not framesExists(frameSource):
            print("problem with reduce noise source video input")
            exit(0)

        if config["source"]["removeRemovalColor"]:
            frameSource = removeRemovalColor(frameSource)

        # for frame in frameSource:
        #     cv2.imshow('removeRemovalColor frame', frame)
        #     keyboard = cv2.waitKey(30)
        #     if keyboard == 'q' or keyboard == 27:
        #         break

        mySource = source_detection_by_yolo(frameSource, yolo,
                                            isVideo=config["source"]["isVideo"],
                                            config=config["source"])
        if mySource is None:
            print("fail to detect human on source video")
            exit(0)

        # source descriptor
        descriptorSource = createDescriptorTarget([mySource])

        # target
        frameTarget = readFromInputVideoFrames(config["target"])
        if not framesExists(frameTarget):
            print("problem with target video input")
            exit(0)

        if config["target"]["reduceNoise"]:
            frameTarget = reduceNoise(frameTarget)

        if not framesExists(frameTarget):
            print("problem with target video input -reduce noise")
            exit(0)

        if config["target"]["removeRemovalColor"]:
            frameTarget = removeRemovalColor(frameTarget)

        myTargets = tracking_by_yolo(frameTarget, yolo, isVideo=config["target"]["isVideo"], config=config["target"])

        if not framesExists(myTargets):
            print("fail to detect humans on target video")
            exit(0)
        # target descriptor

        descriptorTarget = createDescriptorTarget(myTargets)

        # frameExampleTarget = descriptorTarget[0][0]
        # frameExampleSource = descriptorSource[0][0]

        # drawFrameObject(frameExampleSource)
        # drawFrameObject(frameExampleTarget)

        acc_targets = compare_between_two_description(descriptorSource, descriptorTarget)
        """
        acc_target look like :
         {
           id_0 : {
           maxAcc : double,
           target : [arrayOfFrameObject]
           frameTarget : FrameObject
           frameSource : FrameObject
           }
         }
        """
        target = "target", acc_targets[0]["frameTarget"]["frame"]
        source = "source", acc_targets[0]["frameSource"]["frame"]

        target = target[1]
        source = source[1]

        cv2.destroyAllWindows()

        cv2.cvtColor(target, cv2.COLOR_RGB2GRAY, target)
        cv2.cvtColor(source, cv2.COLOR_RGB2GRAY, source)

        plt.subplot(121)
        plt.imshow(source)
        # plt.subplot(122)
        # plt.imshow(target)
        plt.show()

        # w, h, d = target.shape
        #
        #
        # # cv2.imshow("target with template match", target)
        # # cv2.imshow("source with template match", source)
        # # cv2.waitKey(0)
        # # Convert it to HSV
        # img1_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        # img2_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        #
        # # Calculate the histogram and normalize it
        # hist_img1 = cv2.calcHist([img1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # hist_img2 = cv2.calcHist([img2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #
        # # find the metric value
        # metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
        # print(metric_val)

        # cv2.imshow("target with template match", target)
        # cv2.imshow("source with template match", source)
        # cv2.waitKey(0)
        #
        # # Apply template Matching
        # res = cv2.matchTemplate(source, target, cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # top_left = max_loc

        # cv2.imshow(" source match", source)
        #
        # target = resizeImage(target, fy=2, fx=2)
        # cv2.imshow("target with template match", target)
        # cv2.waitKey(0)