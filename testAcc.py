"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint
import cv2
from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import drawFrameObject, drawTargetFinal
from finalProject.utils.keyPoints.AlgoritamKeyPoints import createDescriptorTarget, SurfDetectKeyPoints, \
    KazeDetectKeyPoints
from finalProject.utils.matchers.Matchers import compare_between_two_description, kaze_matcher, flannmatcher
from finalProject.utils.preprocessing.preprocess import readFromInputVideoFrames, framesExists, reduceNoise
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
        frameSource = reduceNoise(frameSource)
        if not framesExists(frameSource):
            print("problem with reduce noise source video input")
            exit(0)

        # for frame in frameSource:
        #     cv2.imshow('extracted frame', frame)
        #     keyboard = cv2.waitKey(30)
        #     if keyboard == 'q' or keyboard == 27:
        #         break

        mySource = source_detection_by_yolo(frameSource, yolo,
                                            isVideo=config["source"]["isVideo"],
                                            config=config["source"])
        if mySource is None:
            print("fail to detect human on source video")
            exit(0)

        # target
        frameTarget = readFromInputVideoFrames(config["target"])
        if not framesExists(frameTarget):
            print("problem with target video input")
            exit(0)

        # pre processing reduce noise background
        frameTarget = reduceNoise(frameTarget)

        if not framesExists(frameTarget):
            print("problem with target video input -reduce noise")
            exit(0)

        myTargets = tracking_by_yolo(frameTarget, yolo, isVideo=config["target"]["isVideo"], config=config["target"])

        if not framesExists(myTargets):
            print("fail to detect humans on target video")
            exit(0)
        # target descriptor

        ks, ds = KazeDetectKeyPoints(mySource.frames[0])
        kt, dt = KazeDetectKeyPoints(myTargets[0].frames[2])
        matches = kaze_matcher(ds, dt)
        print(len(ds))
        print(len(dt))
        print(len(ks))
        print(len(kt))
        print(len(matches))

        acc = len(matches) / min(len(ds), len(dt))
        print(acc)

        ks, ds = SurfDetectKeyPoints(mySource.frames[0])
        kt, dt = SurfDetectKeyPoints(myTargets[0].frames[2])
        matches = flannmatcher(ds, dt)
        acc = len(matches) / min(len(ds), len(dt))
        print(len(ds))
        print(len(dt))
        print(len(ks))
        print(len(kt))
        print(len(matches))
        print(acc)
