"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint
import cv2
from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import drawTargetFinal
from finalProject.utils.keyPoints.AlgoritamKeyPoints import createDescriptorTarget
from finalProject.utils.matchers.Matchers import compare_between_two_description
from finalProject.utils.preprocessing.preprocess import readFromInputVideoFrames, framesExists, reduceNoise, \
    removeRemovalColor
from finalProject.utils.tracking.TrackingByYolo import source_detection_by_yolo, tracking_by_yolo
import matplotlib.pyplot as plt

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

        frameExampleTarget = descriptorTarget[0][0]
        # frameExampleSource = descriptorSource[0][0]

        # drawFrameObject(frameExampleSource)
        #drawFrameObject(frameExampleTarget)
        print("frameExampleTarget:")
        print(frameExampleTarget)

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

        drawTargetFinal(acc_targets, options=config["output"])
