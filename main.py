"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint

from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import drawTargetFinal
from finalProject.utils.keyPoints.AlgoritamKeyPoints import create_key_points_descriptors
from finalProject.utils.matchers.Matchers import compare_between_two_description
from finalProject.utils.preprocessing.preprocess import readFromInputVideoFrames, framesExists, reduceNoise, \
    removeRemovalColor
from finalProject.utils.tracking.TrackingByYolo import source_detection_by_yolo, tracking_by_yolo

if __name__ == "__main__":
    """# import images"""

    # init yolo
    yolo = Yolo()
    yolo.initYolo()
    pp = pprint.PrettyPrinter(indent=4)

    with open('./config.txt') as file_json:
        config = json.load(file_json)

        sourceFrames = readFromInputVideoFrames(config["source"])  # a list of all frames extracted from source video
        if not framesExists(sourceFrames):  # if not len(sourceFrames) > 0
            print("problem with source video input")
            exit(0)

        # pre processing reduce noise background
        if config["source"]["reduceNoise"]:
            sourceFrames = reduceNoise(sourceFrames)
        if not framesExists(sourceFrames):
            print("problem with reduce noise source video input")
            exit(0)

        if config["source"]["removeRemovalColor"]:
            sourceFrames = removeRemovalColor(sourceFrames)

        # for frame in sourceFrames:
        #     cv2.imshow('removeRemovalColor frame', frame)
        #     keyboard = cv2.waitKey(30)
        #     if keyboard == 'q' or keyboard == 27:
        #         break

        """mySource is an object of type Person"""
        mySource = source_detection_by_yolo(sourceFrames, yolo,
                                            isVideo=config["source"]["isVideo"],
                                            config=config["source"])
        if mySource is None:
            print("fail to detect human on source video")
            exit(0)

        sourceDescriptors = create_key_points_descriptors([mySource])  # gets source descriptors

        # target
        targetFrames = readFromInputVideoFrames(config["target"])
        if not framesExists(targetFrames):
            print("problem with target video input")
            exit(0)

        if config["target"]["reduceNoise"]:
            targetFrames = reduceNoise(targetFrames)
        if not framesExists(targetFrames):
            print("problem with target video input -reduce noise")
            exit(0)

        if config["target"]["removeRemovalColor"]:
            targetFrames = removeRemovalColor(targetFrames)

        people_in_target = tracking_by_yolo(targetFrames, yolo, isVideo=config["target"]["isVideo"],
                                            config=config["target"])

        if not framesExists(people_in_target):
            print("fail to detect humans on target video")
            exit(0)
        # target descriptor

        target_descriptors = create_key_points_descriptors(people_in_target)

        frameExampleTarget = target_descriptors[0][0]
        # frameExampleSource = sourceDescriptors[0][0]

        # drawFrameObject(frameExampleSource)
        #drawFrameObject(frameExampleTarget)
        print("frameExampleTarget:")
        print(frameExampleTarget)

        acc_targets = compare_between_two_description(sourceDescriptors, target_descriptors)
        """
        acc_target look like :
         {
           id_0 : {
           maxAcc : double,
           target : [arrayOfFrameObject]
           targetFrames : FrameObject
           sourceFrames : FrameObject
           }
         }
        """

        drawTargetFinal(acc_targets, options=config["output"])
