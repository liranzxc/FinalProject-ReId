"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint

from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import drawTargetFinal
from finalProject.utils.keyPoints.AlgoritamKeyPoints import create_key_points_descriptors
from finalProject.utils.matchers.Matchers import compare_between_two_description
from finalProject.utils.preprocessing.preprocess import readFromInputVideoFrames, framesExists, reduceNoise, removeRemovalColor
from finalProject.utils.tracking.TrackingByYolo import source_detection_by_yolo, tracking_by_yolo

if __name__ == "__main__":
    """# import images"""
    # init yolo
    yolo = Yolo()
    yolo.initYolo()
    pp = pprint.PrettyPrinter(indent=4)

    with open('./config.txt') as file_json:
        config = json.load(file_json)

        """ source video """
        source_frames = readFromInputVideoFrames(config["source"])  # a list of all frames extracted from source video
        if not framesExists(source_frames):  # if not len(source_frames) > 0
            print("problem with source video input")
            exit(0)

        # pre processing reduce noise background
        if config["source"]["reduceNoise"]:
            source_frames = reduceNoise(source_frames)
        if not framesExists(source_frames):
            print("problem with reduce noise source video input")
            exit(0)

        if config["source"]["removeRemovalColor"]:
            source_frames = removeRemovalColor(source_frames)

        # for frame in source_frames:
        #     cv2.imshow('removeRemovalColor frame', frame)
        #     keyboard = cv2.waitKey(30)
        #     if keyboard == 'q' or keyboard == 27:
        #         break

        source_person = source_detection_by_yolo(source_frames, yolo, is_video=config["source"]["isVideo"], config=config["source"])
        if source_person is None:
            print("fail to detect human on source video")
            exit(0)

        source_descriptors = create_key_points_descriptors([source_person])  # gets source descriptors

        """ target video """
        target_frames = readFromInputVideoFrames(config["target"])
        if not framesExists(target_frames):
            print("problem with target video input")
            exit(0)

        if config["target"]["reduceNoise"]:
            target_frames = reduceNoise(target_frames)
        if not framesExists(target_frames):
            print("problem with target video input -reduce noise")
            exit(0)

        if config["target"]["removeRemovalColor"]:
            target_frames = removeRemovalColor(target_frames)

        target_people = tracking_by_yolo(target_frames, yolo, isVideo=config["target"]["isVideo"], config=config["target"])
        if not framesExists(target_people):
            print("fail to detect humans on target video")
            exit(0)

        target_descriptors = create_key_points_descriptors(target_people)

        frameExampleTarget = target_descriptors[0][0]
        # frameExampleSource = source_descriptors[0][0]

        # drawFrameObject(frameExampleSource)
        # drawFrameObject(frameExampleTarget)

        print("frameExampleTarget:")
        print(frameExampleTarget)

        acc_targets = compare_between_two_description(source_descriptors, target_descriptors)
        """
        acc_target look like :
         {
           id_0 : {
           maxAcc : double,
           target : [arrayOfFrameObject]
           target_frames : FrameObject
           source_frames : FrameObject
           }
         }
        """

        drawTargetFinal(acc_targets, options=config["output"])
