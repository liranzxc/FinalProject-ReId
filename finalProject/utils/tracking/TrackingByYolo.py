import cv2

from finalProject.classes.human import Human
from finalProject.utils.drawing.draw import DrawHumans, ShowPeopleTable, DrawSource
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SurfDetectKeyPoints
from finalProject.utils.matchers.Matchers import find_closes_human

import copy


def tracking_by_yolo(sequences: [], yolo, isVideo: bool, config: "file"):
    """this function creates an array containing objects of all people that are in the target video,
    it assigns each person his frame-boxes from the video and other attributes,
    (this is regardless of the person in source video)."""
    my_people = []
    counter_id = 0
    frame_rate = config["frameRate"]
    if config["videoFrameLength"] == -1:
        num_of_frames = len(sequences)
    else:
        num_of_frames = config["videoFrameLength"]

    if config["videoFrameLength"] > len(sequences):
        print("videoFrameLength larger then video")
        num_of_frames = len(sequences)

    if num_of_frames > 1:
        # start capture
        for index in range(0, num_of_frames, frame_rate):
            affected_people = []  # a list of counterId(int) of all people that are in the current frame
            # print("frame {}".format(index))

            if isVideo:
                frame2 = sequences[index]
            else:
                frame2 = cv2.imread(sequences[index])

            drawFrame = copy.copy(frame2)  # a copy list of the frame2 list
            if index == 0:  # first frame
                cropped_image = yolo.forward(frame2)  # returns all frame-boxes of people in the current frame
                cropped_image = list(filter(lambda crop: crop["frame"].size, cropped_image))  # filters frame-boxes
                for c in cropped_image:  # loop on frame-boxes
                    append_human_to_people(my_people, affected_people, counter_id, c)
                    counter_id += 1

            elif index > 0:
                cropped_image = yolo.forward(frame2)
                cropped_image = list(filter(lambda crop: crop["frame"].size, cropped_image))
                for c in cropped_image:  # determine which frame-box is related to which person
                    if len(my_people) > 0:
                        max_match = find_closes_human(c, my_people, config=config)  # returns a list of all people and their accuracy
                        if max_match is None:
                            continue  # skip iteration and continue on with the next iteration

                        max_maximum = max(max_match, key=lambda item: item[1])  # gets the person with max accuracy
                        if max_maximum[1] > config["thresholdAppendToHuman"]:  # accuracy is good enough to determine it's the same person
                            indexer = my_people.index(max_maximum[0])  # returns the index of this person with max accuracy in myPeople list
                            affected_people.append(indexer)  # adds this person to affected_people list
                            my_people[indexer].frames.append(c["frame"])  # adds the box-frame to this person
                            my_people[indexer].locations.append(c["location"])  # adds his locations too

                        elif config["thresholdAppendNewHumanStart"] < max_maximum[1] \
                                < config["thresholdAppendNewHumanEnd"]:  # if accuracy matches the values determine to create a new person
                            append_human_to_people(my_people, affected_people, counter_id, c)
                            counter_id += 1
                    else:  # creates a new person (this is the first person)
                        append_human_to_people(my_people, affected_people, counter_id, c)
                        counter_id += 1

            DrawHumans(my_people, drawFrame, affected_people)
            # find ids from previous frame
            if config["show"]:
                cv2.imshow('frame', drawFrame)
                k = cv2.waitKey(config["WaitKeySecond"]) & 0xff
                if k == 27:
                    break

    return my_people


def append_human_to_people(myPeople, affectedPeople, counterId, c):
    human = Human(counterId)
    affectedPeople.append(counterId)
    human.frames.append(c["frame"])
    human.locations.append(c["location"])
    myPeople.append(human)


# sequences is all frames related to the source
def source_detection_by_yolo(sequences: [], yolo, isVideo: bool, config: "file"):
    human = None
    frameRate = config["frameRate"]
    if config["videoFrameLength"] == -1:
        numOfFrames = len(sequences)
    else:
        numOfFrames = config["videoFrameLength"]

    if config["videoFrameLength"] > len(sequences):
        print("videoFrameLength larger then video")
        numOfFrames = len(sequences)

    if numOfFrames > 1:
        # start capture, looping on the frames, skipping the frameRate
        for index in range(0, numOfFrames, frameRate):
            # print("frame {}".format(index))
            if isVideo:
                frame2 = sequences[index]
            else:
                frame2 = cv2.imread(sequences[index])

            drawFrame = copy.copy(frame2)
            croppedImage = yolo.forward(frame2)
            croppedImage = list(filter(lambda crop: crop["frame"].size, croppedImage))

            if len(croppedImage) > 1:
                print("On source found two people , must be one person")
                return None

            # index is the frame number
            for c in croppedImage:
                if human is None:
                    human = Human(0)
                human.frames.append(c["frame"])
                human.locations.append(c["location"])

            if human is not None:
                DrawSource(human, drawFrame)
                # find ids from previous frame
            if config["show"]:
                cv2.imshow('frame', drawFrame)
                k = cv2.waitKey(config["WaitKeySecond"]) & 0xff
                if k == 27:
                    break
    else:
        print("The number of frames is less than one")

    return human
