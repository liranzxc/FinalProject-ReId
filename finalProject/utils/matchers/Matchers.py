""" **Matchers**
#Kaze Matcher for binary classification - orb , kaze ,brief,fast
"""
import cv2
import numpy as np

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SurfDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SiftDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import ORBDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import KazeDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SurfDetectKeyPoints


def kaze_matcher(desc1, desc2):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, k=2)
    return nn_matches


def find_closes_human(target, myPeople, config: "config file"):
    key_target, description_target = SurfDetectKeyPoints(target["frame"])
    if key_target is None or description_target is None:
        return None, None, None  # dont have key points for this human
    max_match = []
    for p in myPeople:
        # remove trace frames
        if len(p.frames) > config["max_length_frames"]:
            p.history.extend(p.frames[0:len(p.frames) - config["max_length_frames"]])
            p.frames = p.frames[-config["max_length_frames"]:]
            p.keys = p.keys[-config["max_length_frames"]:]
            p.des = p.des[-config["max_length_frames"]:]

        match_p = []
        for index, frame in enumrate(p.frames):
            kp, dp = p.keys[index], p.des[index]
            if kp is None or dp is None:
                continue
            else:
                good_match = flannmatcher(description_target, dp, config["FlannMatcherThreshold"])
            if len(key_target) == 0:
                acc = 0
            else:
                acc = len(good_match) / len(key_target)
            match_p.append(acc)
        if len(match_p) > 0:
            mean_acc = np.mean(match_p)
        else:
            mean_acc = 0
        max_match.append((p, mean_acc))

    return max_match, key_target, description_target


def bf_matcher(des1, des2, threshold):
    # BFMatcher with default params
    bf = cv2.bf_matcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])
    return good


"""#FLANN MATCHER for SURF and SIFT"""


def flannmatcher(des1, des2, threshold=0.8):  # threshold is the distance between the points we're comparing
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if len(des1) >= 2 and len(des2) >= 2:
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        good = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < threshold * n.distance:
                good.append([m])
        return good
    else:
        return []


def compare_between_two_frames_object(sourceFrame, targetFrame):
    binary_algo = [NamesAlgorithms.ORB.name, NamesAlgorithms.KAZE.name]
    float_algo = [NamesAlgorithms.SURF.name, NamesAlgorithms.SIFT.name]
    results = []
    for algo in binary_algo:
        des_s = sourceFrame[algo]["des"]
        des_t = targetFrame[algo]["des"]
        if len(des_s) == 0 or len(des_t) == 0:
            results.append(0)
        else:
            matches = kaze_matcher(des_s, des_t)
            results.append(len(matches))

    for algo in float_algo:
        des_s = sourceFrame[algo]["des"]
        des_t = targetFrame[algo]["des"]
        if len(des_s) == 0 or len(des_t) == 0:
            results.append(0)
        else:
            matches = flannmatcher(des_s, des_t)
            results.append(len(matches))

    return np.mean(results)


def compare_between_two_description(sourceDescriptor, targetDescriptor):
    acc_target = {}
    for _id, target in targetDescriptor.items():
        table_acc = np.zeros(shape=[len(target), len(sourceDescriptor[0])])
        for index_t, frame_t in enumerate(target):
            for index_s, frame_s in enumerate(sourceDescriptor[0]):
                table_acc[index_t, index_s] = compare_between_two_frames_object(frame_s, frame_t)

        max_matches = np.amax(table_acc)
        ind = np.unravel_index(np.argmax(table_acc, axis=None), table_acc.shape)
        acc_target[_id] = {"maxAcc": max_matches,
                           "target": target,
                           "frameTarget": target[ind[0]],
                           "frameSource": sourceDescriptor[0][ind[1]]}
    return acc_target
