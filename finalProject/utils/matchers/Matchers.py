""" **Matchers**
#Kaze Matcher for binary classification - orb , kaze
"""
import cv2
import numpy as np

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SurfDetectKeyPoints


def kaze_matcher(desc1, desc2, threshold=0.8):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []  # Apply ratio test
    for m, n in nn_matches:
        if m.distance < threshold * n.distance:
            good.append([m])
    return good


def find_closest_human(target, people_list, config: "config file"):
    """returns a list of (person, accuracy) pairs of all people in target video
    and their accuracy matching to the person in target argument"""
    key_target, description_target = SurfDetectKeyPoints(target["frame"])
    if key_target is None or description_target is None:
        return None  # don't have key points for this human
    max_match = []
    for p in people_list:
        if len(p.frames) > config["max_length_frames"]:  # remove trace frames
            p.history.extend(p.frames[0:len(p.frames) - config["max_length_frames"]])
            p.frames = p.frames[-config["max_length_frames"]:]

        match_p = []
        for index, frame in enumerate(p.frames):
            kp, dp = SurfDetectKeyPoints(frame.frame_image)
            if kp is None or dp is None:
                continue
            else:
                good_match = flann_matcher(description_target, dp, config["FlannMatcherThreshold"])
            if len(key_target) == 0:
                acc = 0
            else:
                acc = len(good_match) / (len(dp))
            match_p.append(min(acc, 1))
        if len(match_p) > 0:
            mean_acc = np.amax(match_p)
        else:
            mean_acc = 0
        max_match.append((p, mean_acc))

    return max_match


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


def flann_matcher(des1, des2, threshold=0.8):  # threshold is the distance between the points we're comparing
    """#FLANN MATCHER for SURF and SIFT"""
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


def compare_between_descriptors(sourceFrame, targetFrame):
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
            acc = len(matches) / (len(des_s))
            results.append(min(acc, 1))

    for algo in float_algo:
        des_s = sourceFrame[algo]["des"]
        des_t = targetFrame[algo]["des"]
        if len(des_s) == 0 or len(des_t) == 0:
            results.append(0)
        else:
            matches = flann_matcher(des_s, des_t)
            acc = len(matches) / (len(des_s))
            results.append(min(acc, 1))

    return np.mean(results)


def compute_accuracy_table(source_descriptors, target_descriptors):
    acc_table = {}
    for t_id, t_des_list in target_descriptors.items():  # loop through both keys and values of dict
        frames_table = np.zeros(shape=[len(t_des_list), len(source_descriptors[0])])  # rows=num of people in target, cols=num of src descriptors
        for t_des_index, t_des in enumerate(t_des_list):
            for s_des_index, s_des in enumerate(source_descriptors[0]):
                frames_table[t_des_index, s_des_index] = compare_between_descriptors(s_des, t_des)

        max_acc = np.amax(frames_table)
        ind = np.unravel_index(np.argmax(frames_table, axis=None), frames_table.shape)
        acc_table[t_id] = {"maxAcc": max_acc,
                           "target_des": t_des_list,
                           "target_frames": t_des_list[ind[0]],
                           "source_frames": source_descriptors[0][ind[1]]}
    return acc_table
