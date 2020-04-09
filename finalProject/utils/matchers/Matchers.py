""" **Matchers**
#Kaze Matcher for binary classification - orb , kaze
"""
import cv2
import numpy as np

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms
from finalProject.utils.keyPoints.AlgoritamKeyPoints import surf_keypoints_detection


def kaze_matcher(desc1, desc2, threshold=0.8):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []  # Apply ratio test
    for m, n in nn_matches:
        if m.distance < threshold * n.distance:
            good.append([m])
    return good


def find_closest_person(target, people_list, config: "config file"):
    """Returns a list of (person, accuracy) pairs of all people in target video
    and their accuracy matching to the person in target argument"""

    target_keypoints, target_descriptors = surf_keypoints_detection(target["frame"])
    if target_keypoints is None or target_descriptors is None:
        return None  # don't have key points for this human

    max_match = []
    match_p = []

    for p in people_list:
        if len(p.frames) > config["max_length_frames"]:  # remove trace frames
            p.history.extend(p.frames[0:len(p.frames) - config["max_length_frames"]])
            p.frames = p.frames[-config["max_length_frames"]:]

        for index, frame in enumerate(p.frames):
            kp, dp = surf_keypoints_detection(frame.frame_image)
            if kp is None or dp is None:
                continue
            else:
                good_match = flann_matcher(target_descriptors, dp, config["FlannMatcherThreshold"])

            if len(target_keypoints) == 0:
                acc = 0
            else:
                acc = len(good_match) / (len(target_descriptors))

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


def compare_between_descriptors(source_descriptor, target_descriptor):
    binary_algo = [NamesAlgorithms.ORB.name, NamesAlgorithms.KAZE.name]
    float_algo = [NamesAlgorithms.SURF.name, NamesAlgorithms.SIFT.name]
    results = []
    for algo in binary_algo:
        des_s = source_descriptor[algo]
        des_t = target_descriptor[algo]
        if len(des_s) == 0 or len(des_t) == 0:
            results.append(0)
        else:
            b_matches = kaze_matcher(des_s, des_t)
            b_acc = len(b_matches) / (len(des_s))
            results.append(min(b_acc, 1))

    for algo in float_algo:
        des_s = source_descriptor[algo]
        des_t = target_descriptor[algo]
        if len(des_s) == 0 or len(des_t) == 0:
            results.append(0)
        else:
            f_matches = flann_matcher(des_s, des_t)
            f_acc = len(f_matches) / (len(des_s))
            results.append(min(f_acc, 1))

    return np.mean(results)


def compute_accuracy_table(source_person, target_people):
    acc_table = {}
    for t_id, t_person in enumerate(target_people):  # loop through both keys and values of dict
        frames_table = np.zeros(shape=[len(t_person.frames), len(source_person.frames)])  # rows=num of t_frames, cols=num of s_frames
        for t_frame_index, t_frame in enumerate(t_person.frames):
            for s_frame_index, s_frame in enumerate(source_person.frames):
                frames_table[t_frame_index, s_frame_index] = compare_between_descriptors(s_frame.frame_des, t_frame.frame_des)

        max_acc = np.amax(frames_table)
        ind = np.unravel_index(np.argmax(frames_table, axis=None), frames_table.shape)
        acc_table[t_id] = {"maxAcc": max_acc,
                           "target_frame": t_person.frames[ind[0]],
                           "source_frame": source_person.frames[ind[1]]}
    return acc_table
