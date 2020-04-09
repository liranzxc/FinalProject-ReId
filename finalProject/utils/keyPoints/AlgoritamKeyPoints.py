import cv2

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms


def binary_keypoints(image, threshold):
    kpOrb, desOrb = orb_keypoints_detection(image, threshold)
    kpKaze, desKaze = kaze_keypoints_detection(image)
    return [(kpOrb, desOrb), (kpKaze, desKaze)]


def float_keypoints(image, threshold):
    kpSurf, desSurf = surf_keypoints_detection(image)
    kpSift, desSift = sift_keypoints_detection(image)
    return [(kpSurf, desSurf), (kpSift, desSift)]


def sift_keypoints_detection(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des


def surf_keypoints_detection(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    return kp, des


def orb_keypoints_detection(img, n_features=200):
    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=n_features)  # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


def kaze_keypoints_detection(image):
    kaze = cv2.AKAZE_create()
    kp, des = kaze.detectAndCompute(image, None)
    return kp, des


def calculate_key_points(image, keyPointFunction):
    return keyPointFunction(image)


def save_descriptors_to_frame(keypoints, descriptors, algo, frame):
    if keypoints is None or descriptors is None or len(keypoints) == 0 or len(descriptors) == 0:
        frame.frame_keypoints[algo] = []
        frame.frame_des[algo] = []
    else:
        frame.frame_keypoints[algo] = keypoints
        frame.frame_des[algo] = descriptors


def create_keypoints_descriptors(people_list):  # people_list is a list of elements of type Person
    """"returns a dictionary<int,[]> of (key,value)=(person_id,frames_list),
    where frames_list is a list that its elements are CroppedFrames with their keypoints and descriptors"""
    descriptor = {}

    for person in people_list:
        descriptor[person.person_id] = []

        for frame in person.frames:
            kp_orb, des_orb = calculate_key_points(frame.frame_image, orb_keypoints_detection)
            kp_kaze, des_kaze = calculate_key_points(frame.frame_image, kaze_keypoints_detection)
            kp_sift, des_sift = calculate_key_points(frame.frame_image, sift_keypoints_detection)
            kp_surf, des_surf = calculate_key_points(frame.frame_image, surf_keypoints_detection)

            save_descriptors_to_frame(kp_orb, des_orb, NamesAlgorithms.ORB.name, frame)
            save_descriptors_to_frame(kp_kaze, des_kaze, NamesAlgorithms.KAZE.name, frame)
            save_descriptors_to_frame(kp_sift, des_sift, NamesAlgorithms.SIFT.name, frame)
            save_descriptors_to_frame(kp_surf, des_surf, NamesAlgorithms.SURF.name, frame)

            descriptor[person.person_id].append(frame)

    return descriptor
