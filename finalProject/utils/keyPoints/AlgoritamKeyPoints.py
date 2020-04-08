import cv2

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms


def KeyPointsBinary(img, threshold):
    kpOrb, desOrb = OrbDetectKeyPoints(img, threshold)
    kpKaze, desKaze = KazeDetectKeyPoints(img)
    return [(kpOrb, desOrb), (kpKaze, desKaze)]


def KeyPointsFloat(img, threshold):
    kpSurf, desSurf = SurfDetectKeyPoints(img)
    kpSift, desSift = SiftDetectKeyPoints(img)
    return [(kpSurf, desSurf), (kpSift, desSift)]


def SiftDetectKeyPoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def SurfDetectKeyPoints(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    return kp, des


def OrbDetectKeyPoints(img, n_features=200):
    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=n_features)  # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


def KazeDetectKeyPoints(image):
    kaze = cv2.AKAZE_create()
    kp, des = kaze.detectAndCompute(image, None)
    return kp, des


def calculate_key_points(image, keyPointFunction):
    return keyPointFunction(image)


def append_descriptors_to_frame(keys, descriptions, algo, frame):
    if keys is None or descriptions is None or len(keys) == 0 or len(descriptions) == 0:
        frame.frame_keypoints[algo] = []
        frame.frame_des[algo] = []
    else:
        frame.frame_keypoints[algo] = keys
        frame.frame_des[algo] = descriptions


def create_key_points_descriptors(people_list):  # people_list is a list of elements of type Person
    """"returns a dictionary<int,[]> of (key,value)=(person_id,listOfDescriptors),
    where listOfDescriptors is a list that its elements are dictionaries -
    - each dictionary<String,{}> has (key,value)= (algorithmName, dictOfKeysDes),
    where dictOfKeysDes is a dictionary<String,[]> with two elements: 1=('keys', listOfKeyPoints), 2=('des',listOfDescriptors)"""
    descriptor = {}

    for person in people_list:
        descriptor[person.person_id] = []

        for frame in person.frames:
            kOrb, desOrb = calculate_key_points(frame.frame_image, OrbDetectKeyPoints)
            kKaze, desKaze = calculate_key_points(frame.frame_image, KazeDetectKeyPoints)
            kSift, desSift = calculate_key_points(frame.frame_image, SiftDetectKeyPoints)
            kSurf, desSurf = calculate_key_points(frame.frame_image, SurfDetectKeyPoints)

            append_descriptors_to_frame(kOrb, desOrb, NamesAlgorithms.ORB.name, frame)
            append_descriptors_to_frame(kKaze, desKaze, NamesAlgorithms.KAZE.name, frame)
            append_descriptors_to_frame(kSift, desSift, NamesAlgorithms.SIFT.name, frame)
            append_descriptors_to_frame(kSurf, desSurf, NamesAlgorithms.SURF.name, frame)

            descriptor[person.personId].append(frame)

    return descriptor
