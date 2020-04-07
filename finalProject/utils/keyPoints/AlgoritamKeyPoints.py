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


"""# Surf Algorithm"""
def SurfDetectKeyPoints(img):
    suft = cv2.xfeatures2d.SURF_create()
    kp, des = suft.detectAndCompute(img, None)
    return kp, des


"""# Orb algorithm"""


def OrbDetectKeyPoints(img, n_features=200):
    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=n_features)  # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


"""# Kaze algorithm"""
def KazeDetectKeyPoints(img):
    kaze = cv2.AKAZE_create()
    kp, des = kaze.detectAndCompute(img, None)
    return kp, des


def CalculationKeyPoint(image, keyPointFunction):
    return keyPointFunction(image)


def appendToFrameObject(keys, descriptions, label, frameObject):
    if keys is None or descriptions is None or len(keys) == 0 or len(descriptions) == 0:
        frameObject[label] = {"keys": [], "des": []}
    else:
        frameObject[label] = {"keys": keys, "des": descriptions}


def create_key_points_descriptors(people_list):  # people_list is a list of elements of type Person
    """"returns a dictionary<int,[]> of (key,value)=(person_id,listOfDescriptors),
    where listOfDescriptors is a list that its elements are dictionaries -
    - each dictionary<String,{}> has (key,value)= (algorithmName, dictOfKeysDes),
    where dictOfKeysDes is a dictionary<String,[]> with two elements: 1=('keys', listOfKeyPoints), 2=('des',listOfDescriptors)"""
    descriptor = {}

    for target in people_list:
        descriptor[target.personId] = []

        for frame in target.frames:
            kOrb, desOrb = CalculationKeyPoint(frame, OrbDetectKeyPoints)
            kKaze, desKaze = CalculationKeyPoint(frame, KazeDetectKeyPoints)
            kSift, desSift = CalculationKeyPoint(frame, SiftDetectKeyPoints)
            kSurf, desSurf = CalculationKeyPoint(frame, SurfDetectKeyPoints)

            frameObject = {
                "frame": frame,
            }
            appendToFrameObject(kOrb, desOrb, NamesAlgorithms.ORB.name, frameObject)
            appendToFrameObject(kKaze, desKaze, NamesAlgorithms.KAZE.name, frameObject)
            appendToFrameObject(kSift, desSift, NamesAlgorithms.SIFT.name, frameObject)
            appendToFrameObject(kSurf, desSurf, NamesAlgorithms.SURF.name, frameObject)

            descriptor[target.personId].append(frameObject)

    return descriptor
