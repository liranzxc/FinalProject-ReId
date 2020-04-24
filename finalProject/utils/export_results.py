import os
import cv2
import numpy as np
import pip
import base64
import json
import pprint
import matplotlib.pyplot as plt
import random
import pandas as pd
import io
from PIL import Image
from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import draw_final_results
from finalProject.utils.images.imagesUtils import Image
from finalProject.utils.keyPoints.AlgoritamKeyPoints import create_keypoints_descriptors, sift_keypoints_detection
from finalProject.utils.matchers.Matchers import compare_between_descriptors, flann_matcher
from finalProject.utils.preprocessing.preprocess import read_frames_from_video, check_frames_exist, reduce_noise, removeRemovalColor
from finalProject.utils.tracking.TrackingByYolo import source_detection_by_yolo, tracking_by_yolo
import datetime


def sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_intensity = cv2.sqrt(cv2.addWeighted(cv2.pow(sobel_x, 2.0), 1.0, cv2.pow(sobel_y, 2.0), 1.0, 0.0))
    return sobel_intensity


def sobel_keypoints(image):
    sobel_image = sobel(image)
    # norm
    image8bit = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp, des = sift_keypoints_detection(image8bit)
    return kp, des, image8bit


def forward(frames):
    outputs = []
    for frame in frames:
        frame.gray_image = sobel(frame.gray_image)
        key_des_image = sobel_keypoints(frame.gray_image)
        outputs.append(key_des_image)
    return outputs


def save_image(image, folder_name, image_name):
    path = "./reports/"+str(folder_name)+"/"+image_name+".png"
    cv2.imwrite(path, image)


def convert(folder_name, counter):
    def convert_images(output, counter):
        saveImage(output[0], folder_name,"comparison_"+counter)
        saveImage(output[1], folder_name,"source_"+counter)
        saveImage(output[2], folder_name,"target_"+counter)
        return output[3:]
    return convert_images


def export_to_excel(outputs):
    curr_time = datetime.datetime.now()
    print(curr_time.strftime("%d-%m-%Y_%Hh-%Mm"))

    folder_name = "folderReport_" + str(curr_time.strftime("%d-%m-%Y_%Hh-%Mm"))
    path = "./reports/" + folder_name
    os.makedirs(path, mode=0o777, exist_ok=False)

    counts = []
    for i, v in enumerate(outputs):
        counts.append(str(i))

    excel_table = list(map(convert(folder_name, str(i)), outputs, counts))
    df1 = pd.DataFrame(np.array(excel_table),
                       columns=['source_keypoints', 'target_keypoints', 'total_matches', 'accuracy'])
    df1.to_excel("./reports/" + folder_name + "/report_" + folder_name + ".xls", float_format="%.2f", index=False,)


def cross_correction(key_des_image_source, key_des_image_target, threshold=0.8):
    (k1, d1, source_img) = key_des_image_source
    (k2, d2, target_img) = key_des_image_target
    match = flann_matcher(d1, d2, threshold=threshold)

    if len(match) == 0:
        return None

    output = cv2.drawMatchesKnn(source_img, k1, target_img, k2, match, outImg=None)
    acc = len(match)/len(k1)
    return [output, source_img, target_img, len(k1), len(k2), len(match), min(acc, 1)]  # item = (outputs,source_img,target_img,len(k1),len(k2),acc..)


def sort_by_acc_func(output_row):
    return output_row[6]


def save_people_frames(source_person, target_people):
    path = "./reports/"+str(folder_name)+"/"+image_name+".png"
    cv2.imwrite(path, image)


def plot_square(images, _titles, threshold, each_column=10):
    number_images = len(images)
    rows = (number_images // each_column)
    index = 1

    print(rows)
    fig, axes = plt.subplots(rows, each_column, figsize=(48, 24),dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax,image,title in zip(axes.flat, images, _titles):
        ax.imshow(image)
        ax.set_title(title)
        index += 1

    rand = random.uniform(0, 1)
    pathX = "./figures/fig_test"+str(rand)+"_thresh_"+str(threshold)+".svg"
    print("saving on", pathX)
    plt.savefig(pathX, dpi=fig.dpi)
    plt.clf()
    plt.close()
