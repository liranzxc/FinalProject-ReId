"""# resize image"""
from builtins import dict
import random as rnd
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np


class Image(object):
    def __init__(self, path):
        self.path = path
        self.bgr_img = None
        self.gray_img = None
        self.rgb_img = None

    def read_image(self, return_img=False):
        self.bgr_img = cv2.imread(self.path)
        if return_img:
            return self.bgr_img

    def rgb(self, return_img=False):
        self.rgb_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2RGB)
        if return_img:
            return self.rgb_img

    def gray(self, return_img=False):
        self.gray_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2GRAY)
        if return_img:
            return self.gray_img

    def show(self, img, title="Image"):
        if len(img.shape) != 3:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.show()

    def show_all(self, image_list, title_list):
        plt.figure(figsize=[20, 10])
        assert len(image_list) == len(title_list), "Houston we've got a problem"
        N = list(image_list)
        for index, (img, title) in enumerate(zip(image_list, title_list)):
            plt.subplot(1, N, index + 1)
            if len(img.shape) != 3:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(title)
        plt.show()

    @staticmethod
    def convolve2d(image, kernel, stride=1):
        if len(image.shape) == 2:
            rows, cols = image.shape
            F = kernel.shape[0]
            rows_conv = rows - F + 1
            cols_conv = cols - F + 1
            image_filter = np.zeros((rows_conv, cols_conv))
            image_copy = image.copy()
            for i in range(0, rows_conv, stride):
                for j in range(0, cols_conv, stride):
                    win = image_copy[i:i + F, j:j + F]
                    image_filter[i, j] = np.sum(np.dot(kernel, win))

            return image_filter
        else:
            rows, cols, channels = image.shape
            F = kernel.shape[0]
            rows_conv = rows - F + 1
            cols_conv = cols - F + 1
            image_filter = np.zeros((rows_conv, cols_conv, channels))
            image_copy = image.copy()
            for channel in range(0, channels):
                for i in range(0, rows_conv, stride):
                    for j in range(0, cols_conv, stride):
                        win = image_copy[i:i + F, j:j + F, channel]
                        image_filter[i, j, channel] = np.sum(np.dot(kernel, win))

            return image_filter



def distance(pt1, pt2):
    d = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    return d


def getCenter(box):
    return (box[1][0] + box[0][0]) // 2, (box[1][1] + box[0][1]) // 2


def resizeImage(source, fx, fy):  # size is tuple (w,h)
    return cv2.resize(source, None, fx=fx, fy=fy)


def Accuracy(kp, matches):
    return len(matches) / (len(kp))


def ShowMatch(source, kp, target, kp2, matches):
    draw_params = dict(
        singlePointColor=None,
        matchColor=(255, 0, 0),
        flags=2)

    print("matches length {}".format(len(matches)))
    img3 = cv2.drawMatchesKnn(source, kp, target, kp2, matches[:80], None, **draw_params)
    cv2.imshow("Match", img3)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


"""# Quick started"""
