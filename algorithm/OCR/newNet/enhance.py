import os
import random
import cv2
from collections import defaultdict
from algorithm.OCR.newNet.get_test import fillAndResize
import numpy as np


def move(img):
    hline = np.zeros((1, 28))
    vline = np.zeros((28, 1))
    up = np.vstack((img, hline))[1:]
    down = np.vstack((hline, img))[:28]
    left = np.hstack((img, vline))[:, 1:]
    right = np.hstack((vline, img))[:, :28]

    # imgShow = np.hstack((up, down, left, right))
    # cv2.imshow("move", imgShow)
    # cv2.waitKey(0)

    return [up, down, left, right]


def noise(img):
    for i in range(28):
        for j in range(28):
            if random.randint(1, 10) <= 1 and img[i, j] == 0:
                img[i, j] = 255
    # cv2.imshow("noise", img)
    # cv2.waitKey(0)
    return [img]


def shrink(img):
    shrink_ = cv2.resize(img, (24, 24))
    shrink_ = np.vstack((np.zeros((4, 24)), shrink_, np.zeros((4, 24))))
    shrink_ = fillAndResize(shrink_)

    expand_ = cv2.resize(img, (30, 30))
    expand_ = expand_[2:28]
    expand_ = fillAndResize(expand_)

    # cv2.imshow("shrink", np.hstack((shrink_, expand_)))
    # cv2.waitKey(0)

    return [shrink_, expand_]


def zero():
    return 0

def enhance():
    count = defaultdict(zero)
    images = os.listdir("images/all_data")
    for im in images:
        label = im.split("_")[0]
        img = cv2.imread("images/all_data/" + im)
        img = np.squeeze(img[:, :, 0])

        newImages = move(img)+shrink(img)+[img]
        for i in newImages:
            newImages = newImages + noise(i)

        for i in range(len(newImages)):
            if random.randint(1, 2) == 1:
                continue
            count[label] += 1
            cv2.imwrite("images/enhanced/"+label+"_"+str(count[label])+".bmp", newImages[i])


enhance()

