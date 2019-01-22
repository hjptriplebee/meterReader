import os
import random
import cv2
from collections import defaultdict
import numpy as np

def fillAndResize(image):
    """
    将输入图像填充为正方形且变换为（28，28）
    :param image:
    :return:
    """
    h, w = image.shape
    l = max(w, h)
    ret = np.zeros((l, l), np.uint8)
    leftTop = np.array([l/2-w/2, l/2-h/2], np.uint8)
    ret[leftTop[1]:leftTop[1]+h, leftTop[0]:leftTop[0]+w] = image
    ret = cv2.resize(ret, (28, 28), interpolation=cv2.INTER_CUBIC)
    return ret


def move(img):
    pos = random.randint(2, 4)
    hline = np.zeros((pos, 28))
    vline = np.zeros((28, pos))
    up = np.vstack((img, hline))[pos:]
    down = np.vstack((hline, img))[:28]
    left = np.hstack((img, vline))[:, pos:]
    right = np.hstack((vline, img))[:, :28]

    # imgShow = np.hstack((up, down, left, right))
    # cv2.imshow("move", imgShow)
    # cv2.waitKey(0)

    return [up, down, left, right]


def noise(img):
    for i in range(28):
        for j in range(28):
            if random.randint(1, 20) <= 1 and img[i, j] == 0:
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
    tot = "train/"
    source = "images/LCD/"+tot
    target = "images/LCD_enhanced/"+tot
    for i in range(11):
        label = str(i)
        images = os.listdir(source+str(i))

        if os.path.exists(target+label):
            os.system("rm -rf {}".format(target+label))
            os.mkdir(target+label)

        for im in images:
            img = cv2.imread(source+label+"/"+im)
            img = np.squeeze(img[:, :, 0])
            newImages = move(img)+shrink(img)+[img]

            for i in newImages:
                newImages = newImages + noise(i)

            for i in range(len(newImages)):
                # if random.randint(1, 2) == 1:
                #     continue
                count[label] += 1
                cv2.imwrite(target+label+"/"+label+"_"+str(count[label])+".bmp", newImages[i])


enhance()

