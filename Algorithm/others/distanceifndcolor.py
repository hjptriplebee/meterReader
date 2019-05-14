import cv2
import numpy as np
import json
import Algorithm.template as tmp
from Algorithm.others.colordetect import meterFinderNoinfoBySIFT


def FindcolorByDistance(image, info):
    """
    识别含有红色高亮数字区域的模块
    :param image: 输入图片
    :param info:  标记信息
    :return:
    """
    x = 1
    y = 1
    xlen = image.shape[0]
    ylen = image.shape[1]
    xbigin = 0
    ybigin = 0
    result = []
    print(int(xbigin * ybigin * 0.3 * 255))
    for i in range(x):
        for j in range(y):
            preImage = image[xbigin + xlen * j:xbigin + xlen * (j + 1), ybigin + ylen * i:ybigin + ylen * (i + 1)]
            # cv2.namedWindow("preImage", cv2.WINDOW_NORMAL)
            # cv2.imshow("preImage", preImage)
            # cv2.waitKey(0)
            ImageKey = np.sum(preImage)
            if ImageKey > xbigin * ybigin * 0.1 * 255:
                result.append(1)
            else:
                result.append(0)
    return result


def PreProcessing(img, info):
    # 图像预处理
    imgtemp = cv2.imread("C:/Users/crow/Desktop/picturesource/template/light5.jpg")
    imgsift = meterFinderNoinfoBySIFT(img, imgtemp)  # SIFT匹配
    print(imgsift.shape)
    gray = cv2.cvtColor(imgsift, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    ret, thresh = cv2.threshold(gray, gray.max() - 25, gray.max(), cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=3)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    # print(imgtemp.shape,thresh.shape)
    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    cv2.imshow("thresh", imgsift)
    cv2.waitKey(0)
    result = FindcolorByDistance(thresh, info)
    print(result)


if __name__ == '__main__':
    img = cv2.imread("C:/Users/crow/Desktop/0413/7.jpg")
    # file = open("../config/2-2_1" + ".json")
    PreProcessing(img, 0)
