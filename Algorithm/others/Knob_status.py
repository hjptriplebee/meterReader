import math
import cv2
import numpy as np
import json
from Algorithm.others.colordetect import meterFinderNoinfoBySIFT
from Algorithm.others.colordetect import findcenter


def calc_angle(x_point_s, y_point_s, x_point_e, y_point_e):
    angle = 0
    y_se = y_point_e - y_point_s
    x_se = x_point_e - x_point_s
    # print(x_se,y_se)
    if x_se == 0 and y_se > 0:
        angle = 90
    if x_se == 0 and y_se < 0:
        angle = 270
    if y_se == 0 and x_se > 0:
        angle = 0
    if y_se == 0 and x_se < 0:
        angle = 180
    if x_se > 0 and y_se > 0:
        angle = math.atan(abs(y_se / x_se)) * 180 / math.pi
    elif x_se < 0 and y_se > 0:
        angle = 90 + math.atan(abs(x_se / y_se)) * 180 / math.pi
    elif x_se < 0 and y_se < 0:
        angle = 180 + math.atan(abs(y_se / x_se)) * 180 / math.pi
    elif x_se > 0 and y_se < 0:
        angle = 270 + math.atan(abs(x_se / y_se)) * 180 / math.pi
    return angle


def decide_status(angle):
    status = -1
    if angle >= 345 or angle < 15:
        status = 0
    elif angle >= 15 and angle < 75:
        status = 1
    elif angle >= 75 and angle < 105:
        status = 2
    elif angle >= 105 and angle < 165:
        status = 3
    elif angle >= 165 and angle < 195:
        status = 4
    elif angle >= 195 and angle < 255:
        status = 5
    elif angle >= 255 and angle < 285:
        status = 6
    elif angle >= 285 and angle < 345:
        status = 7
    return status


def knobstatus(img, info):
    # tmpimg = info["template"]
    # siftimg = meterFinderNoinfoBySIFT(img,tmpimg)
    green_min = np.array([26, 5, 210])
    green_max = np.array([35, 20, 254])
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(imgHSV, green_min, green_max)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    # thresh = cv2.dilate(thresh, kernel, iterations=1)
    # cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    center = findcenter(thresh)
    # print(thresh.shape)
    x = 0
    y = 0
    w, h = img.shape[:2]
    # print(w,h)
    for i in range(len(center)):
        x = center[i][0] + x
        y = center[i][1] + y
    x = x / len(center)
    y = y / len(center)
    x = x - h / 2
    y = w / 2 - y
    # print(x,y,126 - w/2,119 - h/2)
    angle = calc_angle(126 - h / 2, w / 2 - 119, x, y)
    status = decide_status(angle)
    return status


if __name__ == '__main__':
    img = cv2.imread("E:/picture/template/Knob1.jpg")
    result = knobstatus(img, 0)
    print(result)
# compare_images(img1,img2)
