import cv2
import sys
import numpy as np
import torch
from collections import defaultdict

from algorithm.pressure.digitPressure import digitPressure


def videoDigit(video, info):
    """
    :param video: VideoCapture Input
    :param info: info
    :return: 
    """
    pictures = getPictures(video)  # 获得视频的帧，有少量重复帧
    def emptyLsit():
        return []

    imagesDict = defaultdict(emptyLsit)
    for frame in pictures:
        res = digitPressure(frame, info)
        index = checkFrame(frame, info)

        imagesDict[0] += [res]
    return imagesDict[0]


def getPictures(videoCapture):
    """
    截取视频中每隔固定时间出现的帧
    :param videoCapture: 输入视频
    :return: 截取的帧片段
    """
    pictures = []
    cnt = 0
    skipFrameNum = 30
    while True:
        ret, frame = videoCapture.read()
        # print(cnt, np.shape(frame))
        cnt += 1
        if frame is None:
            break
        if cnt % skipFrameNum:
            continue
        pictures.append(frame)

    videoCapture.release()
    return pictures


def checkFrame(image, info):
    """
    判断图片的类型A，AB等
    :param image:  image
    :param info: info
    :return: 出现次序0.1.2.3
    """
    return 0
