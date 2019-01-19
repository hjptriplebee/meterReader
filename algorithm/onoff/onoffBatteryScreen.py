from algorithm.Common import meterFinderBySIFT
from algorithm.debug import *

import cv2
import numpy as np


def onoffBattery(image, info):
    """
    识别电池屏内部的开关是向上还是向下
    :param image: image
    :param info: information
    :return:
    """
    meter = meterFinderBySIFT(image, info)
    # print(meter.shape)
    meter = cv2.cvtColor(meter, cv2.COLOR_BGR2GRAY)
    meter = cv2.resize(meter, (200, 200))

    # 截取中心区域
    center = meter[60:120, 85:115]

    # 自适应二值化
    ret, thresh = cv2.threshold(center, 0, 255, cv2.THRESH_OTSU)

    # 比较上下区域的色度和，若上部区域比下部区域小，则说明开关是向上打开
    upRegion = np.sum(meter[:30])
    downRegion = np.sum(meter[30:])

    if ifShow:
        imgShow = np.hstack((center, thresh))
        cv2.imshow("ret", imgShow)
        cv2.waitKey(0)
        print(upRegion, downRegion)

    return 1 if upRegion < downRegion else 0
