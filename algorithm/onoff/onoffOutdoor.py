import math
import numpy as np
import cv2
from algorithm.debug import *
from algorithm.Common import AngleFactory, meterFinderByTemplate


def onoffOutdoor(image, info):
    """
    :param image:whole image
    :param info:bileiqi2 config
    :return:
    """
    # need rebuild
    meter = meterFinderByTemplate(image, info["template"])
    if ifShow:
        cv2.imshow("image", meter)
        cv2.waitKey(0)
    return 0
