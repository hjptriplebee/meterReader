import cv2
from algorithm.debug import *
from algorithm.Common import *

def digitArrester(src, info):
    """
    :param src: ROI
    :return:value
    """
    # need rebuild
    meter = meterFinderByTemplate(src, info["template"])
    if ifShow:
        cv2.imshow("image", meter)
        cv2.waitKey(0)
    return 0
