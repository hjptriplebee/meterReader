import cv2
import numpy as np
import functools
import json


def meterFinderByTemplate(image, template):
    """
    locate meter's bbox
    :param image: image
    :param template: template
    :return: bbox image
    """
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    w, h, _ = template.shape

    # for test
    # cv2.imshow("test", img)
    # img = (img * 0.5).astype(np.uint8) # test
    # cv2.imshow("test2", img)
    # cv2.waitKey(0)

    i = 5
    res = cv2.matchTemplate(image, template, methods[i])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    if methods[i] in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    bottomRight = (topLeft[0] + h, topLeft[1] + w)

    return image[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]

