from Common import *


def youwen(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    # your method
    print("YouWen Reader called!!!")
    # template match
    print(info["template"])
    cv2.imshow("image", image)
    cv2.waitKey(0)
    meter = meterFinderByTemplate(image, info["template"])
    cv2.imshow("meter", meter)
    cv2.waitKey(0)

    return 1