from Common import *

def SF6(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    # your method
    print("SF6Reader called!!!")
    # template match
    meter = meterFinderByTemplate(image, info["template"])
    cv2.imshow("meter", meter)
    cv2.waitKey(0)

    return 1