from Common import *

def SF6Reader(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    # your method
    print("SF6Reader called!!!")
    # template match
    # meter = meterFinderByTemplate(image, info["template"])
    meter = meterFinderBySIFT(image, info["template"])
    res = AngleFactory.calPointerValueByAngle(np.array([-1, 0]), np.array([0, -1]), np.array([0, 0]), np.array([1, 1]),
                                              0, 15)
    # cv2.imshow("meter", meter)
    # cv2.waitKey(0)

    return res
