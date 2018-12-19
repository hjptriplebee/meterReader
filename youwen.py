from Common import *

ifshow = True

def red(image):
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    Lower = np.array([156, 120, 80])  # 要识别颜色的下限
    Upper = np.array([179, 240, 220])  # 要识别的颜色的上限
    # mask是把HSV图片中在颜色范围内的区域变成白色，其他区域变成黑色
    mask = cv2.inRange(HSV, Lower, Upper)
    if ifshow:
        cv2.imshow("inRange", mask)
        cv2.waitKey(0)

    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    mask = cv2.erode(mask, kernel_1)
    if ifshow:
        cv2.imshow("erode", mask)
        cv2.waitKey(0)

    mask = cv2.dilate(mask, kernel_2, 2)
    if ifshow:
        cv2.imshow("dilate", mask)
        cv2.waitKey(0)

    mask = cv2.blur(mask, (9, 9))
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if ifshow:
        cv2.imshow("thresh", mask)
        cv2.waitKey(0)
    return mask


def contours_check(image, center):
    img, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    contours.reverse()
    reach = 0
    target = [0, 0]
    l = min(2, len(contours))
    for i in range(l):
        for point in contours[i]:
            point = np.squeeze(point)
            dis = np.sqrt((point[0]-center[0])**2 + (point[1]-center[1])**2)
            if dis > reach:
                reach = dis
                target = point
    if ifshow:
        cv2.circle(image, (target[0], target[1]), 5, (0, 0, 255), -1)
        cv2.imshow("point", image)
        cv2.waitKey(0)
    return target


def angle(a, b, c):
    s = (a[0]-c[0])*(b[1]-c[1])-(b[0]-c[0])*(a[1]-c[1])
    x = a-c
    y = b-c
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    if s < 0:
        angle2 = 360-angle2
    return angle2


def youwen(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    # your method
    print("YouWen Reader called!!!")
    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    # template match
    # print(info["template"])
    cv2.imshow("image", image)
    cv2.waitKey(0)
    meter = meterFinderByTemplate(image, info["template"])
    meter[int(0.6*meter.shape[0]):] *= 0

    mask_meter = red(meter)
    point = contours_check(mask_meter, center)

    # total = angle(start, end, center)
    # read = angle(start, point, center)
    print(start, end, center, point)
    degree = AngleFactory.calPointerValueByOuterPoint(start, end, center, point, info["startValue"], info["totalValue"])

    print("degree", degree)

    cv2.circle(meter, (info["startPoint"]["x"], info["startPoint"]["y"]), 5, (0, 0, 255), -1)
    cv2.circle(meter, (info["endPoint"]["x"], info["endPoint"]["y"]), 5, (0, 0, 255), -1)
    cv2.circle(meter, (info["centerPoint"]["x"], info["centerPoint"]["y"]), 5, (0, 0, 255), -1)
    cv2.circle(meter, (point[0], point[1]), 5, (0, 0, 255), -1)
    cv2.imshow("meter", meter)
    cv2.waitKey(0)
    # return int(read/total*(info["totalValue"]-info["startValue"]))
    return degree
