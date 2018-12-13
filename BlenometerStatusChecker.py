from Common import *
import numpy as np
import functools
import json

def cmp(contour1, contour2):
    return cv2.contourArea(contour2) - cv2.contourArea(contour1)


def cmpCircle(circle1, circle2):
    return circle2[2] - circle1[2]

def checkBelometerUpAndDownStatus(image, info):
    """
    :param image: input image
    :param info: image description
    :return:  dumps string by json format ,such as {"meterId": "VALUE"}.
    """
    src = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(src, (3, 3), 0, None, 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY, None)
    canny = cv2.Canny(gray, 75, 75 * 2, None)
    # cv2.imshow("Canny", canny)
    # cv2.waitKey(0)
    im, image_contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(image_contours,
                             key=functools.cmp_to_key(cmp))

    # extract the rectangle of ROI
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(sorted_contours[0])
    # extract ROI source
    roi = gray[y_roi: y_roi + h_roi, x_roi:x_roi + w_roi]
    # continue search circle  when previous HoughCircle found nothing.
    try_time = 3
    accumulation = 90
    is_find_circles = False
    # cv2.imshow("ROI", roi)
    # cv2.waitKey(0)
    circles_in_roi = []
    for i in range(0, try_time):
        circles_in_roi = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, 2, 10, None, 150, accumulation, 10, 20)
        # adjust the accumulation threshold,find the circle as could as possible
        if circles_in_roi is None:
            accumulation -= 20
            continue
        circles_in_roi = np.uint16(np.round(circles_in_roi))
        circles_in_roi = circles_in_roi[0][:]
        for circle in circles_in_roi:
            # HoughCircles return a [0,0,0] tuple, represents circle not found,
            if circle[2] == 0:
                print("Lost circle location.Search Operation  remained :{} times\n".format(try_time - i - 1))
                break
            is_find_circles = True
        accumulation -= 20
        try_time = try_time - 1

    if is_find_circles == False:
        print("Can not located the element of Blenometer.\n")
        return
    # print(circles_in_roi)
    # get the circle center Y aix of min circle
    min_center = min(circles_in_roi, key=lambda c: c[2])
    center_y = min_center[2]
    # judge if the meter feature is below the medium line of ROI or not.
    # (default recognizes the feature circle is the smallest)
    res = {}
    if info is not None:
        if y_roi + h_roi / 2 > center_y:
            res[info["name"]] = {"value": "Up"}
            print("Status : Up\n")
        else:
            res[info["name"]] = {"value": "Down"}
    else:
        print("Meter info is not provided.Return error message.")
        res['error'] = 'Lost Meter Id.'
    rgb_roi = src[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]
    cv2.circle(rgb_roi, (min_center[0], min_center[1]), min_center[2], (255, 0, 0), cv2.LINE_4)
    cv2.imshow("ROI", rgb_roi)
    cv2.waitKey(0)

    return json.dumps(res)
    # template match
    # meter = meterFinderByTemplate(image, info["template"])

    # cv2.imshow("meter", meter)
    # cv2.waitKey(0)

def readBlenometerStatus(image, info):
    print("Blenometer Reader called!!!")
    if image is None:
        print("Resolve Image Error.Inupt image is Empty.")
        return
    return checkBelometerUpAndDownStatus(image, info)

# test interface

# if __name__ == '__main__':
# src = cv2.imread('image/IMG_7610.JPG')
# cv2.imshow("Imge",src)
#    res1 = readBlenometerStatus(cv2.imread('image/IMG_7612.JPG'), None)
#    print(res1)
#    info = {"name": "Belometer1"}
#    res2 = readBlenometerStatus(cv2.imread('image/IMG_7612.JPG'), info)
#    print(res2)
