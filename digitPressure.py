from Common import *
from OCR.utils import *

import sys
sys.path.append("OCR/LeNet")
ifShow = True

black_range = [np.array([0, 0, 0]), np.array([180, 255, 220])]


def fillAndResize(image):
    h, w = image.shape
    l = max(2*w, h+10)
    ret = np.zeros((l, l), np.uint8)
    leftTop = np.array([l/2-w/2, l/2-h/2], np.uint8)
    ret[leftTop[1]:leftTop[1]+h, leftTop[0]:leftTop[0]+w] = image
    ret = cv2.resize(ret, (28, 28), interpolation=cv2.INTER_CUBIC)
    return ret


def digitPressure(image, info):
    net = leNetOCR()
    svm = svmOCR()
    print("模型加载完毕,,,")
    template = meterFinderByTemplate(image, info["template"])

    start = ([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = ([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = ([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [200, 0], [200, 100], [0, 100]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(template, M, (200, 100))

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 30, 70)

    if ifShow:
        cv2.imshow("edge", edge)
        cv2.waitKey(0)

    split = [2, 34, 69, 102, 132, 163]
    height = [23, 88]
    res = 0

    print("开始测试")
    for i in range(5):
        num = edge[height[0]:height[1], split[i]:split[i+1]]
        num = cv2.resize(num, (0, 0), fx=2, fy=2)

        num = cv2.dilate(num, np.ones((9, 9), np.uint8))
        num = cv2.erode(num, np.ones((3, 3), np.uint8))
        num = cv2.dilate(num, np.ones((7, 7), np.uint8), 3)
        num = cv2.erode(num, np.ones((15, 15), np.uint8), 3)

        num = cv2.resize(num, (0, 0), fx=0.5, fy=0.5)

        ret, num = cv2.threshold(num, 0, 255, cv2.THRESH_BINARY)

        inputNum = fillAndResize(num)
        numNet = net.recognizeNet(inputNum)
        numSvm = svm.recognizeSvm(inputNum)

        if ifShow:
            cv2.imshow("fill", inputNum)
            print(numNet, numSvm )
            cv2.waitKey(0)

        res = 10*res + numNet
    print("读数", res)

    if ifShow:
        cv2.circle(template, (start[0], start[1]), 5, (0, 0, 255), -1)
        cv2.circle(template, (end[0], end[1]), 5, (0, 255, 0), -1)
        cv2.circle(template, (center[0], center[1]), 5, (255, 0, 0), -1)
        cv2.circle(template, (fourth[0], fourth[1]), 5, (255, 255, 0), -1)
        cv2.imshow("tem", template)
        cv2.imshow("rec", dst)
        cv2.imwrite("digit.jpg", dst)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res
