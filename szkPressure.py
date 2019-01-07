import cv2
from Common import *
import numpy as np


def EuclideanDistance(pt1, pt2):
    """
    计算欧式距离
    """
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)


def getPoints(center, radious, angle):
    """
    返回的圆心与指定角度的圆周上一点连线上的所有点的坐标
    :param center: 圆心
    :param radious: 半径
    :param angle: 角度
    :return:
    """
    res = []

    # 找到圆周上的那个点
    farthestPointX = int(center[0]+radious*np.cos(angle/180*np.pi))
    farthestPointY = int(center[1]+radious*np.sin(angle/180*np.pi))

    # 计算直线方程
    delta_y = farthestPointX-center[0]
    delta_y = delta_y if delta_y != 0 else delta_y+1
    k = (farthestPointY-center[1]) / delta_y
    b = center[1] - k*center[0]

    # 将所有满足要求的点加入集合中
    for x in range(min(farthestPointX, center[0]), max(farthestPointX, center[0])):
        for y in range(min(farthestPointY, center[1]), max(farthestPointY, center[1])):
            if k*x+b-2 <= y <= k*x+b+2:
                res.append([x, y])

    return res, [farthestPointX, farthestPointY]


def myPressure(image, info):
    """
    基于圆心扫描线的指针检测
    :param image: 图片
    :param info: 标注信息
    :return: 读数
    """
    meter = meterFinderByTemplate(image, info["template"])

    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]]).astype(np.int16)
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]]).astype(np.int16)
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]]).astype(np.int16)

    # 控制图片大小
    if meter.shape[0] > 500:
        fixHeight = 300
        fixWidth = int(meter.shape[1] / meter.shape[0] * fixHeight)
        resizeCoffX = fixWidth / meter.shape[1]
        meter = cv2.resize(meter, (fixWidth, fixHeight))

        start = (start * resizeCoffX).astype(np.int16)
        end = (end * resizeCoffX).astype(np.int16)
        center = (center * resizeCoffX).astype(np.int16)

    # 计算表盘半径
    radious = int(EuclideanDistance(start, center))

    # 表面预处理，得到边缘检测后的二值图
    src = cv2.GaussianBlur(meter, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    thresh = gray.copy()
    cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV, thresh)
    # thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    mask = np.zeros((src.shape[0], src.shape[1]), np.uint8)
    cv2.circle(mask, (center[0], center[1]), radious, (255, 255, 255), -1)
    thresh = cv2.bitwise_and(thresh, mask)
    cv2.circle(thresh, (center[0], center[1]), int(radious/2),  (0, 0, 0), -1)

    thresh = cv2.erode(thresh, np.ones((3, 3), np.uint8), 3)
    thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8))
    thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 计算起始点与终点的角度，即扫描范围
    startAngle = int(AngleFactory.calAngleClockwise(startPoint=np.array([center[0]+100, center[1]]), centerPoint=center, endPoint=start)*180/np.pi)
    endAngle = int(AngleFactory.calAngleClockwise(startPoint=np.array([center[0]+100, center[1]]), centerPoint=center, endPoint=end)*180/np.pi)

    if endAngle < startAngle:
        endAngle += 360

    # 统计每一个扫描角度中，扫描线中的白点的数量。
    # 白点最多的扫描线就是指针所在处
    maxSum = 0
    outerPoint = start
    for angle in range(startAngle, endAngle):
        pts, outPt = getPoints(center, radious, angle)
        thisSum = 0
        for pt in pts:
            if thresh[pt[1], pt[0]] != 0:
                thisSum += 1
        if thisSum > maxSum:
            maxSum = thisSum
            outerPoint = outPt

    # 防止与起点终点重合
    # 计算角度的函数还需要改一下
    if start[0] == outerPoint[0] and start[1] == outerPoint[1]:
        degree = info["startValue"]
    elif end[0] == outerPoint[0] and end[1] == outerPoint[1]:
        degree = info["totalValue"]
    else:
        degree = AngleFactory.calPointerValueByOuterPoint(start, end, center, outerPoint, info["startValue"], info["totalValue"])

    # print(degree, start, center, outerPoint)
    # cv2.circle(meter, (outerPoint[0], outerPoint[1]), 10, (0, 0, 255), -1)
    # cv2.imshow("test", meter)
    # cv2.waitKey(0)
    return degree