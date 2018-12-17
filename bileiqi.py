import numpy as np
import cv2 as cv
import math
def bileiqi(ROI, info):
    '''
    {
      "distance": 10.0,
      "horizontal": 10.0,
      "vertical": 20.0,
      "name": "1_1",
      "type": "bileiqi",
      "ROI": {
          "x": 200,
          "y": 200,
          "w": 1520,
          "h": 680
      },
      "startPoint": {
          "x": -1,
          "y": -1
      },
      "endPoint": {
          "x": -1,
          "y": -1
      },
      "centerPoint": {
          "x": -1,
          "y": -1
      },
      "startValue": 0.0,
      "totalValue": 2.0
}
    :param image:
    :param info:
    :return:
    '''
    startPoint=(info["startPoint"]["x"],info["startPoint"]["y"])
    endPoint=(info["endPoint"]["x"],info["endPoint"]["y"])

def levelAngle(vector):     #求角度
    cos=(vector[0]*(200.0))/(200.0*math.sqrt(vector[0]**2+vector[1]**2))
    return math.acos(cos)*180.0/math.pi

def line_detect(gray):
    edges = cv.Canny(gray, 70,100, apertureSize=3)
    cv.imshow("wewq",edges)
    linepoint=[]
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=20, maxLineGap=1)
    for line in lines:
        linepoint.append(line[0])
    return linepoint
ROI=cv.imread("template.png")
src_gray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
src_gray = cv.equalizeHist(src_gray)
cv.imshow("wqeq",src_gray)
linepoint = line_detect(src_gray)  # 检测到 的直线变成了一维数组
for line in linepoint:
    x1, y1, x2, y2 = line
    cv.line(ROI, (x1, y1), (x2, y2), (0, 255, 255), 1)
    k1 = (y2 - y1) / ((x2 - x1) + 0.000001)
    b1 = y1 - k1 * x1
    x = (ROI.shape[0] - b1) / (k1 + 0.0000001)
    if (levelAngle((x2 - x1, y2 - y1)) > 40 and levelAngle((x2 - x1, y2 - y1)) < 140 and x < (
            3 * ROI.shape[1] / 5) and x > (ROI.shape[1] / 2)):
        cv.line(ROI, (x1, y1), (x2, y2), (0, 255, 255), 1)
        print("%f度" % levelAngle((x2 - x1, y2 - y1)))
print(23)
cv.imshow("we",ROI)
cv.waitKey()

