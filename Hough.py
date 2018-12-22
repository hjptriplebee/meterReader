import numpy as np
import cv2
import math
import time
from matplotlib import pyplot as plt

# 构建测试图片
# img = cv2.imread('demo.png')
img = cv2.imread('image/SF6/IMG_7586.JPG')
# cv2.imshow("origin", img)
startTime = time.clock()

img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
# plt.figure(figsize=(img.shape[0] * 0.2 / 100, img.shape[1] * 0.2 / 100))
plt.figure(figsize=(20, 20))
# cv2.imshow('source', img)
#
blurred = cv2.GaussianBlur(img, (3, 3), 0)
# cv2.imshow('blurred', blurred？)

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

edges = cv2.Canny(gray, 50, 150)
# cv2.imshow('edge', edges)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 600, 100, 30, 10, 20, 100)
circles = sorted(circles[0], key=lambda c: c[2], reverse=True)

for c in circles:
    cv2.circle(img, (c[0], c[1]), c[2], 255, 2)
# # cv2.imshow("circle", gray)
# cv2.waitKey(0)

centerX = int(circles[0][0])
centerY = int(circles[0][1])
ROrigin = int(circles[0][2])
ROI = img[centerY - ROrigin: centerY + ROrigin, centerX - ROrigin: centerX + ROrigin]

fixHeight = 300
print(ROI.shape)
ratio = fixHeight / ROI.shape[1]
R = int(ROrigin * ratio)
ROI = cv2.resize(ROI, (int(ROI.shape[0] / ROI.shape[1] * fixHeight), fixHeight))

# cv2.illuminationChange(ROI, cv2.bitwise_not(np.zeros_like(ROI)), ROI, 0.1, 0.1)

# 为什么需要转换
ROIYUV = cv2.cvtColor(ROI, cv2.COLOR_BGR2YUV)
ROIYUV[:, :, 0] = cv2.equalizeHist(ROIYUV[:, :, 0])
ROI = cv2.cvtColor(ROIYUV, cv2.COLOR_YUV2BGR)
# 创建全白的遮罩
ROIMask = cv2.bitwise_not(np.zeros(ROI.shape, dtype="uint8"))
tableEdge = 10

# 画表盘的黑圆
cv2.circle(ROIMask, (R, R), R - tableEdge, (0, 0, 0), -1)

#  圆外背景被覆盖为白色
ROI = cv2.bitwise_or(ROI, ROIMask)
# ROIGray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
# ROIEdges = cv2.Canny(ROIGray, 20, 150)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# ROIEdges = cv2.dilate(ROIEdges, kernel)
# cv2.imshow("tem", ROIEdges)
# lines = cv2.HoughLinesP(ROIEdges, 1, np.pi/180, 20, 70, 50)
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(ROI,(x1,y1),(x2,y2),(0,0,255),2)

lowerBlack = np.array([0, 0, 0], dtype="uint8")
upperBlack = np.array([180, 255, 46], dtype="uint8")
lowerGreen = np.array([35, 53, 46], dtype="uint8")
upperGreen = np.array([99, 255, 255], dtype="uint8")
ROIHSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
plt.subplot(325)
plt.imshow(ROIHSV), plt.title("ROIHSV")
# ???范围大小如何确定
# 得到表中黑色像素聚集黑圆弧
blackMask = cv2.inRange(ROIHSV, lowerBlack, upperBlack)
greenMask = cv2.inRange(ROIHSV, lowerGreen, upperGreen)
plt.subplot(326)
plt.imshow(blackMask), plt.title("BeforeMask")
blackROIMask = np.zeros(blackMask.shape, dtype="uint8")
maskEdge = 40
#  圆为白，背景为黑的遮罩
cv2.circle(blackROIMask, (R, R), R - maskEdge, 255, -1)
cv2.imshow("BlackROIMask", blackROIMask)
# 去除黑圈外围噪点
blackMask = cv2.bitwise_and(blackMask, blackROIMask)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 去除细小噪点(比结构元小的噪点像素都会被腐蚀)
blackMask = cv2.erode(blackMask, kernel)
blackMask = cv2.dilate(blackMask, kernel)

# blackMask = cv2.dilate(blackMask, kernel)
greenMask = cv2.erode(greenMask, kernel)
greenMask = cv2.dilate(greenMask, kernel)
# greenMask = cv2.dilate(greenMask, kernel)

_, greenContours, greenHierarchy = cv2.findContours(greenMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# ??????
greenContours = sorted(greenContours, key=lambda c: c.shape[0], reverse=True)
# ?????????
greenContours = [c for c in greenContours if len(c) > 30]
cv2.imshow("greenMask", greenMask)
plt.subplot(321)
plt.imshow(greenMask), plt.title("GreenMask")

maxY = 0
endPoint = []

# 找最下的绿色点
for c in greenContours:
    for p in c:
        if p[0][1] > maxY:
            maxY = p[0][1]
            endPoint = p[0]

cv2.line(ROI, (R, R), (endPoint[0], endPoint[1]), (0, 255, 0), 2)
cv2.line(img, (centerX, centerY),
         (int(endPoint[0] / ratio) + centerX - ROrigin, int(endPoint[1] / ratio) + centerY - ROrigin),
         (0, 255, 0), 2)

_2, blackContours, blackHierarchy = cv2.findContours(blackMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
blackContours = sorted(blackContours, key=lambda c: c.shape[0], reverse=True)
blackContours = [c for c in blackContours if len(c) > 50]
cv2.imshow("blackMask", blackMask)
plt.subplot(322)
plt.imshow(blackMask), plt.title("BlackMask")

pointDis = 0
point = []

for c in blackContours:
    for p in c:
        dis = (p[0][0] - R) * (p[0][0] - R) + (p[0][1] - R) * (p[0][1] - R)
        if dis > pointDis:
            pointDis = dis
            point = p[0]

#？？
cv2.line(ROI, (R, R), (point[0], point[1]), (0, 255, 0), 2)
cv2.line(img, (centerX, centerY),
         (int(point[0] / ratio) + centerX - ROrigin, int(point[1] / ratio) + centerY - ROrigin),
         (0, 255, 0), 2)


def calCounterClockWiseAngleWithXPositiveAxis(origin, point):
    dis = math.sqrt((origin[0] - point[0]) * (origin[0] - point[0]) + (origin[1] - point[1]) * (origin[1] - point[1]))
    angle = math.asin((origin[1] - point[1]) / dis)
    if point[0] < origin[0]:
        angle = angle / abs(angle) * (math.pi) - angle
    if angle < 0:
        angle = angle + 2 * math.pi
    return angle


angleEndPoint = calCounterClockWiseAngleWithXPositiveAxis((R, R), endPoint)
print(angleEndPoint / 2 / math.pi * 360)
anglePoint = calCounterClockWiseAngleWithXPositiveAxis((R, R), point)
print(anglePoint / 2 / math.pi * 360)
res = 0.9 - (anglePoint + 2 * math.pi - angleEndPoint) / (1.5 * math.pi)

endTime = time.clock()

print("value: %.3f" % res)
print("time: %.3fs" % (endTime - startTime))

cv2.imshow("ROI", ROI)
plt.subplot(323)
plt.imshow(ROI), plt.title("ROI")
cv2.imshow("img", img)
plt.subplot(324)
plt.imshow(img), plt.title("IMG")
plt.show()
cv2.waitKey(0)
