import cv2
import numpy


# img 二值化图像
def projectionimg(img, info):
    # resized = cv2.resize(img, (3*width,3*height), interpolation=cv2.INTER_CUBIC)
    # 二值化
    # (_, thresh) = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', thresh)
    # 扩大白色面积，使效果更明显
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))#形态学处理，定义矩形结构
    closed = cv2.dilate(img, None, iterations=5)
    cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
    cv2.imshow('dilate', closed)
    cv2.waitKey(0)
    height, width = closed.shape[:2]
    v = [0] * width
    z = [0] * height
    a = 0
    # 垂直投影
    # 统计并存储每一列的黑点数
    for x in range(width):
        for y in range(height):
            if closed[y, x] != 0:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    # 创建空白图片，绘制垂直投影图
    emptyImage = numpy.zeros((height, width, 3), numpy.uint8)
    for x in range(0, width):
        for y in range(0, v[x]):
            b = (255, 255, 255)
            emptyImage[y, x] = b
    cv2.namedWindow("chuizhi", cv2.WINDOW_NORMAL)
    cv2.imshow('chuizhi', emptyImage)
    cv2.waitKey(0)
    # 水平投影
    # 统计每一行的黑点数
    a = 0
    emptyImage1 = numpy.zeros((height, width, 3), numpy.uint8)
    for y in range(0, height):
        for x in range(0, width):
            if closed[y, x] != 0:
                a = a + 1
            else:
                continue
        z[y] = a
        a = 0
    # 绘制水平投影图
    for y in range(0, height):
        for x in range(0, z[y]):
            b = (255, 255, 255)
            emptyImage1[y, x] = b
    cv2.namedWindow("shuipin", cv2.WINDOW_NORMAL)
    cv2.imshow('shuipin', emptyImage1)
    cv2.waitKey(0)
