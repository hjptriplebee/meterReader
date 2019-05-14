import cv2
import numpy as np
import json
import Algorithm.others.template as tmp
from Algorithm.others.colordetect import meterFinderNoinfoBySIFT


def FindcolorByDistance(image, info):
    """
    识别含有红色高亮数字区域的模块
    :param image: 输入图片
    :param info:  标记信息
    :return:
    """
    x = info["xnum"]
    y = info["ynum"]
    xlen = image.shape[0]
    # xlen = ["xlen"]
    # ylen = info["ylen"]
    ylen = image.shape[1]
    # xw = info["xw"]
    # yh = info["yh"]
    # xbigin = info["xbigin"]
    # ybigin = info["ybigin"]
    xbigin = 0
    ybigin = 0
    xw = image.shape[0]
    yh = image.shape[1]
    result = []
    for i in range(x):
        for j in range(y):
            preImage = image[xbigin + xlen * i:xbigin + xlen * i + xw, ybigin + ylen * j:ybigin + ylen * j + yh]
            # cv2.namedWindow("preImage", cv2.WINDOW_NORMAL)
            # cv2.imshow("preImage", preImage)
            # cv2.waitKey(0)
            ImageKey = np.sum(preImage)
            if ImageKey > xlen * ybigin * 0.02 * 255:
                result.append(1)
            else:
                result.append(0)
    return result


def getPictures(videoCapture):
    pictures = []
    cnt = 0
    skipFrameNum = 20
    while True:
        ret, frame = videoCapture.read()
        # print(cnt, np.shape(frame))
        cnt += 1
        if frame is None:
            break
        if cnt % skipFrameNum:
            continue
        pictures.append(frame)

    videoCapture.release()
    return pictures


def Cutimg(img):
    x, y = img.shape[:2]
    imageone = img[:, 0:int(y / 2)]
    imagetwo = img[:, int(y / 2):y]
    return imageone, imagetwo


def Filter(img):
    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, 2, 4, 2, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, -1, -1, -1, -1]])
    dst = cv2.filter2D(img, -1, kernel_5x5)
    return dst


def Orproccessing(a, b):
    result = a | b
    return result


def PreProcessing(img, info):
    # 图像预处理
    # imgtemp = cv2.imread("E:/picture/20190120/Calibration/template/light9.jpg")
    # imgsift = meterFinderNoinfoB
    bgray = info["bMoreLight"]
    if bgray == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        ret, thresh = cv2.threshold(gray, gray.max() - 25, gray.max(), cv2.THRESH_BINARY)
    else:
        # green_min = np.array([37, 170, 100])  # 电源灯
        # green_max = np.array([52, 255, 255])  # 电源灯
        # green_min = np.array([125, 115, 230]) # 液晶屏
        # green_max = np.array([139, 210, 255]) # 液晶屏
        # green_min = np.array([26, 7, 200])
        # green_max = np.array([35, 20, 254])
        green_min = np.array(info["color_min"])
        green_max = np.array(info["color_max"])
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(imgHSV, green_min, green_max)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh = cv2.erode(thresh, kernel, iterations=3)
    # thresh = cv2.dilate(thresh, kernel, iterations=3)
    # cv2.namedWindow("imgsift", cv2.WINDOW_NORMAL)
    # cv2.imshow("imgsift", thresh)
    # cv2.waitKey(0)
    result = FindcolorByDistance(thresh, info)
    # print(result)
    return result


def indicatorimg(img, info):
    # file = open("../config/2-2_1" + ".json")
    # img1,img2 = Cutimg(img)
    # img1 = Filter(img1)
    # tempimg = cv2.imread("E:/picture/20190120/Calibration/template/light11.jpg")
    # tempimg = cv2.imread("C:/Users/crow/Desktop/picturesource/template/light5.jpg")
    tempimg = info["template"]
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", tempimg)
    # cv2.waitKey(0)
    siftimg1 = meterFinderNoinfoBySIFT(img, tempimg)
    # siftimg2 = meterFinderNoinfoBySIFT(img2,tempimg)
    # img = img[1050:1199,:]
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", siftimg1)
    # cv2.waitKey(0)
    result = PreProcessing(siftimg1, info)
    # resultimg2 = PreProcessing(siftimg2, info)
    # result = Orproccessing(np.array(resultimg1),np.array(resultimg2))
    return result


def cabinet_indicator(img, info):
    num = len(img[0])
    result = np.zeros(num)
    for frame in img:
        oneimg = indicatorimg(frame, info)
        result = Orproccessing(np.array(oneimg), np.array(result))
    return result


if __name__ == '__main__':
    # img1 = cv2.imread("C:/Users/crow/Desktop/picturesource/tp23.jpg")
    # img2 = cv2.imread("C:/Users/crow/Desktop/picturesource/tp24.jpg")
    img = cv2.imread("C:/Users/crow/Desktop/0413/7.jpg")
    # file = open("../config/29-1_1" + ".json")
    file = open("../config/29-1_1" + ".json")
    data = json.load(file)
    indicatorimg(img, data)
    # recognitionData = cv2.VideoCapture("E:/picture/out.avi")
    # pictures = Screenshot(recognitionData)
    # for frame in pictures:
    #     img90 = np.rot90(frame)
    #     imgone,imatwo = Cutimg(img90)
    #     cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #     cv2.imshow("img", img90)
    #     cv2.waitKey(0)
