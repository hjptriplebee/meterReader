import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn import svm

from algorithm.Common import meterFinderBySIFT

def videoDigit(video, info):
    """
    :param video: video
    :param info: info
    :return: 
    """
    result = {}

    start = ([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = ([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = ([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    width = info["rectangle"]["width"]
    height = info["rectangle"]["height"]
    widthSplit = info["widthSplit"]
    heightSplit = info["heightSplit"]
    characSplit = info["characSplit"]
    characSplit = info["characSplit"]

    # 计算数字表的矩形外框，并且拉直矫正
    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # 加载模型
    clf = joblib.load("../algorithm/OCR/svm/train_model.m")

    pictures = getPictures(video)  # 获得视频的帧，有少量重复帧
    for i, frame in enumerate(pictures):
        template = meterFinderBySIFT(frame, info)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.equalizeHist(template)
        dst = cv2.warpPerspective(template, M, (width, height))
        #
        # imgType = dst[characSplit[1][0]:characSplit[1][1], characSplit[0][0]:characSplit[0][1]]
        # imgType = cv2.adaptiveThreshold(imgType, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 11)

        # print(imgType.shape)
        cv2.imshow("type", template)
        cv2.waitKey(0)
        # nx, ny = imgType.shape
        # imgType = imgType.reshape((1, nx*ny))
        # type = clf.predict(imgType)
        # print(type)

    return result


def getPictures(videoCapture):
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    #fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量
    pictures = []
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        if i % int(fps):
            continue
        pictures.append(frame)
    videoCapture.release()
    return pictures
