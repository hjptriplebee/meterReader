import cv2
import sys
import os
import json
import shutil
import numpy as np
import torch
from collections import defaultdict, Counter
import copy

from Algorithm.pressure.digitPressure import digitPressure
from Algorithm.utils.Finder import meterFinderBySIFT, meterReginAndLocationBySIFT
# from Algorithm.utils.Finder import meterFinderBySIFT, meterReginAndLocationBySIFT
from Algorithm.OCR.character.characterNet import characterNet


def videoDigit(video, info):
    """
    :param video: VideoCapture Input
    :param info: info
    :return:
    """
    net = characterNet()
    net.load_state_dict(torch.load("Algorithm/OCR/character/NewNet_minLoss_model_0.965.pkl"))
    pictures = getPictures(video)  # 获得视频的帧，有少量重复帧

    def emptyLsit():
        return []

    imagesDict = defaultdict(emptyLsit)
    box = None
    newinfo = None
    newchange = None
    saveframe = []
    for i, frame in enumerate(pictures):
        res = digitPressure(frame, copy.deepcopy(info))
        eachinfo = copy.deepcopy(info)
        template, change, bbox = meterReginAndLocationBySIFT(frame, eachinfo)
        index, charimg = checkFrame(i, net, template, eachinfo)
        if index < 4 and box is None:
            box = bbox.copy()
            newchange = change.copy()
            newinfo = copy.deepcopy(eachinfo)
        elif index == 4 and box is None:
            saveframe.append(i)
        if index < 4:
            imagesDict[chr(index+ord('A'))] += [res]
    if box is not None:
        for index in saveframe:
            res = digitPressure(frame, copy.deepcopy(info))
            frame = pictures[index]
            src_correct = cv2.warpAffine(frame, newchange, (frame.shape[1], frame.shape[0]))
            newtemplate = src_correct[box[0]:box[1], box[2]:box[3]]
            index, charimg = checkFrame(i, net, newtemplate, newinfo)
            if index < 4:
                imagesDict[chr(index + ord('A'))] += [res]
        # # debug
        # title = str(chr(index+ord('A'))) + str(res)
        # frame = cv2.putText(frame, title, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        # cv2.imwrite(os.path.join(resultdir, info['name']+str(i)+'.jpg'), frame)
        # cv2.imshow(title, frame)
        # cv2.waitKey(0)
    imagesDict = getResult(imagesDict)
    return imagesDict

def getResult(dicts):
    newdicts = {'A': [], 'B': [], 'C': [], 'D': []}
    for ctype, res in dicts.items():
        print("res ", res)
        firsts = [[c for c in str(x[0])] for x in res]
        seconds = [[c for c in str(x[1])] for x in res]
        print("firsts ", firsts)
        print("seconds ", seconds)
        for x in firsts:
            if len(x) < 6:
                x.insert(0, ' ')
        for x in seconds:
            if len(x) < 6:
                x.insert(0, ' ')
        if len(firsts[0]) > 0 and len(seconds[0]) > 0:
            number = ""
            for j in range(6):
                words = [a[j] for a in firsts]
                num = Counter(words).most_common(1)
                number = number + num[0][0]
            newdicts[ctype].append(number)
            number = ""
            for j in range(6):
                words = [a[j] for a in seconds]
                num = Counter(words).most_common(1)
                number = number + num[0][0]
            newdicts[ctype].append(number)
    return newdicts

def getPictures(videoCapture):
    """
    截取视频中每隔固定时间出现的帧
    :param videoCapture: 输入视频
    :return: 截取的帧片段
    """
    pictures = []
    cnt = 0
    skipFrameNum = 15
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

def checkFrame(count, net, template, info):
    """
    判断图片的类型A，AB等
    :param image:  image
    :param info: info
    :return: 出现次序0.1.2.3
    """
    start = [info["startPoint"]["x"], info["startPoint"]["y"]]
    end = [info["endPoint"]["x"], info["endPoint"]["y"]]
    center = [info["centerPoint"]["x"], info["centerPoint"]["y"]]
    width = info["rectangle"]["width"]
    height = info["rectangle"]["height"]
    characSplit = info["characSplit"]

    # 计算数字表的矩形外框，并且拉直矫正
    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # oritemplate = template.copy()
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.equalizeHist(template)
    dst = cv2.warpPerspective(template, M, (width, height))

    imgType = dst[characSplit[1][0]:characSplit[1][1], characSplit[0][0]:characSplit[0][1]]
    imgType = cv2.adaptiveThreshold(imgType, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 11)
    imgType = cv2.bitwise_not(imgType)

    orimage = imgType.copy()

    imgType = torch.Tensor(np.array(imgType, dtype=np.float32))
    imgType = torch.unsqueeze(imgType, 0)
    imgType = torch.unsqueeze(imgType, 0)
    type_probe = net.forward(imgType)
    type_probe = type_probe.detach().numpy()
    maxIndex = np.argmax(type_probe)

    # debug
    # oritemplate = drawTemplatePoints(template, info)
    # dst = drawDstPoints(dst, info)
    # cv2.imshow("M", M)
    # cv2.imshow("template ", oritemplate)
    # cv2.imshow("des ", dst)
    # cv2.imshow(str(chr(maxIndex+ord('A'))), orimage)
    # kind = str(chr(maxIndex+ord('A')))
    # cv2.imwrite(os.path.join(resultdir, info['name'] + '_' + str(count) + '_' + kind + '.bmp'), orimage)
    # cv2.waitKey(0)

    return maxIndex, orimage

def drawTemplatePoints(template, info):
    start = (info["startPoint"]["x"], info["startPoint"]["y"])
    end = (info["endPoint"]["x"], info["endPoint"]["y"])
    center = (info["centerPoint"]["x"], info["centerPoint"]["y"])
    cv2.circle(template, start, 2, (255, 0, 0), 3)
    cv2.circle(template, end, 2, (255, 0, 0), 3)
    cv2.circle(template, center, 2, (255, 0, 0), 3)
    return template

def drawDstPoints(dst, info):
    widthSplit = info["widthSplit"]
    heightSplit = info["heightSplit"]
    characSplit = info["characSplit"]
    pts = []
    pts.append((characSplit[0][0], characSplit[1][0]))
    pts.append((characSplit[0][0], characSplit[1][1]))
    pts.append((characSplit[0][1], characSplit[1][0]))
    pts.append((characSplit[0][1], characSplit[1][1]))
    for x in widthSplit[0]:
        for j in range(len(heightSplit)):
            for y in heightSplit[j]:
                pts.append((x, y))
    for point in pts:
        cv2.circle(dst, point, 1, (255, 0, 0), 3)
    return dst


# resultdir = "/Users/yuanyuan/Documents/GitHub/meterReader/result_character"
# if os.path.exists(resultdir):
#     shutil.rmtree(resultdir)
# if not os.path.exists(resultdir):
#     os.makedirs(resultdir)
