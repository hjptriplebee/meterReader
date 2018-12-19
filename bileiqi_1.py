import math
import numpy as np
import cv2 as cv
from keras.models import load_model
def levelAngle(x1,y1,x2,y2):  #求角度 指针向量的两个点
    """

    :param x1: point1_x
    :param y1: point1_y
    :param x2: point2_x
    :param y2: point2_x
    :return: angle
    """
    if (y1 < y2):
        vector=(x1 - x2, y1 - y2)
    else:
        vector = (x2 - x1, y2 - y1)
    cos=(vector[0]*(200.0))/(200.0*math.sqrt(vector[0]**2+vector[1]**2))
    return 180-math.acos(cos)*180.0/math.pi
def line_detect(gray,low_thres=40,high_thres=110):
    """

    :param gray: graph
    :param low_thres: Canny边缘检测的低阈值
    :param high_thres: Canny边缘检测的高阈值
    :return: two points of line
    """
    edges = cv.Canny(gray, low_thres, high_thres, apertureSize=3)
    #cv.imshow("etr",edges)
    linepoint=[]
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=25, maxLineGap=2)

    for line in lines:
        linepoint.append(line[0])

    return linepoint
def bileiqi_1(src,info):
    """

    :param src: ROI
    :return:value
    """
    low_thres=40     #Canny检测的高低阈值
    high_thres=100
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.equalizeHist(src_gray)
    # 图像的高和宽
    # h = src.shape[0]
    # w = src.shape[1]
    thres, src_bin = cv.threshold(src_gray, 100, 255, cv.THRESH_BINARY)
    cv.imshow("az", src_bin)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    # 闭操作
    src_close = cv.morphologyEx(src_bin, cv.MORPH_CLOSE, kernel)
    image, contours, h = cv.findContours(src_close, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        # cv.drawContours(src,contours,i,(0,255,0),1)
        if (rect[1][0] > (src.shape[1] * 0.5) and rect[1][1] > (src.shape[0] * 0.3) and rect[1][0] < (
                src.shape[1] * 0.8) and rect[1][1] < (src.shape[0] * 0.5)):
            minrect = cv.minAreaRect(contours[i])
            # cv.drawContours(src,contours,i,(0,0,255),1)
            points = cv.boxPoints(minrect)  # 矩形四个点
    # cv.imshow("wq",src)
    #
    #

    # 获得变换角度
    angle = minrect[2]

    # 获得变换矩阵
    change = cv.getRotationMatrix2D(minrect[0], angle, 1)
    src_correct = cv.warpAffine(src, change, (src.shape[1], src.shape[0]))
    src_gray = cv.warpAffine(src_gray, change, (src.shape[1], src.shape[0]))
    src_bin = cv.warpAffine(src_bin, change, (src.shape[1], src.shape[0]))
    # cv.imshow("src_correct",src_correct)

    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # 根据变换矩阵得到新的四个点
    newpoints = []
    for i in range(4):
        point = newchange.dot(np.array([points[i][0], points[i][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)
    src_gray = src_gray[int(newpoints[1][1]) + 10:int(newpoints[0][1] - 10),
               int(newpoints[1][0]) + 10:int(newpoints[2][0]) - 10]

    while (True):
        linepoint = line_detect(src_gray, low_thres, high_thres)  # 变成了一维数组
        Aline = []
        for line in linepoint:
            x1, y1, x2, y2 = line
            # cv.line(src_gray, (x1, y1), (x2, y2), 0,3)
            k1 = (y2 - y1) / ((x2 - x1) + 0.000001)
            b1 = y1 - k1 * x1
            x = (src_gray.shape[0] - b1) / (k1 + 0.0000001)
            if (levelAngle(x1, y1, x2, y2) > 30 and levelAngle(x1, y1, x2, y2) < 150 and x < (
                    src_gray.shape[1] / 1.5) and x > (src_gray.shape[1] / 2.5)):
                Aline.append(line)
                # print("%f度" % levelAngle(x1,y1,x2,y2))
        if (len(Aline) == 1 or low_thres == 120):
            # print("%f度" % levelAngle(Aline[0][0], Aline[0][1], Aline[0][2], Aline[0][3]))
            angle=levelAngle(Aline[0][0], Aline[0][1], Aline[0][2], Aline[0][3])
            #cv.line(src_gray, (Aline[0][0], Aline[0][1]), (Aline[0][2], Aline[0][3]), 0, 2)
            break
        else:
            low_thres += 1
    # 指针检测及角度计算完成



    # 数字的读取
    src = src[0:src.shape[0] // 2, :]
    cv.imshow("swwwwd",src)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.equalizeHist(src_gray)
    # cv.imshow("asd",src_gray)
    thres, src_bin = cv.threshold(src_gray, 100, 255, cv.THRESH_BINARY)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # #
    # src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, kernel)
    # src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, kernel)
    cv.imshow("saaq",src_bin)
    image, contours, h = cv.findContours(src_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        # cv.drawContours(src,contours,i,(0,255,0),1)
        if (rect[1][0] > (src.shape[1] *0.2) and rect[1][1] > (src.shape[0] *0.2) and rect[1][0] < (src.shape[1] *0.3) and
                rect[1][1] < (src.shape[0] *0.4)):
            minrect = cv.minAreaRect(contours[i])
            cv.drawContours(src, contours, i, (255, 0, 0), 1)
            points = cv.boxPoints(minrect)  # 矩形四个点
    # cv.imshow("sadwq",src)

    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # 根据变换矩阵得到新的四个点
    newpoints = []
    for i in range(4):
        point = newchange.dot(np.array([points[i][0], points[i][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)

    src_bin = cv.warpAffine(src_bin, change, (src.shape[1], src.shape[0]))
    #cv.imshow("sazz", src_bin)
    # # for i in range(4):
    # #     cv.circle(src_correct,(int(newpoints[i][0]),int(newpoints[i][1])),2,(0,22,255),-1)
    # #     print(int(newpoints[i][0]),int(newpoints[i][1]))
    #
    src_num = src_bin[int(newpoints[1][1]) + 5:int(newpoints[0][1] - 5),
              int(newpoints[1][0]) + 10:int(newpoints[2][0]) - 5]
    # cv.imshow("num",src_num)
    # src_num=cv.cvtColor(src_num,cv.COLOR_BGR2GRAY)
    # thres,src_num=cv.threshold(src_num,60,255,cv.THRESH_BINARY)
    # cv.imshow("dsaaaa",src_num)
    # 开操作
    #src_num_open = cv.morphologyEx(src_num, cv.MORPH_OPEN, kernel)
    # 进行数字分割
    _,src_num_open=cv.threshold(src_num,50,255,cv.THRESH_BINARY)
    #cv.imshow("wqeq",src_num_open)
    statistic = np.zeros((1, src_num_open.shape[1]), dtype=np.int32)
    for i in range(src_num_open.shape[1]):
        for j in range(src_num_open.shape[0]):
            if (src_num_open[j][i] == 255):
                statistic[0][i] += 1

    iszero = True

    cutpoint = []
    for i in range(statistic.shape[1]):
        if (iszero):
            if (statistic[0][i] != 0):
                cutpoint.append(i - 5)
                iszero = False
        else:
            if (statistic[0][i] == 0):
                cutpoint.append(i + 5)
                iszero = True
    model = load_model('my_model.h5')
    Num=''
    for i in range(2):
        # cv.line(src_num_open,(cutpoint[2*i],0),(cutpoint[2*i],src_num_open.shape[0]),(255),2)
        # cv.line(src_num_open,(cutpoint[2*i+1],0),(cutpoint[2*i+1],src_num_open.shape[0]),(255),2)
        number = src_num_open[:, i * src_num_open.shape[1] // 2:(i + 1) * src_num_open.shape[1] // 2]
        cv.imshow("sd%d" % i, number)
        number = cv.resize(number, (28, 28))

        number = np.expand_dims(number, axis=0)
        number = np.expand_dims(number, axis=3)
        result = model.predict(number)
        max = -1
        num = 0

        for i in range(len(result[0])):
            if (max < result[0][i]):
                max = result[0][i]
                num = i
        Num+=repr(num)
    # cv.imshow("src_num", src_num_open)
    # cv.imshow("result", src_correct)
    cv.waitKey(1)
    return angle,Num


