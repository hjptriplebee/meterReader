import cv2
import math
import numpy as np
import json
import Algorithm.others.template as tmp
from sklearn.metrics.pairwise import pairwise_distances


def greenlight(img):
    green_min = np.array([20, 0, 50])
    green_max = np.array([50, 15, 155])
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, green_min, green_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=5)
    return mask


# def detect_circles(img):
# 	dst = cv2.pyrMeanShiftFiltering(img, 10, 100)
# 	cimg = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
# 	circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=80, minRadius=5, maxRadius=0)
# 	circles = np.uint16(np.around(circles))
# 	for i in circles[0, :]:
# 		cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
# 		cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 2)
# 	cv2.namedWindow("HoughCircles", cv2.WINDOW_NORMAL)
# 	cv2.imshow('HoughCircles', img)
# 	return circles

def findcenter(img):
    TmpImage, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = np.zeros((len(contours), 2))
    num = 0
    for i in contours:
        M = cv2.moments(i)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # print(cX, cY)
        center[num] = [cX, cY]
        # print(center[num])
        # print("cX = %d,cY = %d" %(cX,cY))
        num += 1
        # cv2.drawContours(img, contours,-1, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)
    # cv2.namedWindow("img1", cv2.WINDOW_NORMAL);
    # cv2.imshow('img1', img)
    # cv2.waitKey(0)
    return center.tolist()


def comcenter(center, getcenter):
    if len(center) < len(getcenter):
        return np.zeros(1)
    result = np.zeros(len(center) - len(getcenter))
    n = 0
    for i in range(len(center)):
        mark = True
        for j in range(len(getcenter)):
            dx = abs(center[i][0] - getcenter[j][0])
            dy = abs(center[i][1] - getcenter[j][1])
            if dx + dy < 20:
                mark = False
                break
        if mark:
            try:
                result[n] = i + 1
                n += 1
            except IndexError as ind:
                print(ind)
                break
    return result


def branchlight(center):
    index = []
    numline = 0
    if len(center) == 0:
        return 0
    num = center[0][1]
    # print(center)
    for i in center:
        if i[1] > (num + 100) or i[1] < (num - 100):
            index.append(numline)
        numline += 1
        num = i[1]
    index.append(numline)
    return index


def centerout(brightcenter, darkcenter):
    # print(brightcenter)
    darkcenter.sort(key=lambda a: a[1])
    darkcenter = np.array(darkcenter)
    brightcenter.sort(key=lambda a: a[1])
    brightcenter = np.array(brightcenter)
    # darkcenter.sort(key=lambda a:a[0])
    # brightcenter.sort(key=lambda a:a[0])
    # print(brightcenter)
    darkindex = branchlight(darkcenter)
    brightindex = branchlight(brightcenter)
    # print(darkindex,brightindex)
    # if len(brightindex) > len(darkindex):
    # 	maxindex = brightindex
    # 	# maxcenter = brightcenter
    # 	# minindex = darkindex
    # 	# mincenter = darkcenter
    # else:
    # 	maxindex = darkindex
    # 	maxcenter = darkcenter
    # 	minindex = brightindex
    # 	mincenter = brightcenter
    # alllength = maxlength
    lightstate = brightlight(brightcenter)
    linenum = 1
    baddline = True
    # 计算行
    for i in range(len(darkindex)):
        for j in range(len(brightindex)):
            if abs(darkcenter[darkindex[i] - 1][1] - brightcenter[brightindex[j] - 1][1]) < 100:
                if j == 0:
                    bright = brightcenter[0:int(brightindex[j]), 0]
                else:
                    bright = brightcenter[int(brightindex[j - 1]):int(brightindex[j]), 0]
                if i == 0:
                    dark = darkcenter[0:int(darkindex[i]), 0]
                else:
                    dark = darkcenter[int(darkindex[i - 1]):int(darkindex[i]), 0]
                # bright = sorted(bright)
                # dark = sorted(dark)
                dotstate = linelightstate(bright, dark)
                # darknum = np.ones(len(bright)+len(dark))
                # print(dotstate)
                # lightstate.append([linenum,dotstate])
                # for i in dotstate:
                # 	darknum[int(i)] = 0
                lightstate[j] = dotstate
                break
            else:
                if darkcenter[darkindex[i] - 1][1] > brightcenter[brightindex[j] - 1][1] and baddline == True:
                    linenum += 1
                    baddline = False
                else:
                    baddline = True
                if i == 0:
                    num = darkindex[i]
                else:
                    num = darkindex[i] - darkindex[i - 1]
                darknum = np.zeros(num)
                # lightstate.append([linenum,('num = %d'%num)])
                lightstate[linenum:linenum] = [darknum.tolist()]
            linenum += 1
        # alllength += 1
    return lightstate


# 求出某一行灯上哪个位置灯的状态
def linelightstate(max, min):
    # for i in range(len(maxindex)):
    # 	if i == 0:
    # 		Bright = maxcenter[0:int(maxindex[i]),0]
    # 		#print(Bright)
    # 	else:
    # 		Bright = maxcenter[int(maxindex[i-1]):int(maxindex[i]),0]
    # 		#print(Bright)
    # for i in range(len(minindex)):
    # 	if i == 0:
    # 		dark = mincenter[0:int(minindex[i]),0]
    # 	else:
    # 		dark = mincenter[int(minindex[i-1]):int(minindex[i]), 0]
    merge = [(m, 0) for m in min] + [(m, 1) for m in max]
    merge.sort(key=lambda a: a[0])
    # merge = [[m, 0] for m in min] + [[m, 1] for m in max]
    # sorted(merge,key=lambda a:a[0])
    return [x[1] for x in merge]


# 处理全部不亮的灯
def nobright(darkcenter):
    darkindex = branchlight(darkcenter)
    # print(darkindex)
    lightstate = []

    for i in range(len(darkindex)):
        if i == 0:
            num = darkindex[i]
        else:
            num = darkindex[i] - darkindex[i - 1]
        linenum = np.zeros(num, dtype=np.int)
        lightstate.append(linenum.tolist())
    return lightstate


# 初始化亮的灯
def brightlight(center):
    brightindex = branchlight(center)
    # print(brightindex)
    lightstate = []

    for i in range(len(brightindex)):
        if i == 0:
            num = brightindex[i]
        else:
            num = brightindex[i] - brightindex[i - 1]
        linenum = np.ones(num, dtype=np.int)
        lightstate.append(linenum.tolist())
    return lightstate


def redlight(img, info):
    red_min = np.array([0, 0, 254])
    red_max = np.array([100, 255, 255])
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, red_min, red_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=5)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=12, minRadius=5, maxRadius=0)
    circles = np.uint16(np.around(circles))
    ans = circles[0, :][:, 0:2]
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 10)
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 10)
    cv2.namedWindow("HoughCircles", cv2.WINDOW_NORMAL)
    cv2.imshow('HoughCircles', img)
    return ans.tolist()


def findlight(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, gray.max() - 25, gray.max(), cv2.THRESH_BINARY)
    red_min = np.array([0, 0, 254])
    red_max = np.array([100, 255, 255])
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, red_min, red_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=5)
    # cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    # cv2.imshow('thresh', mask)
    # cv2.waitKey(0)
    # circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=13, minRadius=5, maxRadius=0)
    # circles = np.uint16(np.around(circles))
    # ans = circles[0, :][:, 0:2]
    circles = findcenter(mask)
    # print(circles)
    # for i in circles:
    # 	#cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 10)
    # 	# print(i)
    # 	cv2.circle(image, (int(i[0]), int(i[1])), 2, (255, 0, 0), 10)
    # cv2.namedWindow("HoughCircles", cv2.WINDOW_NORMAL)
    # cv2.imshow('HoughCircles', image)
    # cv2.waitKey(0)
    return circles


def color(img, info):
    # 记录灯的坐标，有多少个灯，对比各个坐标记录差距大的
    # redimg = redlight(img,info)
    # getcenter = findcenter(redimg)
    # center = [[ 799,1035],[1053,1039],[1296. ,1042.],[1543., 1039.],[1773., 1046.],[2009., 1046.],[2232., 1053.],[2460., 1053.],
    #		  [2674., 1053.],[2894., 1042.],[3104., 1049.],[1152.,2242.],[1561., 2232.],[1951., 2214.],[2329., 2203.],[2706., 2199.]]
    # center = info['center']
    # result = comcenter(center,getcenter)
    # print(result)

    # darkmode = cv2.imread('../template/2-2_1.jpg')
    bMoreLight = info["bMoreLight"]
    if bMoreLight == 1:
        darkmode = cv2.imread("./info/20190420/template/template1-4.jpg")
    else:
        darkmode = cv2.imread("./info/20190420/template/template1-4.jpg")
    # cv2.namedWindow("darkmode", cv2.WINDOW_NORMAL)
    # cv2.imshow('darkmode', img)
    # cv2.waitKey(0)
    # darkmode = info["template"]
    # Brightmode = cv2.imread('E:/picture/20190117/light6.jpg')
    # print(np.shape(darkmode))
    try:
        darkimg = tmp.template(darkmode, img, 0.5)
    except:
        darkimg = darkmode
    red_min = np.array([0, 0, 254])
    red_max = np.array([0, 0, 255])
    darkmask = cv2.inRange(darkimg, red_min, red_max)
    darkcenter = findcenter(darkmask)
    if len(darkcenter) == 0:
        return 0
    # Brightimg = tmp.template(Brightmode, img, 0.5)
    # Brightmask = cv2.inRange(Brightimg, red_min, red_max)

    algr = info["bMoreLight"]  # 两种算法方式,选取
    if algr != 1:
        Brightcenter = redlight(img, info)
    else:
        Brightcenter = findlight(img)
    # print(Brightcenter)
    # Brightcenter = findcenter(Brightmask)
    if len(Brightcenter) == 0:
        result = nobright(darkcenter)
    else:
        result = centerout(Brightcenter, darkcenter)
    finish = []
    for i in range(len(result)):
        finish = finish + result[i]
    return finish


def colordetect(image, info):
    # img = cv2.imread("E:/picture/20190120/Calibration/Photos/27-2.jpg")
    # imgtemp = cv2.imread("E:/picture/20190120/Calibration/template/light4-2.jpg")
    # img = cv2.imread("E:/picture/20190120/test/Photos/2-2.jpg")
    bMoreLight = info["bMoreLight"]
    # if bMoreLight == 1:
    # 	imgtemp = cv2.imread('../info/20190420/template/1-1_1.jpg')
    # else:
    # 	imgtemp = cv2.imread("../info/20190420/template/1-1_1.jpg")
    imgtemp = info["template"]
    imgsift = meterFinderNoinfoBySIFT(image, imgtemp)  # SIFT匹配
    if bMoreLight != 1:
        x, y, c = imgsift.shape
        imgsift = imgsift[0:int(x / 2), :]
    # imgsift = img[minY:maxY,minX:maxX]
    result = color(imgsift, info)
    # cv2.namedWindow("imgsift", cv2.WINDOW_NORMAL)
    # cv2.imshow('imgsift', imgsift)
    return result


# if __name__=='__main__':
# 	img = cv2.imread("E:/picture/20190117/8.jpg")
# 	imgtemp = cv2.imread("E:/picture/20190117/light2.jpg")
# 	imgsift = meterFinderNoinfoBySIFT(img, imgtemp)
# 	file = open("D:/PycharmProjects/nandacode/meterReader v4/config/colorRC" + ".json")
# 	# file = open("config/colordetect" + ".json")
# 	data = json.load(file)
# 	result = highlight(imgsift, data)
# 	print(result)

def meterFinderNoinfoBySIFT(image, template):
    """
    locate meter's bbox
    :param image: image
    :param info: info
    :return: bbox image
    """
    # template = info["template"]
    # startPoint = (info["startPoint"]["x"], info["startPoint"]["y"])
    # centerPoint = (info["centerPoint"]["x"], info["centerPoint"]["y"])
    # endPoint = (info["endPoint"]["x"], info["endPoint"]["y"])
    # startPointUp = (info["startPointUp"]["x"], info["startPointUp"]["y"])
    # endPointUp = (info["endPointUp"]["x"], info["endPointUp"]["y"])
    # centerPointUp = (info["centerPointUp"]["x"], info["centerPointUp"]["y"])

    templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
    imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)

    sift = cv2.xfeatures2d.SIFT_create()

    # shape of descriptor n * 128, n is the num of key points.
    # a row of descriptor is the feature of related key point.

    try:
        templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
        imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)
        # for debug
        # templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
        # imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
        # cv2.imshow("template", templateBlurred)
        # cv2.imshow("image", imageBlurred)

        # match

        bf = cv2.BFMatcher()
        # k = 2, so each match has 2 points. 2 points are sorted by distance.
        matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)
        # The first one is better than the second one
        good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]
        if good == []:
            return template
    except:
        return template
    # distance matrix
    templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
    imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
    templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
    imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")

    # del bad match
    distances = []
    maxAbnormalNum = 15
    for i in range(len(good)):
        diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
        # distance between distance features
        diff.sort()
        distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))
    averageDistance = np.average(distances)
    good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]

    # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good, None, flags=2)
    # cv2.imshow("matchImage", matchImage)
    # cv2.waitKey(0)

    if len(good2) < 3:
        # return "the template is not matched!"
        return template

    # 寻找转换矩阵 M
    src_pts = np.float32([templateKeyPoint[m[0].queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    dst_pts = np.float32([imageKeyPoint[m[0].trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    # print(src_pts,dst_pts)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w, _ = template.shape
    # 找出匹配到的图形的四个点和标定信息里的所有点
    pts = np.float32(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]
         # [startPointUp[0], startPointUp[1]],
         # [endPointUp[0], endPointUp[1]],
         # [centerPointUp[0], centerPointUp[1]]
         ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 校正图像
    vector = (dst[3][0][0] - dst[0][0][0], dst[3][0][1] - dst[0][0][1])
    cos = (vector[0] * (200.0)) / (200.0 * math.sqrt(vector[0] ** 2 + vector[1] ** 2))
    if (vector[1] > 0):
        angle = math.acos(cos) * 180.0 / math.pi
    else:
        angle = -math.acos(cos) * 180.0 / math.pi
    # print(angle)

    change = cv2.getRotationMatrix2D((dst[0][0][0], dst[0][0][1]), angle, 1)
    src_correct = cv2.warpAffine(image, change, (image.shape[1], image.shape[0]))
    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # 获得校正后的所需要的点
    newpoints = []
    for i in range(len(pts)):
        point = newchange.dot(np.array([dst[i][0][0], dst[i][0][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)
    src_correct = src_correct[int(round(newpoints[0][1])):int(round(newpoints[1][1])),
                  int(round(newpoints[0][0])):int(round(newpoints[3][0]))]
    width = src_correct.shape[1]
    height = src_correct.shape[0]
    if width == 0 or height == 0:
        return template
    return src_correct


if __name__ == '__main__':
    # img = cv2.imread("E:/picture/20190120/Calibration/Photos/27-2.jpg")
    # imgtemp = cv2.imread("E:/picture/20190120/Calibration/template/light4-2.jpg")
    img = cv2.imread("../info/20190420/image/1-1.jpg")
    cv2.namedWindow("imgsift", cv2.WINDOW_NORMAL)
    cv2.imshow('imgsift', img)
    cv2.waitKey(0)
    file = open("../info/20190420/config/1-1_1" + ".json")
    data = json.load(file)
    result = colordetect(img, data)
    print(result)
    cv2.waitKey(0)
