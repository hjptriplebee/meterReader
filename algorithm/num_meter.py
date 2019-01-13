from algorithm.Common import *
import cv2
import numpy as np
from algorithm.OCR.utils import *

info = {
    "distance": 10.0,
    "horizontal": 10.0,
    "vertical": 20.0,
    "name": "nummeter1_1",
    "type": "num_meter",
    "ROI": {
        "x": 229,
        "y": 114,
        "w": 979,
        "h": 900
    },
    "startPoint": {
        "x": 90,
        "y": 74
    },
    "endPoint": {
        "x": 484,
        "y": 431
    },
    "centerPoint": {
        "x": 485,
        "y": 80
    },
    "rectangle": {
        "width": 400,
        "height": 360
    },
    "widthSplit": [
        [140, 192, 244, 295, 347],
        [140, 192, 244, 295, 347],
        [140, 192, 244, 295, 347],
        [140, 192, 244, 295, 347],
        [54, 86, 119, 151, 185, 218, 245, 281, 315, 347]
    ],
    "heightSplit":[[3,70],
                   [72,138],
                   [139,206],
                   [208,278],
                   [292,340]
                   ],
    "startValue": 0.0,
    "totalValue": 0.0

}


class Cnn(object):
    # 将图像等比例变为28*28
    def resize_28(self, number):
        mask = np.zeros((28, 28))
        number_row = 0
        number_col = 0
        number_row = (28 - number.shape[0]) // 2
        number_col = (28 - number.shape[1]) // 2
        for i in range(number.shape[0]):
            for j in range(number.shape[1]):
                mask[i + number_row][j + number_col] = number[i][j]
        number = mask
        return number

    # 数字识别
    def cnn_num_detect(self, image):
        """
        :param image:  确保是“黑底白字”的二值图，（因为二值化后图可能是“白底黑字”）
        :return: 返回字符型的数字  其中n代表不是数字
        """
        # 将图像等比例变为28*28
        h, w = image.shape
        if (h > 28 or w > 28):
            if (h >= w):
                scale = round(26 / h, 2)
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                image = self.resize_28(image)
            else:
                scale = round(26 / w, 2)
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                image = self.resize_28(image)
        else:
            image = self.resize_28(image)
        # 确保是0和255
        image = np.array([[0 if y < 150 else 255 for y in x] for x in image])
        # 加载模型
        model = load_model('./OCR/tfNet/CNN.h5')
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        result = model.predict(image)
        max = -float('inf')
        num = ""
        for i in range(len(result[0])):
            if (max < result[0][i]):
                max = result[0][i]
                if (i == 10):
                    num = 'n'
                else:
                    num = str(i)
        return num


template = cv2.imread("../template/nummeter1_1.jpg", 1)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.equalizeHist(template)
cv2.imshow("dsa", template)
cv2.waitKey(0)
start = ([info["startPoint"]["x"], info["startPoint"]["y"]])
end = ([info["endPoint"]["x"], info["endPoint"]["y"]])
center = ([info["centerPoint"]["x"], info["centerPoint"]["y"]])
width=info["rectangle"]["width"]
height=info["rectangle"]["height"]
widthSplit=info["widthSplit"]
heightSplit=info["heightSplit"]
# 计算数字表的矩形外框，并且拉直矫正
fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
pts1 = np.float32([start, center, end, fourth])
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(template, M, (width, height))

cv2.imshow("swe", dst)
cnn = Cnn()

for i in range(len(widthSplit)):
    split=widthSplit[i]
    Num=""
    for j in range(len(split)-1):
        img=dst[heightSplit[i][0]:heightSplit[i][1],split[j]:split[j+1]]
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        sum=0
        for row in range(img.shape[0]):
            if(img[row][0]==0):
                sum+=1
            if(img[row][img.shape[1]-1]==0):
                sum+=1
        for col in range(img.shape[1]):
            if(img[0][col]==0):
                sum+=1
            if(img[img.shape[0]-1][col]==0):
                sum+=1
        if(sum<(img.shape[0]+img.shape[1])):
            img=cv2.bitwise_not(img)


        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
        img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        cv2.imshow("%d%d"%(i,j),img)
        cv2.waitKey(0)
        num = cnn.cnn_num_detect(img)
        Num += str(num)
    print(Num)

# images = [dst[3:70, 140:347], dst[72:138, 140:347], dst[139:206, 140:347], dst[208:278, 140:347], dst[293:342, 56:347]]
# for i in range(len(images)):
#     if (i != 4):
#         Num = ""
#         for j in range(4):
#             h, w = images[i].shape
#             img = images[i][:, j * (w // 4):(j + 1) * (w // 4)]
#             _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
#             num = cnn.cnn_num_detect(img)
#             Num += str(num)
#         print(Num)
#     else:
#         Num = ""
#         for j in range(9):
#             h, w = images[i].shape
#             img = images[i][:, j * (w // 9):(j + 1) * (w // 9)]
#             _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
#             num = cnn.cnn_num_detect(img)
#             Num += str(num)
#         print(Num)

cv2.waitKey()


def num_meter(image, info):
    template = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.equalizeHist(template)

    start = ([info["startPoint"]["x"], info["startPoint"]["y"]])
    end = ([info["endPoint"]["x"], info["endPoint"]["y"]])
    center = ([info["centerPoint"]["x"], info["centerPoint"]["y"]])
    width = info["rectangle"]["width"]
    height = info["rectangle"]["height"]
    widthSplit = info["widthSplit"]
    heightSplit = info["heightSplit"]
    # 计算数字表的矩形外框，并且拉直矫正
    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(template, M, (width, height))
    cnn = Cnn()
    result=[]
    for i in range(len(widthSplit)):
        split = widthSplit[i]
        Num = ""
        for j in range(len(split) - 1):
            img = dst[heightSplit[i * 2]:heightSplit[i * 2 + 1], split[j]:split[j + 1]]
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            sum = 0
            for row in range(img.shape[0]):
                if (img[row][0] == 0):
                    sum += 1
                if (img[row][img.shape[1] - 1] == 0):
                    sum += 1
            for col in range(img.shape[1]):
                if (img[0][col] == 0):
                    sum += 1
                if (img[img.shape[0] - 1][col] == 0):
                    sum += 1
            if (sum < (img.shape[0] + img.shape[1])):
                img = cv2.bitwise_not(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # cv2.imshow("%d%d" % (i, j), img)
            num = cnn.cnn_num_detect(img)
            Num += str(num)
        result.append(Num)
        print(Num)
    return result


