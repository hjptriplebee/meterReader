import cv2 as cv
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib
import torch
import sys

sys.path.append(".")


class svmOCR:
    def __init__(self):
        self.svm = joblib.load("algorithm/OCR/pca-svm-master/model.m")
        self.pca = joblib.load("algorithm/OCR/pca-svm-master/pca.m")

    def recognizeSvm(self, image):
        """
        SVM识别图像中的数字
        :param image: 输入图像
        :param svc: SVM模型
        :return: 识别的数字值
        """
        if image.size != 784:
            print("检查输入图片大小！不为28*28")
            return None

        image = image.reshape(1, 784)
        # svm
        test_x = self.pca.transform(image)
        num = self.svm.predict(test_x)
        return num[0]


class leNetOCR:
    def __init__(self):
        """
        初始化LeNet模型
        :return:
        """
        sys.path.append("LeNet")
        self.net = torch.load("algorithm/OCR/LeNet/model/net.pkl")

    def recognizeNet(self, image):
        """
        LeNet识别图像中的数字
        :param image: 输入图像
        :param net: LeNet模型
        :return: 识别的数字值
        """
        if image.size != 784:
            print("检查输入图片大小！不为28*28")
            return None
        image = torch.Tensor(image).view((1, 1, 28, 28))
        image = image.to("cpu")
        result = self.net.forward(image)
        _, predicted = torch.max(result.data, 1)
        num = int(np.array(predicted[0]).astype(np.uint32))
        return num

class Cnn(object):
    #将图像等比例变为28*28
    def resize_28(self,number):
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
    #数字识别
    def cnn_num_detect(self,image):
        """
        :param image:  确保是“黑底白字”的二值图，（因为二值化后图可能是“白底黑字”）
        :return: 返回字符型的数字  其中n代表不是数字
        """
        #将图像等比例变为28*28
        h, w = image.shape
        if (h > 28 or w > 28):
            if (h >= w):
                scale = round(26 / h, 2)
                image = cv.resize(image, (0, 0), fx=scale, fy=scale)
                image = self.resize_28(image)
            else:
                scale = round(26 / w, 2)
                image = cv.resize(image, (0, 0), fx=scale, fy=scale)
                image = self.resize_28(image)
        else:
            image =self. resize_28(image)
        # 确保是0和255
        image = np.array([[0 if y < 150 else 255 for y in x] for x in image])
        # 加载模型
        model = load_model('./CNN.h5')
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
