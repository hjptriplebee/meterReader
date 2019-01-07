import numpy as np
from sklearn.externals import joblib
import torch
import sys
sys.path.append(".")


class svmOCR:
    def __init__(self):
        self.svm = joblib.load("OCR/pca-svm-master/model.m")
        self.pca = joblib.load("OCR/pca-svm-master/pca.m")

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
        self.net = torch.load("OCR/LeNet/model/net.pkl")

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
