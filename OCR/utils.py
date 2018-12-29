import numpy as np
from sklearn.externals import joblib
import torch
import sys

sys.path.append("LeNet")


def svmInit():
    """
    初始化svm参数
    :return: svm模型
    """
    svm = joblib.load("pca-svm-master/model.m")
    pca = joblib.load("pca-svm-master/pca.m")
    return [svm, pca]


def leNetInit():
    """
    初始化LeNet模型
    :return:
    """
    net = torch.load("LeNet/model/net.pkl")
    return net


def recognizeSvm(image, svc):
    """
    SVM识别图像中的数字
    :param image: 输入图像
    :param svc: SVM模型
    :return: 识别的数字值
    """
    if image.size != 784:
        print("检查输入图片大小！不为28*28")
        return None
    svm, pca = svc
    image = image.reshape(1, 784)
    # svm
    test_x = pca.transform(image)
    num = svm.predict(test_x)
    return num[0]


def recognizeNet(image, net):
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
    result = net.forward(image)
    _, predicted = torch.max(result.data, 1)
    num = int(np.array(predicted[0]).astype(np.uint32))
    return num
