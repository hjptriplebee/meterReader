import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader

import scipy.io as sio
import numpy as np
import cv2

from model.LeNet import LeNet

net = torch.load("./model/net.pkl")
print("成功加载模型")
transform = transforms.ToTensor()
BATCH_SIZE = 64
device = "cpu"

# 定义测试数据集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

dict = {}
ifShow = False
with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(testloader):
        images, labels = data

        # 修改minst中的输入，改为二值图
        images = torch.ceil(images)*255

        dict[str(i)] = images
        images, labels = images.to(device), labels.to(device)
        outputs = net.forward(images)

        # 取得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)

        if i == 0 and ifShow:
            for j in range(BATCH_SIZE):
                img = np.array(images[j, 0]).astype(np.uint8)
                print("label:", np.array(labels[j]), "predict:", np.array(predicted[j]))
                cv2.imshow("img", img)
                cv2.waitKey(0)

        total += labels.size(0)
        correct += (predicted == labels).sum()
    sio.savemat("test.mat", dict)
    print('识别准确率为：%d%%' % (100 * correct / total))