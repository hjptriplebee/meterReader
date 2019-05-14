import sys
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

from dataLoader import dataLoader
from net import rgbNet

def train(type):

    torch.manual_seed(10)

    bs = 256
    lr = 0.001
    epoch = 40
    stepLength = 20
    classes = 11

    data = dataLoader(type, "dataset/", bs, ifUpdate=True)
    testInputs, testLabels, _ = data.getTestData()
    print("数据集加载完毕")


    weight = np.zeros(classes)

    for i in range(classes):
        images = os.listdir("dataset/rgb_augmentation/"+str(i))
        weight[i] = len(images)
    weight = weight/np.sum(weight)
    print(weight)


    def adjust_learning_rate(optimizer, epoch, t=10):
        """Sets the learning rate to the initial LR decayed by 10 every t epochs，default=10"""
        new_lr = lr * (0.1 ** (epoch // t))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    net = rgbNet(type)
    net.train()

    criterion = torch.nn.NLLLoss(weight=torch.Tensor(weight))
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-5, lr=lr)

    steps = data.get_rounds()
    train_loss = []

    for e in range(epoch):
        sum_loss = 0.0
        for step in range(steps):

            inputs, labels = data.next_batch()
            inputs = inputs/255
            optimizer.zero_grad()

            # forward + backward
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if step % stepLength == stepLength-1:
                print('epoch %d /step %d: loss:%.03f'
                      % (e + 1, step + 1, sum_loss / stepLength))
                # sum_loss = 0.0
        train_loss.append(sum_loss)
        adjust_learning_rate(optimizer, e)
        outputs = net.forward(testInputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == testLabels.long()).sum()
        print('第%d个epoch的识别准确率为：%d%%' % (e + 1, (100 * correct / testLabels.shape[0])))
    torch.save(net.state_dict(), type + "_" + str(net.__class__.__name__) + "_net.pkl")
    plt.figure(0)
    x = [i for i in range(len(train_loss))]
    plt.plot(x, train_loss)
    plt.savefig("train_loss.jpg")

def test(type):
    import shutil
    result = type + "result/"
    if os.path.exists(result):
        shutil.rmtree(result)
    if not os.path.exists(result):
        os.makedirs(result)
    data = dataLoader(type, "dataset/", bs=256, ifUpdate=False)
    testInputs, testLabels, names = data.getTestData()
    net = rgbNet(type)
    modelname = type + "_" + str(net.__class__.__name__) + "_net.pkl"
    net.load_state_dict(torch.load(modelname))
    net.eval()
    print("model load")
    outputs = net.forward(testInputs)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == testLabels.long()).sum()
    print('识别准确率为：%d%%' % ((100 * correct / testLabels.shape[0])))

    # show result picture
    for i in range(testInputs.shape[0]):
        testimg = cv2.imread(names[i])
        res = net.forward(testInputs[i].view(1, 3, 28, 28))
        _, predicted = torch.max(res.data, 1)
        # cv2.imshow("test", test)
        name = os.path.join(result, str(i) + "__" + str(predicted.numpy()) + ".bmp")
        cv2.imwrite(name, testimg)
        # print(predicted)
        # cv2.waitKey(0)
    print("done!")

if __name__ == "__main__":
    train('rgb')
    test('rgb')



