from algorithm.OCR.newNet.dataLoader import *
from algorithm.OCR.newNet.LeNet import *

import sys
sys.path.append("../LeNet")

import numpy as np
import torch
import torch.optim as optim
import torch.nn.init as init


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


torch.manual_seed(10)

bs = 64
lr = 0.001
epoch = 5

data = dataLoader("train", "test", bs)

# net = torch.load("../LeNet/model/net.pkl")
net = myNet()
net.apply(weights_init)
# torch.initial_seed()
# net = LeNet()

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

steps = data.get_rounds()

for e in range(epoch):
    sum_loss = 0.0
    for step in range(steps):
        inputs, labels = data.next_batch()

        optimizer.zero_grad()

        # forward + backward
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        if step % 10 == 0:
            print('epoch %d /step %d: loss:%.03f'
                  % (e + 1, step + 1, sum_loss / 10))
            sum_loss = 0.0

    testInputs, testLabels = data.readImages(data.testPath)

    outputs = net.forward(testInputs)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == testLabels.long()).sum()
    print('第%d个epoch的识别准确率为：%d%%' % (e + 1, (100 * correct / testLabels.shape[0])))

# for i in range(testInputs.shape[0]):
#     test = np.array(testInputs[i].view(28, 28))
#     res = net.forward(testInputs[i].view(1,1,28,28))
#     cv2.imshow("test", test)
#     print(torch.max(res.data, 1))
#     cv2.waitKey(0)
torch.save(net.state_dict(), "net.pkl")


