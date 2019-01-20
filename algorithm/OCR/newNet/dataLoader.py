import numpy as np
import torch
import os
import cv2


class dataLoader():
    def __init__(self, trainPath, testPath, bs):
        self.trainPath = trainPath
        self.testPath = testPath
        self.bs = bs
        self.pointer = 0

        self.trainData, self.trainLabel = self.readImages(self.trainPath)

    def readImages(self, path):
        images = os.listdir(path)
        data = torch.Tensor(cv2.imread(path+"/"+images[0])[:, :, 0]).view(1, 1, 28, 28)
        label = [int(images[0].split("_")[0])]
        for im in images:
            if im.split(".")[-1] != "bmp": # or im.split("_")[0] == "n":
                continue
            this = torch.Tensor(cv2.imread(path+"/"+im)[:, :, 0]).view(1, 1, 28, 28)
            data = torch.cat((data, this), 0)
            thisLabel = im.split("_")[0]
            if thisLabel == "n":
                label.append(10)
            else:
                label.append(int(thisLabel))
        return data, torch.Tensor(np.array(label))

    def next_batch(self):
        if self.pointer*self.bs == self.trainData.shape[0]:
            self.pointer = 0

        if (self.pointer+1)*self.bs > self.trainData.shape[0]:
            temp = self.pointer
            self.pointer = 0
            return self.trainData[temp*self.bs:], \
                   self.trainLabel[temp*self.bs:]

        temp = self.pointer
        self.pointer += 1

        return self.trainData[temp*self.bs:self.pointer*self.bs],\
               self.trainLabel[temp*self.bs:self.pointer*self.bs]

    def get_rounds(self):
        return int(self.trainData.shape[0]/self.bs)+1


dl = dataLoader("train", "test", 64)

for i in range(2):
    input_, label = dl.next_batch()
    if i == 1:
        for n in range(input_.shape[0]):
            img = np.array(input_[n].view(28, 28, 1))
            cv2.imshow("img",img)
            print(label[n])
            cv2.waitKey(0)
    print(input_.shape, label)
