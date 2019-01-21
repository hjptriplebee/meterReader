import numpy as np
import torch
import os
import cv2
import pickle
import random


class dataLoader():
    # todo 调整batch，使每个batch顺序都不一样
    def __init__(self, trainPath, testPath, bs):
        self.trainPath = trainPath
        self.testPath = testPath
        self.bs = bs
        self.pointer = 0

        if not os.path.exists("train"):
            self.readImagesIntoPkl("train")
        if not os.path.exists("test"):
            self.readImagesIntoPkl("test")

        self.trainData, self.trainLabel = self.readTrainFromPkl()

    def readImagesIntoPkl(self, path):
        images = os.listdir(path)
        image = cv2.imread(path+"/"+images[0])[:, :, 0]
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        data = torch.Tensor(image).view(1, 1, 28, 28)
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

        fp = open(path+".pkl", "wb")
        pickle.dump([data, torch.Tensor(np.array(label)).long()], fp)
        fp.close()

    def readTrainFromPkl(self):
        with open("train.pkl", "rb") as fp:
            trainData, trainLabel = pickle.load(fp)
            return trainData, trainLabel

    def readTestFromPkl(self):
        with open("test.pkl", "rb") as fp:
            testData, testLable = pickle.load(fp)
            return testData, testLable

    def shuffle(self):
        li = list(range(self.trainData.shape[0]))
        random.shuffle(li)
        self.trainData = self.trainData[li]
        self.trainLabel = self.trainLabel[li]

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
dl.shuffle()
# train, trl = dl.readTrainFromPkl()
# test, tel = dl.readTestFromPkl()
#
# print(train.shape, trl)
# print(test.shape, tel)
#

