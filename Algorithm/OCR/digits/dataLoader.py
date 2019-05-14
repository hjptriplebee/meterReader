import numpy as np
import torch
import os
import cv2
import pickle
import random


class dataLoader():
    # todo 调整batch，使每个batch顺序都不一样
    def __init__(self, type, path, bs, ifUpdate):
        self.meanPixel = 80
        self.bs = bs
        self.pointer = 0
        self.type = type
        if type == 'rgb':
            self.train_path = os.path.join(path, "rgb_augment_train.pkl")
            self.test_path = os.path.join(path, "rgb_test.pkl")
        elif type == 'bit':
            self.train_path = os.path.join(path, "bit_augment_train.pkl")
            self.test_path = os.path.join(path, "bit_test.pkl")

        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            ifUpdate = True

        if ifUpdate:
            os.system("rm -rf {}".format(self.train_path))
            os.system("rm -rf {}".format(self.test_path))
            self.readImagesFromMultiFils(path)

        self.readDataFromPkl()
        self.shuffle()

    def readImagesFromMultiFils(self, path):
        for t in ["rgb_augmentation", "rgb_test"]:
            if self.type == 'bit':
                data = torch.Tensor(np.zeros((1, 1, 28, 28)))
            elif self.type == 'rgb':
                data = torch.Tensor(np.zeros((1, 3, 28, 28)))
            label = []
            names = []

            for i in range(11):
                root = path + "/" + t + "/" + str(i) + "/"
                images = os.listdir(root)
                for im in images:
                    if im.split(".")[-1] != "bmp":
                        continue
                    # print(img.shape)
                    names.append(root+im)
                    if self.type == "bit":
                        img = cv2.imread(root + im)[:, :, 0]
                        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
                        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 11)
                        # 增强
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
                        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                        # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    elif self.type == 'rgb':
                        img = cv2.imread(root + im)
                        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
                        if len(img.shape) == 2:
                            img = np.array(img, img, img)
                            print("convert to rgb: ", img.shape)

                    temp = torch.Tensor(img).view(1, 3, 28, 28) - self.meanPixel
                    data = torch.cat((data, temp), 0)

                    label.append(i)
            if t.endswith("test"):
                fp = open(self.test_path, "wb")
                pickle.dump([data[1:], torch.Tensor(np.array(label)).long(), names], fp)
            else:
                fp = open(self.train_path, "wb")
                pickle.dump([data[1:], torch.Tensor(np.array(label)).long()], fp)
            fp.close()

    def readDataFromPkl(self):
        with open(self.train_path, "rb") as fp:
            self.trainData, self.trainLabel = pickle.load(fp)
        with open(self.test_path, "rb") as fp:
            self.testData, self.testLabel, self.names = pickle.load(fp)

    def getTrainData(self):
        return self.trainData, self.trainLabel

    def getTestData(self):
        return self.testData, self.testLabel, self.names

    def shuffle(self):
        li = list(range(self.trainData.shape[0]))
        random.shuffle(li)
        self.trainData = self.trainData[li]
        self.trainLabel = self.trainLabel[li]

    def next_batch(self):
        if self.pointer * self.bs == self.trainData.shape[0]:
            self.pointer = 0

        if (self.pointer + 1) * self.bs > self.trainData.shape[0]:
            temp = self.pointer
            self.pointer = 0
            return self.trainData[temp * self.bs:], \
                   self.trainLabel[temp * self.bs:]

        temp = self.pointer
        self.pointer += 1

        return self.trainData[temp * self.bs:self.pointer * self.bs], \
               self.trainLabel[temp * self.bs:self.pointer * self.bs]

    def get_rounds(self):
        return int(self.trainData.shape[0] / self.bs) + 1


# if __name__ == "__main__":
#     dl = dataLoader("dataset/", 64, True)
#     dl.shuffle()
#     train, trl = dl.getTrainData()
#     test, tel = dl.getTestData()
#     #
#     print(train.shape, trl)
#     print(test.shape, tel)
#     #
#
#     print(dl.trainLabel)
#     dl.shuffle()
#     print(dl.trainLabel)
