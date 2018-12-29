import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.externals import joblib
import time
import cv2
import numpy as np

'''
ReadMe:
该脚本用于训练SVM识别数字的模型框架
采用Mnist作为训练测试机进行训练
可修改参数：
train_num：训练集大小
test_num：测试集大小
PCA-n_components ：主成分比例
SVM中的各项参数等
'''

if __name__ == "__main__":
    train_num = 8000
    test_num = 2000
    data = pd.read_csv('train_v1.csv').values
    print(data.shape)
    # train_data = data[0:train_num, 2:]
    train_data = data[0:train_num, 2:]
    train_label = data[0:train_num:, 1]
    test_data = data[train_num:train_num+test_num, 2:]
    test_label = data[train_num:train_num+test_num, 1]

    _, train_data = cv2.threshold(np.array(train_data).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    _, test_data = cv2.threshold(np.array(test_data).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    t = time.time()

    print(np.unique(train_label))

    # PCA降维
    pca = PCA(n_components=0.9, whiten=True)
    print('start pca...')
    train_x = pca.fit_transform(train_data)

    print(train_x.shape)

    # svm训练
    print('start svc...')
    svc = svm.SVC(kernel='rbf', C=11)
    svc.fit(train_x, train_label)

    # 保存模型
    joblib.dump(svc, 'model.m')
    joblib.dump(pca, 'pca.m')

    # 计算准确率
    # svc = joblib.load("model.m")
    # pca = joblib.load("pca.m")
    test_x = pca.transform(test_data)
    pre = svc.predict(test_x)
    score = svc.score(test_x, test_label)
    print(u'准确率：%f,花费时间：%.2fs' % (score, time.time() - t))




