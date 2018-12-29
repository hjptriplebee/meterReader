from sklearn.externals import joblib
import cv2 as cv


if __name__ =="__main__":

    img = cv.imread("/home/wayne/temp/mnist/6.1932.jpg", 0)
    test = img.reshape(1, 784)

    # 加载模型
    svc = joblib.load("model.m")
    pca = joblib.load("pca.m")

    # svm
    print('start pca...')
    test_x = pca.transform(test)
    print(test_x.shape)

    pre = svc.predict(test_x)
    print(pre)


