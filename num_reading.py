import cv2
import tensorflow as tf
import numpy as np

#test
sess=tf.Session()
saver = tf.train.import_meta_graph('./checkpoint/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
graph = tf.get_default_graph()
image=graph.get_tensor_by_name("img:0")
predict=graph.get_tensor_by_name("predict:0")
def _get_num_by_threshold(Img):
    ROIYUV = cv2.cvtColor(Img,cv2.COLOR_BGR2YUV)
    ROIYUV[:,:,0] = cv2.equalizeHist(ROIYUV[:,:,0])
    Img = cv2.cvtColor(ROIYUV,cv2.COLOR_YUV2BGR)
    GrayImage=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(GrayImage,GrayImage.max()-25,GrayImage.max(),cv2.THRESH_BINARY)
    thresh=np.minimum(thresh,1)
    return thresh
def _get_each_roi(img1,n):
    all_test=[]
    l=img1.shape[1]//n
    for j in range(n):
        one=img1[:,j*l:(j+1)*l]
        one=cv2.resize(one,(28,28))
        oneMask=_get_num_by_threshold(one)
        oneMask=np.reshape(oneMask,(1,784))
        all_test.append(oneMask)
    return all_test
# input img,and value number
def get_value(img,n):
    all_test=_get_each_roi(img,n)
    results=[]
    for tests in all_test:
        prediction=sess.run(predict,feed_dict={image:tests})
        print("predict number:",prediction[0])
        results.append(prediction[0])
    return results
if __name__ == '__main__':
    img1=cv2.imread("s.jpg")
    get_value(img1,2)
