import numpy as np
import cv2
import json
import RasancFitCircle as rasan
import PlotUtil as plot

plot_img_index = 0
window_name = "Meter Line Connection"
ed_src = None


def inc():
    global plot_img_index
    plot_img_index += 1
    return plot_img_index


def recognizePointerInstrument(image, info):
    global window_name, kernel_size, max_kernel_size, ed_src
    doEqualizeHist = False
    if image is None:
        print("Open Error.Image is empty.")
        return
    src = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
    rgb_src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=inc(), title="Src")
    # A. The Template Image Processing
    src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    # src = cv2.medianBlur(src, ksize=9)

    # to make image more contrast and obvious by equalizing histogram
    # if doEqualizeHist:
    #     src = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    #     src[:, :, 0] = cv2.equalizeHist(src[:, :, 0])
    #     src = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)
    # plot.subImage(src=src, index=++plot_img_index, title="Src")

    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=inc(), title="Blurred")
    canny = cv2.Canny(src, 75, 75 * 2, edges=None)
    # calculate edge by Sobel operator
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    if doEqualizeHist:
        gray = cv2.equalizeHist(gray)
    plot.subImage(src=gray, index=inc(), title='Gray', cmap='gray')
    # usa a large structure element to fix high light in case otust image segmentation error.
    structuring_element = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(101, 101))
    gray = cv2.morphologyEx(src=gray, op=cv2.MORPH_BLACKHAT, kernel=structuring_element)
    plot.subImage(src=gray, index=inc(), title='BlackHap', cmap='gray')
    # B. Edge Detection
    grad_x = cv2.Sobel(src=gray, dx=1, dy=0, ddepth=cv2.CV_8UC1, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src=gray, dx=0, dy=1, ddepth=cv2.CV_8UC1, borderType=cv2.BORDER_DEFAULT)
    cv2.convertScaleAbs(grad_x, grad_x)
    cv2.convertScaleAbs(grad_y, grad_y)
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    # get binarization image by Otsu'algorithm
    ret, ostu = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = ostu
    canny = cv2.Canny(src, 75, 75 * 2)
    # plot.subImage(cmap='gray', src=gray, title='gray', index=inc())
    # plot.subImage(cmap='gray', src=grad_x, title="GradX", index=inc())
    # plot.subimage(cmap='gray', src=grad_y, title="grady", index=inc())
    plot.subImage(cmap='gray', src=grad, title="grad", index=inc())
    # plot.subimage(cmap='gray', src=otsu, title="otsu", index=inc())
    # plot.subimage(cmap='gray', src=adaptive, title="adaptive", index=inc())

    plot.subImage(cmap='gray', src=canny, title="Canny", index=inc())
    kernel = cv2.getStructuringElement(ksize=(5, 5), shape=cv2.MORPH_ELLIPSE)
    kernel_cy = cv2.getStructuringElement(ksize=(2, 2), shape=cv2.MORPH_CROSS)
    ed_canny = cv2.erode(canny, kernel_cy)
    ed_canny = cv2.dilate(ed_canny, kernel_cy)
    plot.subImage(src=ed_canny, index=inc(), title='EDCanny', cmap='gray')
    threshold = cv2.dilate(threshold, kernel)
    threshold = cv2.erode(threshold, kernel)
    canny = cv2.dilate(canny, kernel)
    canny = cv2.erode(canny, kernel)
    # cv2.createTrackbar("Kernel:", window_name, 1, 20, dilate_erode)
    # cv2.imshow(window_name, otsu)
    plot.subImage(cmap='gray', src=threshold, title='DilateAndErodeThresh', index=inc())
    plot.subImage(cmap='gray', src=canny, title='DilateAndErodeCanny', index=inc())

    img, contours, hierarchy = cv2.findContours(canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    # filter some large contours, the pixel number of scale line should be small enough.
    # and the algorithm will find the pixel belong to the scale line as we need.
    contours = [c for c in contours if len(c) < 45]
    ## Draw Contours
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    cv2.drawContours(src, contours, -1, (0, 255, 0), thickness=1)
    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), title='Contours', index=inc())

    # C. Figuring out Centroids of the Scale Lines
    centroids = []
    mut = []
    for contour in contours:
        mu = cv2.moments(contour)
        if mu['m00'] != 0:
            centroids.append((mu['m10'] / mu['m00'], mu['m01'] / mu['m00']))
    length = len(centroids)
    # for i in range(0, length - 1):
    #     p1 = (int(centroids[i][0]), int(centroids[i][1]))
    #     p2 = (int(centroids[i + 1][0]), int(centroids[i + 1][1]))
    #     cv2.line(src, p1, p2, color=(0, 255, 0), thickness=3)
    for centroid in centroids:
        rgb_src[int(centroid[0]), int(centroid[1])] = (0, 255, 0)
    plot.subImage(src=rgb_src, index=inc(), title="Centroids")
    # plot.subImage(src=src, index=inc(), title='Line')
    rasan_iteration = rasan.getIteration(0.7, 0.3)
    avg_circle = np.zeros(3, dtype=np.float64)
    avg_fit_num = 0
    dst_threshold = 40
    period_rasanc_time = 100  # 每趟rasanc 的迭代次数,rasanc内置提前终止的算法
    iter_time = 5  # 启动rasanc拟合算法的固定次数
    hit_time = 0 # 成功拟合到圆的次数
    # 为了确保拟合所得的圆的确信度，多次拟合求平均值
    for i in range(iter_time):
        best_circle, max_fit_num, best_consensus_pointers = rasan.randomSampleConsensus(centroids,
                                                                                        max_iterations=period_rasanc_time,
                                                                                        dst_threshold=dst_threshold,
                                                                                        inliers_threshold=len(
                                                                                            centroids) / 2,
                                                                                        optimal_consensus_num=int(
                                                                                            len(centroids) * 0.8))
        if max_fit_num > 0:
            hit_time += 1
            avg_circle += best_circle
            avg_fit_num += max_fit_num
    if hit_time > 0:
        avg_circle /= hit_time
        avg_fit_num /= hit_time
    # 求平均值减少误差
    if avg_fit_num > len(centroids) / 2:
        cv2.circle(rgb_src, center=(np.int(avg_circle[0]), np.int(avg_circle[1])), radius=np.int(avg_circle[2]),
                   color=(0, 0, 255),
                   thickness=2,
                   lineType=cv2.LINE_AA)
        plot.subImage(src=rgb_src, index=inc(), title='Fitted Circle')
    else:
        print("Fitting Circle Failed.")
    plot.show()


def on_touch(val):
    return None


def dilate_erode(kernel_size):
    kernel = cv2.getStructuringElement(ksize=(kernel_size * 2 + 1, kernel_size * 2 + 1), shape=cv2.MORPH_ELLIPSE)
    src = cv2.dilate(ed_src, kernel)
    src = cv2.erode(src, kernel)
    # cv2.imshow(window_name, src)
    return src


def compareEqualizeHistBetweenDiffEnvironment():
    src1 = cv2.imread('image/SF6/IMG_7638.JPG', cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread('image/SF6/IMG_7640.JPG', cv2.IMREAD_GRAYSCALE)
    src1 = cv2.resize(src1, (0, 0), fx=0.2, fy=0.2)
    src2 = cv2.resize(src2, (0, 0), fx=0.2, fy=0.2)
    if src1 is None or src2 is None:
        return
    print(src1.shape)
    print(src2.shape)
    hist1 = cv2.calcHist(images=[src1], channels=[0], histSize=[256], ranges=[0, 256], mask=None)
    hist2 = cv2.calcHist(images=[src2], channels=[0], histSize=[256], ranges=[0, 256], mask=None)
    equalizedSrc1 = cv2.equalizeHist(src1)
    equalizedSrc2 = cv2.equalizeHist(src2)
    equalizedHist1 = cv2.calcHist(images=[equalizedSrc1], channels=[0], histSize=[256], ranges=[0, 256], mask=None)
    equalizedHist2 = cv2.calcHist(images=[equalizedSrc2], channels=[0], histSize=[256], ranges=[0, 256], mask=None)
    # hist_np1 = np.histogram(src1.ravel(), 256, [0, 256])
    # hist_np2 = np.histogram(src2.ravel(), 256, [0, 256])
    # plot.subImage(cmap='gray', src=src1, title='Src1', index=inc())
    plot.subImage(cmap='gray', src=src2, title='Src2', index=inc())
    plot.plot(hist1, index=inc(), title="Hist1")
    plot.plot(hist2, index=inc(), title="Hist2")
    plot.subImage(cmap='gray', src=equalizedSrc1, title="EqualizedSrc1", index=inc())
    plot.subImage(cmap='gray', src=equalizedSrc2, title="EqualizedSrc2", index=inc())
    plot.plot(equalizedHist1, index=inc(), title='EqualizedHist1')
    plot.plot(equalizedHist2, index=inc(), title='EqualizedHist2')
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(41, 41))
    black_hat = cv2.morphologyEx(src1, kernel=kernel, op=cv2.MORPH_BLACKHAT)
    plot.subImage(cmap='gray', src=black_hat, title='TopTrans', index=inc())
    # cv2.imshow("hist1",hist1)
    # cv2.imshow("hist2", src2)
    # cv2.waitkey(0)
    # plot.subimage(cmap='gray', src=hist_np1, title="histnp1", index=inc())
    # plot.subimage(cmap='gray', src=hist_np2, title="histnp2", index=inc())
    plot.show()


if __name__ == '__main__':
    recognizePointerInstrument(cv2.imread("image/SF6/IMG_DOUBLE.JPG"), None)

    # compareEqualizeHistBetweenDiffEnvironment()
