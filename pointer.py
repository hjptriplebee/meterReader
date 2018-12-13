import numpy as np
import cv2
import json
import PlotUtil as plot

plot_img_index = 0


def inc():
    global plot_img_index
    plot_img_index += 1
    return plot_img_index


def recognizePointerInstrument(image, info):
    if image is None:
        print("Open Error.Image is empty.")
        return
    src = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
    plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=inc(), title="Src")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    # to make image more contrast and obvious by equalizing histogram
    src[:, :, 0] = cv2.equalizeHist(src[:, :, 0])
    src = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)
    # plot.subImage(src=src, index=++plot_img_index, title="Src")
    plot.subImage(src=src, index=inc(), title="EqualizedHistSrc")
    canny = cv2.Canny(src, 75, 75 * 2, edges=None)
    plot.subImage(src=canny, title="Canny", index=inc())
    # calculate edge by Sobel operator
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(src=gray, dx=1, dy=0, ddepth=cv2.CV_8UC1, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src=gray, dx=0, dy=1, ddepth=cv2.CV_8UC1, borderType=cv2.BORDER_DEFAULT)
    cv2.convertScaleAbs(grad_x, grad_x)
    cv2.convertScaleAbs(grad_y, grad_y)
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    # get binarization image by Otsu'algorithm
    ret, otsu = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(ret)
    adaptive = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    canny = cv2.Canny(src, 75, 75 * 2)
    plot.subImage(cmap='gray', src=gray, title='gray', index=inc())
    plot.subImage(cmap='gray', src=grad_x, title="GradX", index=inc())
    plot.subImage(cmap='gray', src=grad_y, title="GradY", index=inc())
    plot.subImage(cmap='gray', src=grad, title="Grad", index=inc())
    plot.subImage(cmap='gray', src=otsu, title="Otsu", index=inc())
    plot.subImage(cmap='gray', src=adaptive, title="Adaptive", index=inc())
    plot.subImage(cmap='gray', src=canny, title="Canny", index=inc())
    plot.show()


if __name__ == '__main__':
    recognizePointerInstrument(cv2.imread("image/SF6/IMG_7640.JPG"), None)
