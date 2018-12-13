import numpy as np
import cv2 as cv
import json
from matplotlib import pyplot as plot


def recognizePointerInstrument(image, info):
    if (image is None):
        print("Open Error.Image is empty.")
        return
    src = cv.resize(image, (0, 0), fx=0.2, fy=0.2)
    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    plot.figure(figsize=(10, 20))
    plot.imshow(src), plot.title("Src")
    canny = cv.Canny(src, 75, 75 * 2, edges=None)
    plot.imshow(canny), plot.title("Canny")
    plot.show()


if __name__ == '__main__':
    recognizePointerInstrument(cv.imread("image/SF6/IMG_7640.JPG"), None)
