import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim


def getImageVar(imgPath):
    image = cv2.imread(imgPath)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar
