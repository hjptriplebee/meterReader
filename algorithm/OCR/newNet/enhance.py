import os
import random
import cv2
from collections import defaultdict
import numpy as np



def enhance():
    images = os.listdir("images/all_data")
    for im in images:
        img = cv2.imread("images/all_data/" + im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)