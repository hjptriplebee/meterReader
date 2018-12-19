from Common import *

import math
import numpy as np


def pressure(image, info):
	"""
	:param image: ROI image
	:param info: information for this meter
	:return: value
	"""
	# your method
	print("twodial Reader called!!!")
	# template match
	meter = meterFinderByTemplate(image, info["template"])
	# cv2.imshow("ROI ", image)
	# cv2.imshow("meter", meter)
	pointer = getLine(meter)
	cv2.waitKey(0)
	if not pointer:
		return []
	else:
		pointer = np.array(pointer)
	start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
	end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
	center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
	totalvalue = info["totalValue"]
	startvalue = info["startValue"]
	# print("start: ", start, " end:", end, " center:", center)
	# print("pointer:", pointer)
	# print(startvalue, totalvalue)
	result = AngleFactory.calPointerValueByPointerVector(start, end, center, pointer, startvalue, totalvalue)
	print("result: ", result)
	
	return result




def getLine(img):
	img = cv2.GaussianBlur(img, (3, 3), 0)
	# cv2.imshow("GaussianBlur ", img)
	edges = cv2.Canny(img, 100, 200, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)  # 这里对最后一个参数使用了经验型的值
	width, height, _ = img.shape
	# print("img shape: ", img.shape)
	result = img.copy()
	pointer = []
	try:
		for rho, theta in lines[0]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1, y1, x2, y2 = getPointer(x0, y0, width, height)
			cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
			pointer.append([x2-x1, y2-y1])
	except:
		print("no lines detect ")
		return []
	# cv2.imshow('Canny', edges)
	cv2.imshow('HoughLines', result)
	cv2.waitKey(0)
	return pointer[0]


def getPointer(x0, y0, width, height):
	xcenter = int(width / 2)
	ycenter = int(height / 2)
	if xcenter < x0 or (xcenter == x0 and ycenter > y0):
		x1 = xcenter
		x2 = x0
		y1 = ycenter
		y2 = y0
	elif xcenter > x0 or (xcenter == x0 and ycenter < y0):
		x1 = x0
		x2 = xcenter
		y1 = y0
		y2 = ycenter
	else:
		print("error ", x0, y0, xcenter, ycenter)
		exit(1)
	return x1, y1, x2, y2
