from Common import *

import numpy as np

def pressure(image, info):
	"""
	:param image: ROI image
	:param info: information for this meter
	:return: value
	"""
	# your method
	print("Rressure Reader called!!!")
	# template match
	meter = meterFinderByTemplate(image, info["template"])
	# cv2.imshow("meter", meter)
	img = cv2.GaussianBlur(meter, (3, 3), 0)
	# cv2.imshow("GaussianBlur ", img)
	edges = cv2.Canny(img, 100, 200, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)  # 这里对最后一个参数使用了经验型的值
	width, height, _ = img.shape
	pointer = []
	try:
		for rho, theta in lines[0]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1, y1, x2, y2 = getPointer(x0, y0, width, height)
			# draw line
			"""
			cv2.line(meter, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.imshow("HoughLine", meter)
			cv2.waitKey(0)
			"""
			pointer.append([x2-x1, y2-y1])
	except:
		print("no lines detect ")
		return 0
	pointer = np.array(pointer[0])
	start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
	end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
	center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
	totalvalue = info["totalValue"]
	startvalue = info["startValue"]
	result = AngleFactory.calPointerValueByPointerVector(start, end, center, pointer, startvalue, totalvalue)
	return result

def getPointer(x0, y0, width, height):
	"""
	limit line in the image
	:param x0:
	:param y0:
	:param width: image width
	:param height: image height
	:return: two point of the detected line
	"""
	xcenter = int(width / 2)
	ycenter = int(height / 2)
	if xcenter < x0 or (xcenter == x0 and ycenter > y0):
		x1 = xcenter
		x2 = x0
		y1 = ycenter
		y2 = y0
	else:
		x1 = x0
		x2 = xcenter
		y1 = y0
		y2 = ycenter
	return x1, y1, x2, y2
