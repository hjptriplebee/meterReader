from Common import meterFinderByTemplate, meterFinderBySIFT, scanPointer
import json
import cv2
import numpy as np

plot_index = 0

def normalPressure(image, info):
	center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
	start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
	end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
	meter = meterFinderByTemplate(image, info["template"])
	result = scanPointer(meter, [start, end, center], info["startValue"], info["totalValue"])
	return result

def inc():
	global plot_index
	plot_index += 1
	return plot_index


def readPressure(image, info):
	src = meterFinderByTemplate(image, info["template"])
	pyramid = 0.5
	if 'pyramid' in info and info['pyramid'] is not None:
		pyramid = info['pyramid']
		src = cv2.resize(src, (0, 0), fx=pyramid, fy=pyramid)
	src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
	gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
	thresh = gray.copy()
	cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV, thresh)
	thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
	do_hist = info["enableEqualizeHistogram"]
	if do_hist:
		gray = cv2.equalizeHist(gray)
	canny = cv2.Canny(src, 75, 75 * 2)
	dilate_kernel = cv2.getStructuringElement(ksize=(3, 3), shape=cv2.MORPH_ELLIPSE)
	erode_kernel = cv2.getStructuringElement(ksize=(1, 1), shape=cv2.MORPH_ELLIPSE)
	# fill scale line with white pixels
	canny = cv2.dilate(canny, dilate_kernel)
	canny = cv2.erode(canny, erode_kernel)
	# find contours
	img, contours, hierarchy = cv2.findContours(canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
	# filter the large contours, the pixel number of scale line should be small enough.
	# and the algorithm will find the pixel belong to the scale line as we need.
	contours_thresh = info["contoursThreshold"]
	contours = [c for c in contours if len(c) < contours_thresh]
	# draw contours
	src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
	cv2.drawContours(src, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
	# prasan_iteration = rasan.getIteration(0.7, 0.3)
	# load meter calibration form configuration
	double_range = info['doubleRange']
	start_ptr = info['startPoint']
	end_ptr = info['endPoint']
	ptr_resolution = info['ptrResolution']
	if ptr_resolution is None:
		ptr_resolution = 15
	start_ptr = cvtPtrDic2D(start_ptr)
	end_ptr = cvtPtrDic2D(end_ptr)
	center = 0  # 表盘的中心
	radius = 0  # 表盘的半径
	center = info['centerPoint']
	center = cvtPtrDic2D(center)
	# 起点和始点连接，分别求一次半径,并得到平均值
	radius = calAvgRadius(center, end_ptr, radius, start_ptr)
	# 清楚可被清除的噪声区域，噪声区域(文字、刻度数字、商标等)的area 可能与指针区域的area形似,应该被清除，
	# 防止在识别指针时出现干扰。值得注意，如果当前指针覆盖了该干扰区域，指针的一部分可能也会被清除
	canny = cleanNoisedRegions(canny, info, src.shape)
	# 用直线Mask求指针区域
	hlt = np.array([center[0] + radius, center[1]])  # 通过圆心的水平线与圆的右交点
	# 计算夹角的弧度角
	start_radians = AngleFactory.calAngleClockwise(hlt, start_ptr, center)
	end_radians = AngleFactory.calAngleClockwise(end_ptr, hlt, center)
	# 从特定范围搜索指针
	pointer_mask, theta, line_ptr = findPointerFromBinarySpace(thresh, center, radius, 0, 2 * np.pi,
	                                                           patch_degree=0.5,
	                                                           ptr_resolution=ptr_resolution)
	print("Best theta:", theta)
	line_ptr = cv2PtrTuple2D(line_ptr)
	cv2.line(src, (start_ptr[0], start_ptr[1]), (center[0], center[1]), color=(0, 0, 255), thickness=1)
	cv2.line(src, (end_ptr[0], end_ptr[1]), (center[0], center[1]), color=(0, 0, 255), thickness=1)
	cv2.circle(src, (start_ptr[0], start_ptr[1]), 5, (0, 0, 255), -1)
	cv2.circle(src, (end_ptr[0], end_ptr[1]), 5, (0, 0, 255), -1)
	cv2.circle(src, (center[0], center[1]), 2, (0, 0, 255), -1)
	if double_range:
		start_value_in = info['startValueIn']
		total_value_in = info['totalValueIn']
		start_value_out = info['startValueOut']
		total_value_out = info['totalValueOut']
		valueIn = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr, centerPoint=center,
		                                              point=line_ptr, startValue=start_value_in,
		                                              totalValue=total_value_in)
		valueOut = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr, centerPoint=center,
		                                               point=line_ptr, startValue=start_value_out,
		                                               totalValue=total_value_out)
		return json.dumps({
			"valueIn": valueIn,
			"valueOut": valueOut
		})
	else:
		start_value = info['startValue']
		total = info['totalValue']
		value = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr,
		                                            centerPoint=center,
		                                            point=line_ptr, startValue=start_value,
		                                            totalValue=total)
		return json.dumps({"value": value})


def calAvgRadius(center, end_ptr, radius, start_ptr):
	radius_1 = np.sqrt(np.power(start_ptr[0] - center[0], 2) + np.power(start_ptr[1] - center[1], 2))
	radius_2 = np.sqrt(np.power(end_ptr[0] - center[0], 2) + np.power(end_ptr[1] - center[1], 2))
	radius = np.int64((radius_1 + radius_2) / 2)
	return radius


def cvtPtrDic2D(dic_ptr):
	"""
	point.x,point.y转numpy数组
	:param dic_ptr:
	:return:
	"""
	if dic_ptr['x'] and dic_ptr['y'] is not None:
		dic_ptr = np.array([dic_ptr['x'], dic_ptr['y']])
	else:
		return np.array([0, 0])
	return dic_ptr


def cv2PtrTuple2D(tuple):
	"""
	tuple 转numpy 数组
	:param tuple:
	:return:
	"""
	if tuple[0] and tuple[1] is not None:
		tuple = np.array([tuple[0], tuple[1]])
	else:
		return np.array([0, 0])
	return tuple


def cleanNoisedRegions(src, info, shape):
	"""
	根据标定信息清楚一些有干扰性的区域
	:param src:
	:param info:
	:param shape:
	:return:
	"""
	if 'noisedRegion' in info and info['noisedRegion'] is not None:
		region_roi = info['noisedRegion']
		for roi in region_roi:
			mask = cv2.bitwise_not(np.zeros(shape=(shape[0], shape[1]), dtype=np.uint8))
			mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 0
			src = cv2.bitwise_and(src, mask)
	return src

