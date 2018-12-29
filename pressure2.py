import json
from Common import *


def readPressureValueFromImage(image, info):
    doEqualizeHist = False
    pyramid = 0.2
    if info['pyramid'] is not None:
        pyramid = info['pyramid']
    src = cv2.resize(image, (0, 0), fx=pyramid, fy=pyramid)
    src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    if doEqualizeHist:
        gray = cv2.equalizeHist(gray)
    canny = cv2.Canny(src, 75, 75 * 2)
    dilate_kernel = cv2.getStructuringElement(ksize=(5, 5), shape=cv2.MORPH_ELLIPSE)
    erode_kernel = cv2.getStructuringElement(ksize=(3, 3), shape=cv2.MORPH_ELLIPSE)
    # plot.subImage(src=canny, index=inc(), title='DilatedCanny', cmap='gray')
    # fill scale line with white pixels
    canny = cv2.dilate(canny, dilate_kernel)
    canny = cv2.erode(canny, erode_kernel)
    # find contours
    img, contours, hierarchy = cv2.findContours(canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    # filter the large contours, the pixel number of scale line should be small enough.
    # and the algorithm will find the pixel belong to the scale line as we need.
    contours = [c for c in contours if len(c) < 40]
    # draw contours
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    cv2.drawContours(src, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
    # prasan_iteration = rasan.getIteration(0.7, 0.3)
    dst_threshold = 35
    period_rasanc_time = 100  # 每趟rasanc 的迭代次数,rasanc内置提前终止的算法
    iter_time = 5  # 启动rasanc拟合算法的固定次数
    hit_time = 0  # 成功拟合到圆的次数lot.subImage(src=src, title='Contours', index=inc())
    theta = 0
    # load meter calibration form configuration
    start_value = info['startValue']
    total = info['totalValue']
    start_ptr = info['startPoint']
    end_ptr = info['endPoint']
    ptr_resolution = info['ptrResolution']
    if ptr_resolution is None:
        ptr_resolution = 15
    validateConfig(end_ptr, start_ptr, start_value, total)
    start_ptr = cvtPtrDic2D(start_ptr)
    end_ptr = cvtPtrDic2D(end_ptr)
    center = 0  # 表盘的中心
    radius = 0  # 表盘的半径
    # 使用拟合方式求表盘的圆,进而求出圆心
    if info['enableFit']:
        # figuring out centroids of the scale lines
        center, radius = figureOutDialCircleByScaleLine(contours, dst_threshold,
                                                        iter_time, period_rasanc_time)
    # 使用标定信息
    else:
        center = info['centerPoint']
        assert center is not None
        center = cvtPtrDic2D(center)
        # 起点和始点连接，分别求一次半径,并得到平均值
        radius_1 = np.sqrt(np.power(start_ptr[0] - center[0], 2) + np.power(start_ptr[1] - center[1], 2))
        radius_2 = np.sqrt(np.power(end_ptr[0] - center[0], 2) + np.power(end_ptr[1] - center[1], 2))
        radius = np.int64((radius_1 + radius_2) / 2)

    # 清楚可被清除的噪声区域，噪声区域(文字、刻度数字、商标等)的area 可能与指针区域的area形似,应该被清除，
    # 防止在识别指针时出现干扰。值得注意，如果当前指针覆盖了该干扰区域，指针的一部分可能也会被清除
    canny = cleanNoisedRegions(canny, info, src.shape)
    # 用直线Mask求指针区域
    hlt = np.array([center[0] + radius, center[1]])  # 通过圆心的水平线与圆的右交点
    # start_degree = math.degrees(AngleFactory.calAngleClockwise(hlt, start_ptr, center))
    # end_degree = math.degrees(AngleFactory.calAngleClockwise(end_ptr, hlt, center))
    # 计算夹角的弧度角
    start_radians = AngleFactory.calAngleClockwise(hlt, start_ptr, center)
    end_radians = AngleFactory.calAngleClockwise(end_ptr, hlt, center)
    # print("Start degree:", start_degree)
    # print("End degree:", end_degree)
    # 从特定范围搜索指针
    pointer_mask, theta, line_ptr = findPointerFromBinarySpace(canny, center, radius, start_radians, end_radians,
                                                               patch_degree=0.5,
                                                               ptr_resolution=ptr_resolution)
    value = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr,
                                                centerPoint=center,
                                                point=cv2PtrTuple2D(line_ptr), startValue=start_value,
                                                totalValue=total)
    return json.dumps({"value": value})


def validateConfig(end_ptr, start_ptr, start_value, total):
    """
    validate the meter's configuration,the specialized param should exist for algorithm effectiveness.
    :param end_ptr:
    :param start_ptr:
    :param start_value:
    :param total:
    :return:
    """
    assert start_value is not None
    assert start_ptr is not None
    assert end_ptr is not None
    assert total is not None


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
        # plot.subImage(src=src, index=inc(), title='CleanNoisedRegion', cmap='gray')
    return src


def readPressureValueFromDir(img_dir, config):
    img = cv2.imread(img_dir)
    file = open(config)
    info = json.load(file)
    assert info is not None
    readPressureValueFromImage(img, info)


def readPressureValueFromImg(img, info):
    if img is None:
        raise Exception("Failed to resolve the empty image.")
    return readPressureValueFromImage(img, info)


if __name__ == '__main__':
    readPressureValueFromDir('image/pressure2_1.jpg', 'config/pressure2_1.json')
    # readPressureValueFromDir('image/SF6/IMG_7666.JPG', 'config/otg_1.json')
    # demarcate_roi('image/SF6/IMG_7666.JPG')
