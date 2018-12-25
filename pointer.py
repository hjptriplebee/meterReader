import json
from Common import *
from uitl import PlotUtil as plot, RasancFitCircle as rasan, DrawSector as ds, ROIUtil as roiutil
import math

plot_img_index = 0
ed_src = None


def inc():
    global plot_img_index
    plot_img_index += 1
    return plot_img_index


def recognizePointerInstrument(image, info):
    doEqualizeHist = False
    if image is None:
        print("Open Error.Image is empty.")
        return
    pyramid = 0.2
    if info['pyramid'] is not None:
        pyramid = info['pyramid']
    src = cv2.resize(image, (0, 0), fx=pyramid, fy=pyramid)
    rgb_src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    # plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=inc(), title="Src")
    # A. The Template Image Processing
    src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    # src = cv2.medianBlur(src, ksize=9)

    # to make image more contrast and obvious by equalizing histogram
    # if doEqualizeHist:
    #     src = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    #     src[:, :, 0] = cv2.equalizeHist(src[:, :, 0])
    #     src = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)
    # # plot.subImage(src=src, index=++plot_img_index, title="Src")

    # plot.subImage(src=cv2.cvtColor(src, cv2.COLOR_BGR2RGB), index=inc(), title="Blurred")
    canny = cv2.Canny(src, 75, 75 * 2, edges=None)
    # calculate edge by Sobel operator
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    if doEqualizeHist:
        gray = cv2.equalizeHist(gray)
    # plot.subImage(src=gray, index=inc(), title='Gray', cmap='gray')
    # usa a large structure element to fix high light in case otust image segmentation error.
    structuring_element = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(101, 101))
    gray = cv2.morphologyEx(src=gray, op=cv2.MORPH_BLACKHAT, kernel=structuring_element)
    # plot.subImage(src=gray, index=inc(), title='BlackHap', cmap='gray')
    # B. Edge Detection
    grad_x = cv2.Sobel(src=gray, dx=1, dy=0, ddepth=cv2.CV_8UC1, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src=gray, dx=0, dy=1, ddepth=cv2.CV_8UC1, borderType=cv2.BORDER_DEFAULT)
    cv2.convertScaleAbs(grad_x, grad_x)
    cv2.convertScaleAbs(grad_y, grad_y)
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    # plot.subImage(cmap='gray', src=grad, title="grad", index=inc())
    grad = cv2.GaussianBlur(grad, (3, 3), sigmaX=0, sigmaY=0)
    # plot.subImage(src=grad, index=inc(), title='BlurredGrad', cmap='gray')
    # get binarization image by Otsu'algorithm
    # ret, ostu = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # adaptive = cv1.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # threshold = ostu
    canny = cv2.Canny(src, 75, 75 * 2)
    # # plot.subImage(cmap='gray', src=gray, title='gray', index=inc())
    # # plot.subImage(cmap='gray', src=grad_x, title="GradX", index=inc())
    # # plot.subImage(cmap='gray', src=grad_y, title="grady", index=inc())
    # # plot.subImage(cmap='gray', src=otsu, title="otsu", index=inc())
    # # plot.subImage(cmap='gray', src=adaptive, title="adaptive", index=inc())
    # plot.subImage(cmap='gray', src=canny, title="Canny", index=inc())
    dilate_kernel = cv2.getStructuringElement(ksize=(5, 5), shape=cv2.MORPH_ELLIPSE)
    erode_kernel = cv2.getStructuringElement(ksize=(3, 3), shape=cv2.MORPH_ELLIPSE)
    # # plot.subImage(src=ed_canny, index=inc(), title='EDCanny', cmap='gray')
    # threshold = cv2.dilate(threshold, dilate_kernel)
    # threshold = cv2.erode(threshold, dilate_kernel)
    # canny = cv2.dilate(canny, dilate_kernel)
    dilate_canny = canny.copy()
    # plot.subImage(src=canny, index=inc(), title='DilatedCanny', cmap='gray')
    canny = cv2.dilate(canny, dilate_kernel)
    canny = cv2.erode(canny, erode_kernel)
    zhang_thinning = cv2.ximgproc.thinning(canny, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # guo_thinning = cv2.ximgproc.thinning(canny, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    # plot.subImage(cmap='gray', src=zhang_thinning, title='ZhangeThinning', index=inc())
    # # plot.subImage(cmap='gray', src=guo_thinning, title='GuoThinning', index=inc())
    # cv2.createTrackbar("Kernel:", window_name, 1, 20, dilate_erode)
    # cv2.imshow(window_name, otsu)
    # # plot.subImage(cmap='gray', src=threshold, title='DilateAndErodeThresh', index=inc())
    # plot.subImage(cmap='gray', src=canny, title='DilateAndErodeCanny', index=inc())
    # find contours ,at least included all lines
    img, contours, hierarchy = cv2.findContours(canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    # filter some large contours, the pixel number of scale line should be small enough.
    # and the algorithm will find the pixel belong to the scale line as we need.
    contours = [c for c in contours if len(c) < 40]
    ## Draw Contours
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    cv2.drawContours(src, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
    prasan_iteration = rasan.getIteration(0.7, 0.3)
    avg_circle = np.zeros(3, dtype=np.float64)
    avg_fit_num = 0
    dst_threshold = 35
    period_rasanc_time = 100  # 每趟rasanc 的迭代次数,rasanc内置提前终止的算法
    iter_time = 5  # 启动rasanc拟合算法的固定次数
    hit_time = 0  # 成功拟合到圆的次数lot.subImage(src=src, title='Contours', index=inc())
    # C. Figuring out Centroids of the Scale Lines
    theta = 0
    start_value = info['startValue']
    total = info['totalValue']
    start_ptr = info['startPoint']
    end_ptr = info['endPoint']
    ptr_resolution = info['ptrResolution']
    if ptr_resolution is None:
        ptr_resolution = 15
    assert start_value is not None
    assert start_ptr is not None
    assert end_ptr is not None
    assert total is not None
    start_ptr = cvtPtrDic2D(start_ptr)
    end_ptr = cvtPtrDic2D(end_ptr)
    center = 0
    radius = 0
    # 使用拟合方式求表盘的圆,进而求出圆心
    if info['enableFit']:
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
    # dilate_canny = cleanNoisedRegions(dilate_canny, info, src)
    zhang_thinning = cleanNoisedRegions(zhang_thinning, info, src.shape)
    dilate_canny = cleanNoisedRegions(dilate_canny, info, src.shape)
    # 用直线Mask求指针区域
    hlt = np.array([center[0] + radius, center[1]])  # 通过圆心的水平线与圆的右交点
    # start_degree = math.degrees(AngleFactory.calAngleClockwise(hlt, start_ptr, center))
    # end_degree = math.degrees(AngleFactory.calAngleClockwise(end_ptr, hlt, center))
    # 计算夹角
    start_radians = AngleFactory.calAngleClockwise(hlt, start_ptr, center)
    end_radians = AngleFactory.calAngleClockwise(end_ptr, hlt, center)
    # print("Start degree:", start_degree)
    # print("End degree:", end_degree)
    pointer_mask, theta, line_ptr = pointerMaskByLine(dilate_canny, center, radius, start_radians, end_radians,
                                                      patch_degree=0.5,
                                                      ptr_resolution=ptr_resolution)

    # 求始点与水平线
    res = AngleFactory.calPointerValueByPoint(startPoint=start_ptr, endPoint=end_ptr,
                                              centerPoint=center,
                                              point=cv2PtrTuple2D(line_ptr), startValue=start_value,
                                              totalValue=total)
    print(res)
    # plot.subImage(src=pointer_mask, index=inc(), title='PointerMask', cmap='gray')
    plot.subImage(src=cv2.bitwise_or(dilate_canny, pointer_mask), index=inc(), title='Pointer', cmap='gray')
    # drawDialFeature(center, end_ptr, radius, rgb_src, start_ptr)
    plot.show()


def drawDialFeature(center, end_ptr, radius, rgb_src, start_ptr):
    """
    绘图功能函数，绘出表盘的大致区域
    :param center:
    :param end_ptr:
    :param radius:
    :param rgb_src:
    :param start_ptr:
    :return:
    """
    fitted_circle_img = drawDial((center[0], center[1]), radius, rgb_src)
    cv2.circle(img=fitted_circle_img, center=(start_ptr[0], start_ptr[1]),
               color=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
               radius=4, thickness=2)
    cv2.circle(img=fitted_circle_img, center=(end_ptr[0], end_ptr[1]),
               color=(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
               radius=4, thickness=2)
    # plot.subImage(src=fitted_circle_img, index=inc(), title='FittedCircle')


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
    if info['noisedRegion'] is not None:
        region_roi = info['noisedRegion']
        for roi in region_roi:
            mask = cv2.bitwise_not(np.zeros(shape=(shape[0], shape[1]), dtype=np.uint8))
            mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 0
            src = cv2.bitwise_and(src, mask)
        # plot.subImage(src=src, index=inc(), title='CleanNoisedRegion', cmap='gray')
    return src


def drawDial(center, radius, rgb_src):
    fitted_circle_img = rgb_src.copy()
    cv2.circle(fitted_circle_img, center, radius,
               color=(0, 0, 255),
               thickness=2,
               lineType=cv2.LINE_AA)
    cv2.circle(fitted_circle_img, center, radius=6, color=(0, 0, 255), thickness=2)
    # # plot.subImage(src=fitted_circle_img, index=inc(), title='Circle')
    return fitted_circle_img


# patch_degree = 5
# mask_res = []
# areas = []
# patch_index = 0
# pointerMaskBySector(areas, canny, center, patch_degree, patch_index, radius)
# mask_res = sorted(mask_res, key=lambda r: r[1], reverse=True)
# print("Max Area: ", mask_res[0][1])
# print("Degree: ", patch_degree * mask_res[0][0])
# for res in mask_res[0:5]:
#    index = inc()
#    # plot.subImage(src=areas[res[0]], index=index, title='Mask Res :' + str(index), cmap='gray')


def figureOutDialCircleByScaleLine(contours, dst_threshold, iter_time,
                                   period_rasanc_time):
    """
    无圆心、半径标定情况，根据刻度线拟合出表盘的圆模型
    :param contours: 刻度轮廓
    :param dst_threshold: 被视为inliers的阈值
    :param iter_time: 执行rasanc算法的次数
    :param period_rasanc_time: 每趟rasanc 的迭代次数
    :return: 拟合得到的圆心、半径
    """
    avg_circle = np.array([0, 0])
    avg_fit_num = 0
    hit_time = 0
    centroids = []
    for contour in contours:
        mu = cv2.moments(contour)
        if mu['m00'] != 0:
            centroids.append((mu['m10'] / mu['m00'], mu['m01'] / mu['m00']))
    # for centroid in centroids:
    #     # rgb_src[int(centroid[0]), int(centroid[1])] = (0, 255, 0)
    #     r = np.random.randint(0, 256)
    #     g = np.random.randint(0, 256)
    #     b = np.random.randint(0, 256)

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
    center = (np.int(avg_circle[0]), np.int(avg_circle[1]))
    radius = np.int(avg_circle[2])
    if avg_fit_num > len(centroids) / 2:
        return center, radius
    else:
        print("Fitting Circle Failed.")
        return (0, 0), 0


def extractLines(gray):
    """
    HoughLine的随机方法、全参数空间方法找直线方程
    :param gray:输入的灰度图
    :return:
    """
    p_lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=5, minLineLength=1, maxLineGap=10)
    lines = cv2.HoughLines(gray, 1, np.pi / 180, threshold=5)
    p_src_lines = np.zeros(shape=(gray.shape[0], gray[1], 3), dtype=np.uint8)
    src_lines = np.zeros(shape=(gray.shape[0], gray[1], 3), dtype=np.uint8)
    for line in p_lines[0]:
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        cv2.line(p_src_lines, (line[0], line[1]), (line[2], line[3]), color=(r, g, b), thickness=1)
        cv2.circle(p_src_lines, (line[0], line[1]), radius=4, color=(r, g, b), thickness=1)
        cv2.circle(p_src_lines, (line[2], line[3]), radius=4, color=(r, g, b), thickness=1)
    for line in lines[0]:
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        cb = np.random.randint(0, 256)
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        ptr1 = (np.int(x0 + 1000 * (-b)), np.int(y0 + 1000 * a))
        ptr2 = (np.int(x0 - 1000 * (-b)), np.int(y0 - 1000 * a))
        cv2.line(src_lines, ptr1, ptr2, color=(r, g, cb), thickness=1)
        cv2.circle(src_lines, ptr1, radius=4, color=(r, g, cb), thickness=1)
        cv2.circle(src_lines, ptr2, radius=4, color=(r, g, cb), thickness=1)
    return lines[0], p_lines[0], src_lines, p_src_lines


def pointerMaskBySector(areas, gray, center, patch_degree, radius):
    """
    用扇形遮罩的方法求直线位置。函数接受一个已经被灰度/二值化的图，图中默认保留了比较清晰的指针轮廓，
    该方法将圆的表盘分成360 / patch_degree 个区域，每个区域近似一个扇形，计算每个扇形面积的灰度和，
    然后在所有扇形区域中取出面积最大的那一个，如果处理预处理妥当，指针应该位于灰度值最大的区域.算法的
    精度决定于patch_degree的大小
    :param areas: 每个遮罩取出的区域
    :param gray: 灰度图
    :param center:圆的中心
    :param patch_degree:每个扇形区域所占的角度,该值越小，产生的遮罩越多，获取到的区域越细小
    :param radius: 圆的半径
    :return: 以(index,sum)形式组织的有序列表，index是扇形递增的序号，即每个扇形所在的区域向量与水平线夹角为index * patch_degree度
    """
    mask_res = []
    patch_index = 0
    masks, mask_centroids = ds.buildCounterClockWiseSectorMasks(center, radius, gray.shape, patch_degree,
                                                                (255, 0, 0),
                                                                reverse=True)
    for mask in masks:
        and_mask = cv2.bitwise_and(mask, gray)
        areas.append(and_mask)
        # mask_res.append((patch_index, np.sum(and_mask), and_mask))
        mask_res.append((patch_index, np.sum(and_mask)))
        patch_index += 1
    mask_res = sorted(mask_res, key=lambda r: r[1], reverse=True)
    return mask_res, mask_res[0][1] * patch_degree


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
    # # plot.subImage(cmap='gray', src=src1, title='Src1', index=inc())
    # plot.subImage(cmap='gray', src=src2, title='Src2', index=inc())
    plot.plot(hist1, index=inc(), title="Hist1")
    plot.plot(hist2, index=inc(), title="Hist2")
    # plot.subImage(cmap='gray', src=equalizedSrc1, title="EqualizedSrc1", index=inc())
    # plot.subImage(cmap='gray', src=equalizedSrc2, title="EqualizedSrc2", index=inc())
    plot.plot(equalizedHist1, index=inc(), title='EqualizedHist1')
    plot.plot(equalizedHist2, index=inc(), title='EqualizedHist2')
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(41, 41))
    black_hat = cv2.morphologyEx(src1, kernel=kernel, op=cv2.MORPH_BLACKHAT)
    # plot.subImage(cmap='gray', src=black_hat, title='TopTrans', index=inc())
    # cv2.imshow("hist1",hist1)
    # cv2.imshow("hist2", src2)
    # cv2.waitkey(0)
    # # plot.subImage(cmap='gray', src=hist_np1, title="histnp1", index=inc())
    # # plot.subImage(cmap='gray', src=hist_np2, title="histnp2", index=inc())


def pointerMaskByLine(src, center, radius, radians_low, radians_high, patch_degree=1.0, ptr_resolution=5,
                      low_ptr_color=np.array([0, 0, 221]), up_ptr_color=np.array([180, 30, 255])):
    """
    接收一张预处理过的二值图（默认较完整保留了指针信息），从通过圆心水平线右边的点开始，连接圆心顺时针建立直线遮罩，取出遮罩范围下的区域,
    计算对应区域灰度和，灰度和最大的区域即为指针所在的位置。直线遮罩的粗细程度、搜索的梯度决定了算法侦测指针的细粒度。该算法适合搜索指针形状
    为直线的仪表盘，原理与@pointerMaskBySector类似。
    :param low_ptr_color: 指针的HSV颜色空间的下界
    :param up_ptr_color:  指针的HSV颜色空间的上界
    :param radians_low:圆的搜索范围(弧度制表示)
    :param radians_high:圆的搜索范围(弧度制表示)
    :param src: 二值图
    :param center: 刻度盘的圆心
    :param radius: 圆的半径
    :param patch_degree:搜索梯度，默认每次一度
    :param ptr_resolution: 指针的粗细程度
    :return: 指针遮罩、直线与圆相交的点
    """
    _shape = src.shape
    img = src.copy()
    # 弧度转化为角度值
    low = math.degrees(radians_low)
    high = math.degrees(radians_high)
    # _img1 = cv2.erode(_img1, kernel3, iterations=1)
    # _img1 = cv2.dilate(_img1, kernel3, iterations=1)
    # 157=pi/2*100
    mask_info = []
    max_area = 0
    best_theta = 0
    iteration = np.abs(int((high - low) / patch_degree))
    for i in range(iteration):
        pointer_mask = np.zeros([_shape[0], _shape[1]], np.uint8)
        # theta = float(i) * 0.01
        theta = float(i * patch_degree / 180 * np.pi)
        y1 = int(center[1] - np.sin(theta) * radius)
        x1 = int(center[0] + np.cos(theta) * radius)
        # cv2.circle(black_img, (x1, y1), 2, 255, 3)
        # cv2.circle(black_img, (item[0], item[1]), 2, 255, 3)
        cv2.line(pointer_mask, (center[0], center[1]), (x1, y1), 255, ptr_resolution)
        and_img = cv2.bitwise_and(pointer_mask, img)
        not_zero_intensity = cv2.countNonZero(and_img)
        mask_info.append((not_zero_intensity, theta))
        # if not_zero_intensity > mask_intensity:
        #     mask_intensity = not_zero_intensity
        #     mask_theta = theta
        # imwrite(dir_path+'/2_line1.jpg', black_img)
    mask_info = sorted(mask_info, key=lambda m: m[0], reverse=True)
    # thresh = mask_info[0][0] / 30
    # over_index = 1
    # sum = thresh
    # for info in mask_info[1:]:
    #     if mask_info[0][0] - info[0] > thresh:
    #         break
    #     over_index += 1
    best_theta = mask_info[0][1]
    pointer_mask = np.zeros([_shape[0], _shape[1]], np.uint8)
    y1 = int(center[1] - np.sin(best_theta) * radius)
    x1 = int(center[0] + np.cos(best_theta) * radius)
    cv2.line(pointer_mask, (center[0], center[1]), (x1, y1), 255, ptr_resolution)
    #
    # black_img1 = np.zeros([_shape[0], _shape[1]], np.uint8)
    # r = item[2]-20 if item[2]==_heart[1][2] else _heart[1][2]+ _heart[0][1]-_heart[1][1]-20
    # y1 = int(item[1] - math.sin(mask_theta) * (r))
    # x1 = int(item[0] + math.cos(mask_theta) * (r))
    # cv2.line(black_img1, (item[0], item[1]), (x1, y1), 255, 7)
    # src = cv2.subtract(src, line_mask)
    # img = cv2.subtract(img, line_mask)
    best_theta = 180 - best_theta * 180 / np.pi
    if best_theta < 0:
        best_theta = 360 - best_theta
    return pointer_mask, best_theta, (x1, y1)


def demarcate_roi(img_dir):
    # ROI 选择\
    src = cv2.resize(cv2.imread(img_dir), (0, 0), fx=0.2, fy=0.2)
    regions = roiutil.selectROI(src)
    print(regions)


def reg_ptr(img_dir, config):
    img = cv2.imread(img_dir)
    file = open(config)
    info = json.load(file)
    assert info is not None
    recognizePointerInstrument(img, info)


if __name__ == '__main__':
    # reg_ptr('image/SF6/IMG_7640.JPG', 'config/pressure_1.json')
    reg_ptr('image/SF6/IMG_7666.JPG', 'config/otg_1.json')
    # demarcate_roi('image/SF6/IMG_7666.JPG')
