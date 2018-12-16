import numpy as np
import math
import PlotUtil as plot
import cv2


def randomSampleConsensus(data, best_circle=None, max_iterations=None, dst_threshold=None, fit_num_thresh=0):
    assert len(data) > 0, 'Input observable data is empty.'
    assert max_iterations is not None, 'Iteration should be specified.'
    assert dst_threshold is not None, 'Threshold should be specified.'
    max_fit_num = 0
    best_error = 0.0
    best_consensus_pointers = []
    data_size = len(data)
    for i in range(0, max_iterations):
        idx1 = np.random.randint(low=0, high=data_size)
        idx2 = np.random.randint(low=0, high=data_size)
        idx3 = np.random.randint(low=0, high=data_size)
        # 假设样本遵循均匀分布，圆的三个点是独立选择的;maxIteration是选取不重复点的上限
        if idx1 == idx2:
            continue
        if idx1 == idx3:
            continue
        if idx3 == idx2:
            continue
        consensus_pointers = [data[idx1], data[idx2], data[idx3]]
        # 三点确定一个圆
        circle = getCircle(data[idx1], data[idx2], data[idx3])
        # 求剩余点的拟合程度
        current_fit_num = fitNum(data, circle, dst_threshold, [idx1, idx2, idx3], consensus_pointers)
        # 如果当前得到的inliers数目超过了拟合的阈值，则认为找到了一个更好的模型
        if current_fit_num > max_fit_num:
            # this_error = current_fit_num / data_size
            if current_fit_num > fit_num_thresh:
                max_fit_num = current_fit_num
                best_circle = circle
                best_consensus_pointers = consensus_pointers
    if max_fit_num == 0:
        print("Could not fit a circle from data.")
    return best_circle, max_fit_num, best_consensus_pointers


def fitNum(pointers, circle, threshold, inliers, consensus_pointers=None):
    if consensus_pointers is None:
        consensus_pointers = []
    num = 0
    data_size = len(pointers)
    inliers_size = len(inliers)
    center = (circle[0], circle[1])
    radius = circle[2]
    for i in range(0, data_size):
        is_inliers = False
        for j in range(0, inliers_size):
            if i == inliers[j]:
                is_inliers = True
        if is_inliers:
            continue
        dis = distance(pointers[i], center) - radius
        print(dis)
        if math.fabs(distance(pointers[i], center) - radius) < threshold:
            num += 1
            consensus_pointers.append(pointers[i])
    return num


def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))


def getCircle(pointer1, pointer2, pointer3):
    """
    三点确定一个圆
    :param pointer1:
    :param pointer2:
    :param pointer3:
    :return:
    """
    circle = np.zeros(shape=(3, 1), dtype=np.float16)
    x1 = float(pointer1[0])
    x2 = float(pointer2[0])
    x3 = float(pointer3[0])

    y1 = float(pointer1[1])
    y2 = float(pointer2[1])
    y3 = float(pointer3[1])

    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0
    f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0
    det = b * c - a * d

    if math.fabs(det) < 1e-5:
        circle = [0, 0, -1]
        return circle
    circle[0] = -(d * e - b * f) / det
    circle[1] = -(a * f - c * e) / det
    circle[2] = math.sqrt((circle[0] - x1) * (circle[0] - x1) + (circle[1] - y1) * (circle[1] - y1))
    return circle

# if __name__ == '__main__':
#     samples_num = 100
#     R = 100
#     t = np.random.random(size=samples_num) * 2 * np.pi - np.pi
#     # x = np.uint8(R * np.cos(t) + R / 2)
#     # y = np.uint8(R * np.sin(t) + R / 2)
#     pointers = []
#     for index in range(len(t)):
#         x = np.uint8(np.cos(t[index]) * R + R)
#         y = np.uint8(np.sin(t[index]) * R + R)
#         pointers.append((x, y))
#     # for i in i_set:
#     #     len = np.sqrt(np.random.random())
#     #     x[i] = x[i] * len
#     #     y[i] = y[i] * len
#     print(pointers)
#     circle_img = np.zeros((2 * R, 2 * R, 3), dtype=np.uint8) + 255
#     for p in pointers:
#         b = np.random.randint(low=0, high=255)
#         g = np.random.randint(low=0, high=255)
#         r = np.random.randint(low=0, high=255)
#         print((b, g, r))
#         circle_img[p[0]][p[1]] = (b, g, r)
#     # cv2.circle(circle_img, center=(R, R), radius=R, color=(0, 255, 0), thickness=1)
#     # plot.subImage(src=circle_img, index=1, title='Circle')
#     # plot.show()
