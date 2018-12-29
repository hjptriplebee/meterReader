import cv2
import numpy as np


def drawApproSectorMask(dst, begin_point, end_point, center, color, thickness, lineType):
    """
    draw sector mask
    :param dst: 
    :param begin_point: 
    :param end_point: 
    :param center: 
    :param color: 
    :param thickness: 
    :param lineType: 
    :return:sector shape masks 
    """
    x = (begin_point[0] + end_point[0] + center[0]) / 3
    y = (begin_point[1] + end_point[1] + center[1]) / 3
    sector_center = (np.int64(x), np.int64(y))
    mask = np.zeros((dst.shape[0] + 2, dst.shape[1] + 2, 1), dtype=np.uint8)
    begin_point = (np.int64(begin_point[0]), np.int64(begin_point[1]))
    end_point = (np.int64(end_point[0]), np.int64(end_point[1]))
    center = (np.int64(center[0]), np.int64(center[1]))
    cv2.line(dst, begin_point, end_point, color=color, thickness=thickness, lineType=lineType)
    cv2.line(dst, begin_point, center, color=color, thickness=thickness, lineType=lineType)
    cv2.line(dst, center, end_point, color=color, thickness=thickness, lineType=lineType)
    cv2.floodFill(dst, mask=mask, seedPoint=sector_center, newVal=color, loDiff=0, upDiff=0)
    return x, y


def buildCounterClockWiseSectorMasks(center, radius, shape, patch_degree, color, reverse=False, masks=None):
    iteration = int(360 / patch_degree)
    reverse_flag = 1
    centorids = []
    if masks is None:
        masks = []
    if reverse:
        reverse_flag = -1
    for i in range(iteration):
        mask = np.zeros(shape, dtype=np.uint8)
        x_theta1 = np.cos(np.pi - (i * patch_degree / 180 * np.pi))
        y_theta1 = np.sin(np.pi - (i * patch_degree / 180 * np.pi)) * reverse_flag
        x_theta2 = np.cos(np.pi - ((i + 1) * patch_degree / 180 * np.pi))
        y_theta2 = np.sin(np.pi - ((i + 1) * patch_degree / 180 * np.pi)) * reverse_flag
        x1 = x_theta1 * radius + center[0]
        y1 = y_theta1 * radius + center[1]
        x2 = x_theta2 * radius + center[0]
        y2 = y_theta2 * radius + center[1]
        centroid_x, centroid_y = drawApproSectorMask(mask, (x1, y1), (x2, y2), center, color, thickness=2,
                                                     lineType=cv2.FILLED)
        masks.append(mask)
        centorids.append((np.uint64(centroid_x), np.uint64(centroid_y)))

    return masks, centorids
