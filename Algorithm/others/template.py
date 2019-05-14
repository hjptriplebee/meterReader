import cv2 as cv
import numpy as np
import os


def template(mode, img, threshold):
    # 模板图片
    # mode = cv.imread('E:/picture/mode3.jpg')
    # 目标图片
    # img = cv.imread('E:/picture/6.jpg')
    # cv.namedWindow('template', cv.WINDOW_NORMAL)
    # cv.imshow('template', mode)
    # cv.namedWindow('img', cv.WINDOW_NORMAL)
    # cv.imshow('img', img)
    #
    # methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    # 获得模板的高宽
    th, tw = mode.shape[:2]
    '''
    for md in methods:

        # 执行模板匹配
        # img：目标图片
        # mode：模板图片
        # 匹配模式
        result = cv.matchTemplate(img, mode, md)
        # 寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc

        br = (tl[0] + tw, tl[1] + th)
        # 绘制矩形边框，将匹配区域标注出来
        # img：目标图像
        # tl：矩形定点
        # br：举行的宽高
        # (0,0,255)：矩形边框颜色
        # 2：矩形边框大小
        cv.rectangle(img, tl, br, (0, 0, 255), 2)
        cv.namedWindow('match-'+ np.str(md), cv.WINDOW_NORMAL)
        cv.imshow('match-' + np.str(md), img)
    '''
    # 匹配模式
    res = cv.matchTemplate(img, mode, cv.TM_CCOEFF_NORMED)
    img2 = img.copy()
    # cv.normalize(res, res, 0, 1, cv.NORM_MINMAX)
    loc = np.where(res >= threshold)
    # center = np.zeros((len(loc), 2))
    # print(len(list(zip(*loc[::-1]))))
    for pt in zip(*loc[::-1]):
        cv.rectangle(img2, pt, ((pt[0] + tw, pt[1] + th)), (0, 0, 255), 2)
    # center[num] = pt
    # print(pt)
    # cv.line(img_rgb,(0,0),(1, 1), (255,0,0), 1)
    # cv.namedWindow('img', cv.WINDOW_NORMAL)
    # cv.imshow('img', img2)
    return img2


def cutimg(img, pt):
    imgout = img[int(pt[1]):int(pt[1]) + int(pt[3]), int(pt[0]):int(pt[0]) + int(pt[2])]
    return imgout


def getroi(center, mode):
    th, tw = mode.shape[:2]
    roi = [center[0] - tw / 2, center[1] - th / 2, tw, th]
    return roi


if __name__ == '__main__':
    # 模板图片
    mode = cv.imread('E:/tupian/light6.jpg')
    # 目标图片
    img = cv.imread('E:/tupian/14.jpg')
    cv.namedWindow('template', cv.WINDOW_NORMAL)
    cv.imshow('template', mode)
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    template(mode, img, 0.7)
    cv.waitKey(0)
