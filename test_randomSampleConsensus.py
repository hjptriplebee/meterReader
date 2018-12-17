from unittest import TestCase
import unittest
import numpy as np
import RasancFitCircle as rasanc
import cv2
import PlotUtil as plot


def generateCirclePointers(center, radius, theta, noise_range, pointers=None):
    """
    :param center:圆的圆心
    :param radius:圆的半径
    :param pointers:生成的数据点
    :param noise_range: 噪点分布范围,即RANSANC中提及的Outliers
    :param theta: theta 向量
    :return: pointers
    """
    if pointers is None:
        pointers = []
    for index in range(len(theta)):
        # 将极坐标转换为中心在(radius,radius)的圆坐标
        assert center[0] > 0 and center[1] > 0
        noise_range_len = (-1 + 2 * np.random.random()) * (noise_range[1] - noise_range[0]) + radius
        # 将所有点移动到图像的中心
        x = np.uint8(np.cos(theta[index]) * noise_range_len + center[0])
        y = np.uint8(np.sin(theta[index]) * noise_range_len + center[1])
        pointers.append((x, y))
    return pointers


def generateCirclePointersInsideCircle(center, radius, theta, pointers=None):
    """
    生成在圆内均匀分布的点
    :param center:圆的圆心
    :param radius:圆的半径
    :param pointers: 生成的数据点
    :param theta: theta向量
    :return:pointers
    """
    if pointers is None:
        pointers = []
    for index in range(len(theta)):
        # 生成0~radius之间的长度线段
        less_radius = np.random.random() * radius
        x = np.uint8(np.cos(theta[index]) * less_radius + center[0])
        y = np.uint8(np.sin(theta[index]) * less_radius + center[1])
        pointers.append((x, y))
    return pointers


def randomTheta(size):
    return np.random.random(int(size)) * 2 * np.pi - np.pi


def colorfulPointers(shape, pointers, circle_img=None):
    """
    给所有点上色
    :param shape: 图片的大小和信道
    :param pointers:
    :param circle_img:
    :return:circle_img
    """
    if circle_img is None:
        circle_img = np.zeros(shape, dtype=np.uint8) + 255
    for p in pointers:
        b = np.random.randint(low=0, high=255)
        g = np.random.randint(low=0, high=255)
        r = np.random.randint(low=0, high=255)
        circle_img[p[0]][p[1]] = (b, g, r)
    return circle_img


def generatePointersMixedOutliers(img_center, inliers_num, noise_range, outliers_ratio, radius):
    inliers_theta = randomTheta(inliers_num)
    outliers_theta = randomTheta(inliers_num / (1 - outliers_ratio) * outliers_ratio)
    inliers = generateCirclePointers(center=img_center, radius=radius, theta=inliers_theta, noise_range=(0, 0))
    outliers = generateCirclePointers(center=img_center, radius=radius, theta=outliers_theta,
                                      noise_range=noise_range)
    pointers = inliers + outliers
    np.random.shuffle(pointers)
    return pointers


class TestRandomSampleConsensus(TestCase):
    def test_get_generation(self):
        iteration = rasanc.getIteration(0.99, 0.5)
        print(iteration)

    def test_randomSampleConsensus_perfect_fitting(self):
        """
        在圆上所有点的测试样例
        :return:
        """
        inliers_num = 100
        radius = 100
        img_center = (2 * radius, 2 * radius)
        circle_img_shape = (img_center[0] * 2, img_center[1] * 2, 3)
        noise_range = (0.3 * radius, radius * 0.5)
        outliers_ratio = 0.3
        pointers = generatePointersMixedOutliers(img_center, inliers_num, noise_range, outliers_ratio, radius)
        circle_img = colorfulPointers(circle_img_shape, pointers)
        plot.subImage(src=circle_img, index=1, title='Circle')
        best_circle, max_fit_num, best_consensus_pointers = rasanc.randomSampleConsensus(data=pointers,
                                                                                         max_iterations=100,
                                                                                         dst_threshold=20,
                                                                                         inliers_threshold=len(
                                                                                             pointers) * 0.5)
        self.assertTrue(len(best_consensus_pointers) > 0)
        print("Best circle:", best_circle)
        print("Max fit number of pointers:", max_fit_num)
        print("Best consensus pointers:", best_consensus_pointers)

        cv2.circle(circle_img, center=(best_circle[0], best_circle[1]), radius=best_circle[2], color=(0, 255, 0),
                   thickness=1,
                   lineType=cv2.LINE_AA)
        plot.subImage(src=circle_img, index=2, title='Fitted Circle')
        plot.show()

    def test_randomSampleConsensus_uniform_fitting(self):
        """
        测试分布在一个圆内的且符合均匀分布的点
        :return:
        """
        samples_num = 200
        radius = 100
        theta = randomTheta(samples_num)
        circle_shape = (radius * 2, radius * 2, 3)
        pointers_in_circle = generateCirclePointers(radius=radius, theta=theta)
        pointers_inside_circle = generateCirclePointersInsideCircle(radius=radius, theta=theta)
        # mix up two kinds of pointers,one is on the circle edge,the another in inside the circle.
        pointers = pointers_in_circle + pointers_inside_circle
        np.random.shuffle(pointers)
        img = colorfulPointers(shape=circle_shape, pointers=pointers)
        plot.subImage(src=img, index=1, title='Img')
        best_circle, max_fit_num, best_consensus_pointers = rasanc.randomSampleConsensus(data=pointers,
                                                                                         max_iterations=100,
                                                                                         dst_threshold=10,
                                                                                         inliers_threshold=len(
                                                                                             pointers) / 2)
        print("Best circle:", best_circle)
        print("Max fit number of pointers:", max_fit_num)
        print("Best consensus pointers:", best_consensus_pointers)
        cv2.circle(img, center=(best_circle[0], best_circle[1]), radius=best_circle[2], color=(0, 0, 255), thickness=1,
                   lineType=cv2.LINE_AA)
        plot.subImage(src=img, index=2, title='Fitting Circle')
        plot.show(save=True)
        self.assertTrue(len(best_consensus_pointers) > 0)
# def suite():
#     suite = unittest.TestSuite()
#     suite.addTest(TestRandomSampleConsensus('test_randomSampleConsensus'))
#     return suite
#
#
# if __name__ == '__main__':
#     runner = unittest.TextTestRunner()
#     test_suite = suite()
#     runner.run(test_suite)
