from unittest import TestCase
import tabulate as table
import unittest
import numpy as np
import RasancFitCircle as rasanc
import cv2
import PlotUtil as plot

plot_index = 0


def inc():
    global plot_index
    plot_index += 1
    return plot_index


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
        # 将以(0,0)为中心的圆变换为以center为中心
        x = np.uint32(np.cos(theta[index]) * noise_range_len + center[0])
        y = np.uint32(np.sin(theta[index]) * noise_range_len + center[1])
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
    """

    :param img_center: 要把圆绘在一副图像中方便测试算法效果，该参数指明了图像的中心，以进行平移变换
    :param inliers_num: 要产生inliers的数目
    :param noise_range: 表示outliers在圆边内外两侧分布的范围
    :param outliers_ratio: outliers占全部点数量的比率
    :param radius: 圆半径
    :return:
    """
    # theta 角,有负值
    inliers_theta = randomTheta(inliers_num)
    outliers_theta = randomTheta(inliers_num / (1 - outliers_ratio) * outliers_ratio)
    inliers = generateCirclePointers(center=img_center, radius=radius, theta=inliers_theta, noise_range=(0, 0))
    outliers = generateCirclePointers(center=img_center, radius=radius, theta=outliers_theta,
                                      noise_range=noise_range)
    pointers = inliers + outliers
    np.random.shuffle(pointers)
    return pointers


def makeRasancTestCase(circle_img_shape, img_center, inliers_num, noise_range, outliers_ratio, radius):
    pointers = generatePointersMixedOutliers(img_center, inliers_num, noise_range, outliers_ratio, radius)
    circle_img = colorfulPointers(circle_img_shape, pointers)
    # plot.subImage(src=circle_img, index=1, title='Circle')
    max_iteration = rasanc.getIteration(1 - outliers_ratio, outliers_ratio)
    inliers_threshold = len(pointers) * (1 - outliers_ratio)
    # print("Max iteration:", max_iteration)
    best_circle, max_fit_num, best_consensus_pointers = rasanc.randomSampleConsensus(data=pointers,
                                                                                     max_iterations=max_iteration,
                                                                                     dst_threshold=10,
                                                                                     inliers_threshold=inliers_threshold
                                                                                     )
    return best_circle, best_consensus_pointers, circle_img, max_fit_num, max_iteration


def formatResults(best_circle, inliers_num, outliers_ration, max_fit_num, outliers_ratio, max_iteration, best, res):
    # row = np.array((1, 6), dtype=np.float32)
    row = np.zeros(9, dtype=np.float32)
    total_num = np.round(inliers_num / (1 - outliers_ratio))
    error_ration = 1 - max_fit_num / total_num
    loss = np.sqrt(np.power(best[0] - best_circle[0], 2) + np.power(best[1] - best_circle[1], 2))
    row[0] = best_circle[0]
    row[1] = best_circle[1]
    row[2] = best_circle[2]
    row[3] = outliers_ratio
    row[4] = max_fit_num
    row[5] = total_num
    row[6] = error_ration
    row[7] = max_iteration
    row[8] = loss
    # row.append(best_circle[0])
    # row.append(best_circle[1])
    # row.append(best_circle[2])
    res.append(row)
    return res


class TestRandomSampleConsensus(TestCase):
    def test_get_generation(self):
        iteration = rasanc.getIteration(0.99, 0.5)
        print(iteration)

    def test_random_sample_consensus_circle_fitting_when_outliers_ration_increase(self):
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
        n_test_case = 3
        res = []
        headers = ['x', 'y', 'radius', 'outliers_ratio', 'max_fit_num', 'total', 'error_ration', 'max_iteration',
                   'loss']
        i = 0
        while outliers_ratio <= 0.9:
            outliers_ratio = outliers_ratio + 0.1 * i
            best_circle, best_consensus_pointers, circle_img, max_fit_num, max_iter = makeRasancTestCase(
                circle_img_shape,
                img_center, inliers_num,
                noise_range,
                outliers_ratio, radius)

            # self.assertTrue(len(best_consensus_pointers) > 0)
            print("Best circle:", best_circle)
            print("max fit number of pointers:", max_fit_num)
            print("best consensus pointers:", best_consensus_pointers)
            if len(best_consensus_pointers) > 0:
                cv2.circle(circle_img, center=(best_circle[0], best_circle[1]), radius=best_circle[2],
                           color=(0, 255, 0),
                           thickness=1,
                           lineType=cv2.LINE_AA)
                formatResults(best_circle, inliers_num, outliers_ratio, max_fit_num, outliers_ratio, max_iter,
                              img_center, res)
                plot.subImage(src=circle_img, index=inc(), title='Fitted Circle ' + str(i + 1))
            else:
                res.append([])
            i += 1
        plot.show()
        print(table.tabulate(tabular_data=res, headers=headers, tablefmt='fancy_grid'))
#     def test_randomSampleConsensus_uniform_fitting(self):
#         """
#         测试分布在一个圆内的且符合均匀分布的点
#         :return:
#         """
#         samples_num = 200
#         radius = 100
#         theta = randomTheta(samples_num)
#         circle_shape = (radius * 2, radius * 2, 3)
#         pointers_in_circle = generateCirclePointers(radius=radius, theta=theta)
#         pointers_inside_circle = generateCirclePointersInsideCircle(radius=radius, theta=theta)
#         # mix up two kinds of pointers,one is on the circle edge,the another in inside the circle.
#         pointers = pointers_in_circle + pointers_inside_circle
#         np.random.shuffle(pointers)
#         img = colorfulPointers(shape=circle_shape, pointers=pointers)
#         plot.subImage(src=img, index=1, title='Img')
#         best_circle, max_fit_num, best_consensus_pointers = rasanc.randomSampleConsensus(data=pointers,
#                                                                                          max_iterations=100,
#                                                                                          dst_threshold=10,
#                                                                                          inliers_threshold=len(
#                                                                                              pointers) / 2)
#         print("Best circle:", best_circle)
#         print("Max fit number of pointers:", max_fit_num)
#         print("Best consensus pointers:", best_consensus_pointers)
#         cv2.circle(img, center=(best_circle[0], best_circle[1]), radius=best_circle[2], color=(0, 0, 255), thickness=1,
#                    lineType=cv2.LINE_AA)
#         plot.subImage(src=img, index=2, title='Fitting Circle')
#         plot.show(save=True)
#         self.assertTrue(len(best_consensus_pointers) > 0)
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
