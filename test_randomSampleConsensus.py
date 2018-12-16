from unittest import TestCase
import unittest
import numpy as np
import RasancFitCircle as rasanc


class TestRandomSampleConsensus(TestCase):
    def test_randomSampleConsensus(self):
        samples_num = 100
        R = 100
        t = np.random.random(size=samples_num) * 2 * np.pi - np.pi
        # x = np.uint8(R * np.cos(t) + R / 2)
        # y = np.uint8(R * np.sin(t) + R / 2)
        pointers = []
        for index in range(len(t)):
            x = np.uint8(np.cos(t[index]) * R + R)
            y = np.uint8(np.sin(t[index]) * R + R)
            pointers.append((x, y))
        # for i in i_set:
        #     len = np.sqrt(np.random.random())
        #     x[i] = x[i] * len
        #     y[i] = y[i] * len
        circle_img = np.zeros((2 * R, 2 * R, 3), dtype=np.uint8) + 255
        for p in pointers:
            b = np.random.randint(low=0, high=255)
            g = np.random.randint(low=0, high=255)
            r = np.random.randint(low=0, high=255)
            print((b, g, r))
            circle_img[p[0]][p[1]] = (b, g, r)
        # cv2.circle(circle_img, center=(R, R), radius=R, color=(0, 255, 0), thickness=1)
        # plot.subImage(src=circle_img, index=1, title='Circle')
        # plot.show()
        best_circle, max_fit_num, best_consensus_pointers = rasanc.randomSampleConsensus(data=pointers,
                                                                                         max_iterations=40,
                                                                                         dst_threshold=50,
                                                                                         fit_num_thresh=len(
                                                                                             pointers) / 2)
        print("Best circle:", best_circle)
        print("Max fit number of pointers:", max_fit_num)
        print("Best consensus pointers:", best_consensus_pointers)
        self.assertTrue(len(best_consensus_pointers) > 0)


def suite():
    suite = unittest.TestSuite()

    suite.addTest(TestRandomSampleConsensus('test_randomSampleConsensus'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run(test_suite)
