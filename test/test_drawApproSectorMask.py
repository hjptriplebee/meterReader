from unittest import TestCase
# from DrawSector import drawApproSectorMask
import DrawSector as ds
import numpy as np
import PlotUtil as plot


class TestDrawApproSectorMask(TestCase):
    def test_drawApproSectorMask(self):
        center = (256, 256)
        radius = 256
        patch_degree = 5
        iteration = int(360 / patch_degree)
        masks = ds.buildCounterClockWiseSectorMasks(center, radius, shape=(512, 512), patch_degree=10, color=(255, 255), reverse=True)
        index = 1
        for mask in masks:
            plot.subImage(src=mask, index=index, title='Mask ' + str(index), cmap='gray')
            index += 1
        plot.show()
