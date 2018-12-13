from matplotlib import pyplot as plot
import cv2

PLOT_ROW = 9
PLOT_COL = 2

plot.figure(figsize=(20, 80))


def id(index):
    return PLOT_ROW * 100 + PLOT_COL * 10 + index


def subImage(src, index=0, figsize=None, plot_row=None, plot_col=None, title=None, cmap=None):
    global PLOT_ROW, PLOT_COL
    if index == 0:
        raise Exception("Index should be specified")
    if plot_row is not None:
        PLOT_ROW = plot_row
    if plot_col is not None:
        PLOT_COL = plot_col
    if figsize is not None:
        plot.figure(figsize=figsize)
    if index > PLOT_ROW * PLOT_COL:
        raise Exception("Index over plot size range.")
    if src is None:
        raise Exception("Image is None.Plot error.")
    # print(id(index))
    if index > 9:
        plot.figure(figsize=(20, 80))
        index %= 9
    plot.subplot(id(index))
    if cmap is not None:
        plot.imshow(src, cmap=cmap)
    else:
        plot.imshow(src)
    plot.title(title)


def show():
    plot.show()
