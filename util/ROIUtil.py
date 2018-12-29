import cv2
import numpy as np

drawing = False  # 是否开始画图
mode = True  # True：画矩形，False：画圆
start = (-1, -1)
img = np.zeros((512, 512, 3), np.uint8)
window_name = 'Image'
regions = []


def mouse_event(event, x, y, flags, param):
    global start, drawing, mode, img, regions
    current_img = img.copy()
    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(current_img, start, (x, y), (0, 255, 0), 1)
            else:
                cv2.circle(current_img, (x, y), 5, (0, 0, 255), -1)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(current_img, start, (x, y), (0, 255, 0), 1)
            width = np.abs(start[0] - x)
            height = np.abs(start[1] - y)
            smaller_ptr = (min(start[0], x), min(start[1], y))
            regions.append((smaller_ptr[0], smaller_ptr[1], width, height))
        else:
            cv2.circle(current_img, (x, y), 5, (0, 0, 255), -1)
            radius = np.sqrt(np.power(start[0] - x, 2) + np.power(start[1] - y, 2))
            regions.append((start[0], start[1], radius))
        img = current_img.copy()
    text = 'x = ' + str(x) + ',' + 'y = ' + str(y)
    cv2.putText(current_img, text, (5, current_img.shape[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow(window_name, current_img)


def selectROI(src):
    """
    选择ROI，并返回[(start.x,start.y,end.x,end.y)]形式的数组
    用于简单标定
    :param src:
    :return: regions
    """
    global img, mode
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_event)
    img = src.copy()
    cv2.imshow(window_name, img)
    while True:
        cv2.imshow(window_name, img)
        # 按下m切换模式
        if cv2.waitKey(1) == ord('m'):
            mode = not mode
        # Esc退出
        elif cv2.waitKey(1) == 27:
            break
    r = regions.copy()
    regions.clear()
    return r
