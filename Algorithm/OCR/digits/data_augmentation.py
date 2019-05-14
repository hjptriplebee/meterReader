import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

'''
定义hsv变换函数：
hue_delta是色调变化比例
sat_delta是饱和度变化比例
val_delta是明度变化比例
'''
def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

'''
随机hsv变换
hue_vari是色调变化比例的范围
sat_vari是饱和度变化比例的范围
val_vari是明度变化比例的范围
'''
def random_hsv_transform(img, hue_vari=10, sat_vari=0.1, val_vari=0.1):
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)

'''
定义gamma变换函数：
gamma就是Gamma
'''
def gamma_transform(img, gamma=1.0):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

'''
随机gamma变换
gamma_vari是Gamma变化的范围[1/gamma_vari, gamma_vari)
'''
def random_gamma_transform(img, gamma_vari=2.0):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

def randomGaussian(image, mean=0.2, sigma=0.3):
    """
     对图像进行高斯噪声处理
    :param image:
    :return:
    """
    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将图像转化成数组
    img = np.asarray(image)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return np.uint8(img)

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.fromarray(image)
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    img = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    return np.array(img)


def augmentation(origin, dest):
    for sub in os.listdir(origin):
        subpath = os.path.join(origin, sub)
        destpath = os.path.join(dest, sub)
        if not os.path.exists(destpath):
            os.makedirs(destpath)
        for file in os.listdir(subpath):
            filename = os.path.join(subpath, file)
            img = cv2.imread(filename)
            cv2.imwrite(os.path.join(destpath, file[:-4]+"_origin.bmp"), img)
            # 随机hsv变换
            img_hsv = random_hsv_transform(img.copy())
            destname = os.path.join(destpath, file[:-4]+"_hsv.bmp")
            cv2.imwrite(destname, img_hsv)
            # 随机gamma变换
            img_gamma = random_gamma_transform(img.copy())
            destname = os.path.join(destpath, file[:-4] + "_gamma.bmp")
            cv2.imwrite(destname, img_gamma)
            # 对图像进行颜色抖动
            img_color = randomColor(img.copy())
            destname = os.path.join(destpath, file[:-4]+"_color.bmp")
            cv2.imwrite(destname, img_color)
            # 对图像进行高斯噪声处理
            img_gaussian = randomGaussian(img.copy())
            destname = os.path.join(destpath, file[:-4] + "_gaussian.bmp")
            cv2.imwrite(destname, img_gaussian)




origin = "dataset/rgb_train"
dest = "dataset/rgb_augmentation"
augmentation(origin, dest)