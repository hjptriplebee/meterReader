from Common import *


def readBlenometerStatus(image, info):
    print("Blenometer Reader called!!!")
    if image is None:
        print("Resolve Image Error.Inupt image is Empty.")
        return
    return checkBelometerUpAndDownStatus(image, info)

# test interface

# if __name__ == '__main__':
# src = cv2.imread('image/IMG_7610.JPG')
# cv2.imshow("Imge",src)
#    res1 = readBlenometerStatus(cv2.imread('image/IMG_7612.JPG'), None)
#    print(res1)
#    info = {"name": "Belometer1"}
#    res2 = readBlenometerStatus(cv2.imread('image/IMG_7612.JPG'), info)
#    print(res2)
