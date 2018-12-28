import cv2
from Common import AngleFactory,meterFinderByTemplate
from num_reading import get_value
import math
def bileiqi2(image, info):
    """
    :param image:whole image
    :param info:bileiqi2 config
    :return:
    """
    # your method
    print("bileiqi Reader called!!!")
    # template match
    #print(info["template"])
    meter = meterFinderByTemplate(image, info["template"])
    n_value,p_value=bileiqi_pointer_num_reading(meter,info["totalValue"],info["numSize"])
    return p_value,n_value
def __CalculateLineAngle(p1, p2):
	xDis = p2[0] - p1[0]
	yDis = p2[1] - p1[1]
	angle = math.atan2(yDis, xDis)
	angle = angle / math.pi *180
	return angle;
def __rotateImage(img,degree,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    heightNew = int(width * math.fabs(math.sin(degree)) + height * math.fabs(math.cos(degree)))
    widthNew = int(height * math.fabs(math.sin(degree)) + width * math.fabs(math.cos(degree)))
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt2 = list(pt2)
    pt4 = list(pt4)
    [[pt2[0]], [pt2[1]]] = np.dot(matRotation, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(matRotation, np.array([[pt4[0]], [pt4[1]], [1]]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    pt4 = (int(pt4[0]), int(pt4[1]))
    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt2[0]):int(pt4[0])]
    return imgOut

def get_num_area(image,box):
    #get number area
    imgResize=cv2.resize(image,(500,500))
    pt1=box[0]
    pt2=box[1]
    pt3=box[2]
    pt4=box[3]
    angle=__CalculateLineAngle(pt1,pt4)
    imgRotation=__rotateImage(imgResize,angle,pt1,pt2,pt3,pt4)
    return imgRotation
def bileiqi_pointer_num_reading(image,totalValue,num):
    #pointer value reading & number reading
    image=cv2.resize(image,(500,500))
    Img=image
    R=Img.shape[0]//2
    ROIMask = cv2.bitwise_not(np.zeros(Img.shape, dtype = "uint8"))
    tableEdge = 110
    cv2.circle(ROIMask, (R, R), R - tableEdge, (0, 0, 0), -1)
    Img = cv2.bitwise_or(Img, ROIMask)

    HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)

    #green start line
    lowerGreen = np.array([35, 30, 46], dtype="uint8")
    upperGreen = np.array([99, 255, 255], dtype="uint8")
    greenMask = cv2.inRange(HSV, lowerGreen, upperGreen)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    greenMask = cv2.erode(greenMask, kernel)
    greenMask = cv2.dilate(greenMask, kernel)

    _, greenContours, greenHierarchy = cv2.findContours(greenMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    greenContours = sorted(greenContours, key=lambda c: c.shape[0], reverse=True)
    greenContours = [c for c in greenContours if len(c) > 5 ]
    maxY = 0
    startPoint = []

    for c in greenContours:
        for p in c:
            if p[0][1] > maxY:
                maxY = p[0][1]
                startPoint = p[0]
    print("startPoint:",startPoint)

    #red end line
    lowerRed = np.array([156, 43, 46], dtype="uint8")
    upperRed = np.array([180, 255, 255], dtype="uint8")
    redMask1 = cv2.inRange(HSV, lowerRed, upperRed)
    lowerRed = np.array([0, 20, 46], dtype="uint8")
    upperRed = np.array([10, 255, 255], dtype="uint8")
    redMask2 = cv2.inRange(HSV, lowerRed, upperRed)
    redMask=cv2.bitwise_or(redMask1,redMask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    redMask = cv2.erode(redMask, kernel)
    redMask = cv2.dilate(redMask, kernel)

    _, redContours, redHierarchy = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    redContours = sorted(redContours, key=lambda c: c.shape[0], reverse=True)
    redContours = [c for c in redContours if len(c) > 5 ]
    maxY = 0
    endPoint = []

    for c in redContours:
        for p in c:
            if p[0][1] > maxY:
                maxY = p[0][1]
                endPoint = p[0]
    print("endPoint:",endPoint)

    #black pointer line & number area
    lowerBlack=np.array([0,0,0],dtype="uint8")
    upperBlack=np.array([180, 255, 110],dtype="uint8")
    blackMask=cv2.inRange(HSV,lowerBlack,upperBlack)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    blackMask1=cv2.erode(blackMask,kernel)
    blackMask1=cv2.dilate(blackMask1,kernel)

    _, blackContours, blackHierarchy = cv2.findContours(blackMask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blackContours = sorted(blackContours, key=lambda c: c.shape[0], reverse=True)
    blackContours = [c for c in blackContours if len(c) > 10 ]
    for c in blackContours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        break

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    blackMask=cv2.erode(blackMask,kernel)
    blackMask=cv2.dilate(blackMask,kernel)
    _, blackContours, blackHierarchy = cv2.findContours(blackMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blackContours = sorted(blackContours, key=lambda c: c.shape[0], reverse=True)
    blackContours = [c for c in blackContours if len(c) > 5 ]
    maxY = R
    pointerPoint = []
    for c in blackContours:
        for p in c:
            if p[0][1] < maxY:
                maxY = p[0][1]
                pointerPoint = p[0]

    print("pointerPoint:",pointerPoint)
    centerPoint=[R,R]
    print("center_point:",centerPoint)
    angleRange=AngleFactory.__calAngleBetweenTwoVector(startPoint-centerPoint,endPoint-centerPoint)
    angle = AngleFactory.__calAngleBetweenTwoVector(startPoint - centerPoint, pointerPoint - centerPoint)

    value = angle / angleRange * totalValue + 0.0
    print("p_value",value)

    num_img = get_num_area(image, box)
    results = get_value(num_img, num)
    return value,results


