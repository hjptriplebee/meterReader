import cv2
from SF6 import SF6

def meterReaderCallBack(image, info):
    """call back function"""
    return info["type"](image, info)

def getInfo(ID):
    """
    get info from file
    :param ID: meter ID
    :return: info = {
            "distance": 10,
            "horizontal": 10,
            "vertical": 20,
            "name": "1_1",
            "type": SF6,
            "template": "template.jpg",
            "ROI": {
                "x": 200,
                "y": 200,
                "w": 1520,
                "h": 680
            },
            "startPoint": {
                "x": -1,
                "y": -1
            },
            "endPoint": {
                "x": -1,
                "y": -1
            },
            "centerPoint": {
                "x": -1,
                "y": -1
            },
            "startValue": 0,
            "totalValue": 2
        } 
    """
    info = {
        "distance": 10,
        "horizontal": 10,
        "vertical": 20,
        "name": "1_1",
        "type": SF6,
        "template": "template.jpg",
        "ROI": {
            "x": 200,
            "y": 200,
            "w": 1520,
            "h": 680
        },
        "startPoint": {
            "x": -1,
            "y": -1
        },
        "endPoint": {
            "x": -1,
            "y": -1
        },
        "centerPoint": {
            "x": -1,
            "y": -1
        },
        "startValue": 0,
        "totalValue": 2
    }
    return info


def meterReader(image, meterIDs):
    """
    global interface
    :param image: camera image
    :param meterIDs: list of meter ID
    :return: 
    """
    results = {}
    for ID in meterIDs:
        # get info from file
        info = getInfo(ID)
        # ROI extract
        x = info["ROI"]["x"]
        y = info["ROI"]["y"]
        w = info["ROI"]["w"]
        h = info["ROI"]["h"]
        ROI = image[x:x + h, y:y + w]
        # call back
        results[ID] = meterReaderCallBack(ROI, info)
    return results


image = cv2.imread("image/2018-11-20-16-22-02.jpg")

print(meterReader(image, ["1_1"]))

