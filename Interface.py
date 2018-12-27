import cv2
import json
from Blenometer import checkBleno
from SF6 import SF6Reader
from Remember import remember
from youwen import youwen
from pressure import pressure
from absorb import absorb
from switch import switch
from bileiqi1 import bileiqi1
# from bileiqi2 import bileiqi2


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
    file = open("config/" + ID + ".json")
    info = json.load(file)
    # string to pointer
    if info["type"] == "SF6":
        info["type"] = SF6Reader
    elif info["type"] == "youwen":
        info["type"] = youwen
    elif info["type"] == "pressure":
        info["type"] = pressure
    elif info["type"] == "bileiqi1":
        info["type"] = bileiqi1
    # elif info["type"] == "bileiqi2":
    #     info["type"] = bileiqi2
    elif info["type"] == "blenometer":
        info["type"] = checkBleno
    elif info["type"] == "absorb":
        info["type"] = absorb
    elif info["type"] == "switch":
        info["type"] = switch
    elif info["type"] == "remember":
        info["type"] = remember
    else:
        print("meter type not support!")

    info["template"] = cv2.imread("template/" + ID + ".jpg")
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
        ROI = image[y:y + h, x:x + w]
        # call back
        results[ID] = meterReaderCallBack(ROI, info)
    return results
