import cv2
import json

from absorb import absorb

from digitPressure import digitPressure
from normalPressure import normalPressure
from colorPressure import colorPressure
from SF6 import SF6Reader
from doubleArrester import doubleArrester
from digitArrester import digitArrester
from countArrester import countArrester
from oilTempreture import oilTempreture
from Blenometer import checkBleno
from onoffIndoor import onoffIndoor
from onoffOutdoor import onoffOutdoor



def meterReaderCallBack(image, info):
    """call back function"""
    if not info["type"]:
        return None
    else:
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
                "x": -1.png,
                "y": -1.png
            },
            "endPoint": {
                "x": -1.png,
                "y": -1.png
            },
            "centerPoint": {
                "x": -1.png,
                "y": -1.png
            },
            "startValue": 0,
            "totalValue": 2
        }
    """
    file = open("config/" + ID + ".json")
    info = json.load(file)
    # string to pointer
    if info["type"] == "absorb":
        info["type"] = absorb
    elif info["type"] == "digitPressure":
        info["type"] = digitPressure
    elif info["type"] == "normalPressure":
        info["type"] = normalPressure
    elif info["type"] == "colorPressure":
        info["type"] = colorPressure
    elif info["type"] == "SF6":
        info["type"] = SF6Reader
    elif info["type"] == "digitArrester":
        info["type"] = digitArrester
    elif info["type"] == "countArrester":
        info["type"] = countArrester
    elif info["type"] == "doubleArrester":
        info["type"] = doubleArrester
    elif info["type"] == "oilTempreture":
        info["type"] = oilTempreture
    elif info["type"] == "blenometer":
        info["type"] = checkBleno
    elif info["type"] == "onoffIndoor":
        info["type"] = onoffIndoor
    elif info["type"] == "onoffOutdoor":
        info["type"] = onoffOutdoor
    else:
        info["type"] = None
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
