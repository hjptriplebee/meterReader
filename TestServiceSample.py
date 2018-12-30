import requests
import base64
import json
import os
import time
import cv2
import multiprocessing

from Interface import meterReader
from util.JsonModifier import JsonModifier


def startServer():
    os.system("python FlaskService.py")


def startClient(results):
    # test reader interface

    # ===========================bileiqi1 test===========================
    image = open("image/bileiqi1_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "bileiqi1"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    receive = json.loads(r.text)
    print(receive)

    if not "bileiqi1_1" in receive:
        results.append(False)
    else:
        results.append(True)

    # ===========================bileiqi2 test===========================
    # image = open("image/bileiqi2_1.jpg", "rb")
    # imageByte = base64.b64encode(image.read())
    # data = json.dumps({
    #     "image": imageByte.decode("ascii"),
    #     "imageID": "bileiqi2"
    # })
    # r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    # receive = json.loads(r.text)
    # print(receive)
    # if not "bileiqi2_1" in receive:
    #     results.append(False)
    # else:
    #     results.append(True)

    # ===========================SF6 test===========================
    image = open("image/SF6_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "SF6"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    receive = json.loads(r.text)
    print(receive)

    if not "SF6_1" in receive:
        results.append(False)
    else:
        results.append(True)

    # ===========================youwen test===========================
    image = open("image/youwen_4.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "youwen"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    receive = json.loads(r.text)
    print(receive)

    if not "youwen_4" in receive:
        results.append(False)
    else:
        results.append(True)

    # ===========================pressure test===========================
    image = open("image/pressure_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "pressure"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    receive = json.loads(r.text)
    print(receive)

    if not "pressure_1" in receive:
        results.append(False)
    else:
        results.append(True)
    # ===========================pressure2 test===========================
    image = open("image/pressure2_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "pressure2"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    receive = json.loads(r.text)
    print(receive)

    if not "pressure2_1" in receive:
        results.append(False)
    else:
        results.append(True)

    # ===========================absorb test===========================
    image = open("image/absorb_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "absorb"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    receive = json.loads(r.text)
    print(receive)

    if not "absorb_1" in receive:
        results.append(False)
    else:
        results.append(True)

    # ===========================switch test===========================
    image = open("image/switch_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "switch"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    receive = json.loads(r.text)
    print(receive)

    if not "switch_1" in receive:
        results.append(False)
    else:
        results.append(True)

    # ===========================bleno test===================================
    image = open("image/blenometer_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "blenometer"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    receive = json.loads(r.text)
    print(receive)

    if not "blenometer_1" in receive:
        results.append(False)
    else:
        results.append(True)


def codecov():
    image = cv2.imread("image/bileiqi1_1.jpg")
    receive2 = meterReader(image, ["bileiqi1_1"])

    image = cv2.imread("image/SF6_1.jpg")
    receive2 = meterReader(image, ["SF6_1"])

    image = cv2.imread("image/youwen_4.jpg")
    receive2 = meterReader(image, ["youwen_4"])

    image = cv2.imread("image/pressure_1.jpg")
    receive2 = meterReader(image, ["pressure_1"])

    # 基于遮罩算法的pressure表读数测试`1
    meter_id = "pressure2_1"
    config_base_path = "config/"
    image = cv2.imread("image/pressure2_1.jpg")
    test_pressure_recognition(config_base_path, image, meter_id)

    image = cv2.imread("image/absorb_1.jpg")
    receive2 = meterReader(image, ["absorb_1"])

    image = cv2.imread("image/switch_1.jpg")
    receive2 = meterReader(image, ["switch_1"])

    image = cv2.imread("image/blenometer_1.jpg")
    receive2 = meterReader(image, ["blenometer_1"])


def test_pressure_recognition(config_base_path, image, meter_id):
    json_modifier = JsonModifier(meter_id, config_base_path, revert_before_del=True)
    # case 1:不使用圆拟合算法,使用标定信息定位圆
    receive2 = meterReader(image, [meter_id])
    print("Case 1: ", receive2)
    # case 2:使用圆拟合算法
    json_modifier.modifyDic({
        "enableFit": True
    })
    receive2 = meterReader(image, [meter_id])
    print("Case 2: ", receive2)
    # case 3:不使用圆拟合算法,指针分辨率为30
    json_modifier.revert()  # 回退
    json_modifier.modifyKv("ptrResolution", 30)
    receive2 = meterReader(image, [meter_id])
    print("Case 3: ", receive2)
    # case 4:使用圆拟合算法,指针分辨率为50
    json_modifier.revert()
    json_modifier.modifyDic({
        "enableFit": True,
        "ptrResolution": 50
    }
    )
    print("Case 4: ", receive2)
    # case 5:使用直方图均衡化,不使用圆拟合算法,使用标定信息
    json_modifier.revertToOriginal()  # 回滚到原配置
    json_modifier.modifyKv("enableEqualizeHistogram", True)
    json_modifier.modifyKv("ptrResolution", 40)
    receive2 = meterReader(image, [meter_id])
    print("Case 5: ", receive2)


if __name__ == "__main__":
    serverProcess = multiprocessing.Process(target=startServer)
    results = multiprocessing.Manager().list()
    clientProcess = multiprocessing.Process(target=startClient, args=(results,))
    serverProcess.start()
    time.sleep(30)
    clientProcess.start()
    clientProcess.join()
    serverProcess.terminate()

    codecov()

    for result in results:
        if result == False:
            exit(100)

# test store interface
# image = open("template/1_1.jpg", "rb")
# imageByte = base64.b64encode(image.read())
# data = json.dumps({
#     "template": imageByte.decode("ascii"),
#     "imageID": "1",
#     "config": {
#       "distance": 10.0,
#       "horizontal": 10.0,
#       "vertical": 20.0,
#       "name": "1_1",
#       "type": "SF6",
#       "ROI": {
#           "x": 200,
#           "y": 200,
#           "w": 1520,
#           "h": 680
#       },
#       "startPoint": {
#           "x": -1,
#           "y": -1
#       },
#       "endPoint": {
#           "x": -1,
#           "y": -1
#       },
#       "centerPoint": {
#           "x": -1,
#           "y": -1
#       },
#       "startValue": 0.0,
#       "totalValue": 2.0
#     }
# })
#
# r = requests.post("http://127.0.0.1:5000/store", data=data.encode("utf-8"))
#
# print(r.text)
