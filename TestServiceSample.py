import requests
import base64
import json
import os
import time
import cv2
import multiprocessing
from Interface import meterReader


def startServer():
    os.system("python FlaskService.py")


def startClient(results):
    images = os.listdir("image")
    for im in images:
        image = open("image/"+im, "rb")
        imageByte = base64.b64encode(image.read())
        data = json.dumps({
            "image": imageByte.decode("ascii"),
            "imageID": im.split('.')[0]
        })
        r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
        receive = json.loads(r.text)
        print(im, receive)

        if None in receive:
            results.append(False)
        else:
            results.append(True)

# def startClient(results):
#     # test reader interface
#
#     # ===========================bileiqi1 test===========================
#     image = open("image/bileiqi1.jpg", "rb")
#     imageByte = base64.b64encode(image.read())
#     data = json.dumps({
#         "image": imageByte.decode("ascii"),
#         "imageID": "bileiqi1"
#     })
#     r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
#     receive = json.loads(r.text)
#     print(receive)
#
#     if not "bileiqi1_1" in receive:
#         results.append(False)
#     else:
#         results.append(True)
#
#     # ===========================bileiqi2 test===========================
#     # image = open("image/bileiqi2_1.jpg", "rb")
#     # imageByte = base64.b64encode(image.read())
#     # data = json.dumps({
#     #     "image": imageByte.decode("ascii"),
#     #     "imageID": "bileiqi2"
#     # })
#     # r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
#     # receive = json.loads(r.text)
#     # print(receive)
#     # if not "bileiqi2_1" in receive:
#     #     results.append(False)
#     # else:
#     #     results.append(True)
#
#     # ===========================SF6 test===========================
#     image = open("image/SF6_1.jpg", "rb")
#     imageByte = base64.b64encode(image.read())
#     data = json.dumps({
#         "image": imageByte.decode("ascii"),
#         "imageID": "SF6"
#     })
#     r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
#     receive = json.loads(r.text)
#     print(receive)
#
#     if not "SF6_1" in receive:
#         results.append(False)
#     else:
#         results.append(True)
#
#     # ===========================youwen test===========================
#     image = open("image/youwen_4.jpg", "rb")
#     imageByte = base64.b64encode(image.read())
#     data = json.dumps({
#         "image": imageByte.decode("ascii"),
#         "imageID": "youwen"
#     })
#     r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
#     receive = json.loads(r.text)
#     print(receive)
#
#     if not "youwen_4" in receive:
#         results.append(False)
#     else:
#         results.append(True)
#
#     # ===========================pressure test===========================
#     image = open("image/pressure_1.jpg", "rb")
#     imageByte = base64.b64encode(image.read())
#     data = json.dumps({
#         "image": imageByte.decode("ascii"),
#         "imageID": "pressure"
#     })
#     r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
#     receive = json.loads(r.text)
#     print(receive)
#
#     if not "pressure_1" in receive:
#         results.append(False)
#     else:
#         results.append(True)
#
#     # ===========================absorb test===========================
#     image = open("image/absorb_1.jpg", "rb")
#     imageByte = base64.b64encode(image.read())
#     data = json.dumps({
#         "image": imageByte.decode("ascii"),
#         "imageID": "absorb"
#     })
#     r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
#     receive = json.loads(r.text)
#     print(receive)
#
#     if not "absorb_1" in receive:
#         results.append(False)
#     else:
#         results.append(True)
#
#     # ===========================switch test===========================
#     image = open("image/switch_1.jpg", "rb")
#     imageByte = base64.b64encode(image.read())
#     data = json.dumps({
#         "image": imageByte.decode("ascii"),
#         "imageID": "switch"
#     })
#     r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
#     receive = json.loads(r.text)
#     print(receive)
#
#     if not "switch_1" in receive:
#         results.append(False)
#     else:
#         results.append(True)
#
#     # ===========================bleno test===================================
#     image = open("image/blenometer_1.jpg", "rb")
#     imageByte = base64.b64encode(image.read())
#     data = json.dumps({
#         "image": imageByte.decode("ascii"),
#         "imageID": "blenometer"
#     })
#     r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
#     receive = json.loads(r.text)
#     print(receive)
#
#     if not "blenometer_1" in receive:
#         results.append(False)
#     else:
#         results.append(True)


def codecov():
    images = os.listdir("image")
    config = os.listdir("config")
    for im in images:
        image = cv2.imread("image/"+im)
        for i in range(1, 6):
            cfg = im.split(".jpg")[0]+"_"+str(i)
            if cfg+".json" in config:
                receive2 = meterReader(image, [cfg])


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
        if not result:
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
