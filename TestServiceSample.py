import requests
import base64
import json
import os
import time
import multiprocessing

def startServer():
    os.system("python FlaskService.py")

def startClient():
    # test reader interface
    image = open("image/bileiqi1_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "bileiqi1"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    print(r.text)

    image = open("image/SF6_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "SF6"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    print(r.text)

    image = open("image/youwen_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "youwen"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    print(r.text)

    image = open("image/pressure_1.jpg", "rb")
    imageByte = base64.b64encode(image.read())
    data = json.dumps({
        "image": imageByte.decode("ascii"),
        "imageID": "pressure"
    })
    r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
    print(r.text)

if __name__ == "__main__":

    serverProcess = multiprocessing.Process(target=startServer)
    clientProcess = multiprocessing.Process(target=startClient)
    serverProcess.start()
    time.sleep(20)
    clientProcess.start()





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
