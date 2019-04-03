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
        path = "image/" + im
        data = json.dumps({
            "path": path,
            "pointID": im.split('.')[0]+"_1"
        })
        print(path, im)
        r = requests.post("http://127.0.0.1:5000/locate", data=data.encode("utf-8"))
        receive = json.loads(r.text)
        print(im, receive)

        results.append(True)


def codecov():
    images = os.listdir("image")
    config = os.listdir("config")

    for im in images:
        image = cv2.imread("image/"+im)
        print(im)

        for i in range(1, 6):
            cfg = im.split(".jpg")[0]+"_"+str(i)
            if cfg+".json" in config:
                receive2 = meterReader(image, [cfg])
                print(cfg, receive2)

    print("codecov done")

def testVideo():
    video_path = ("video_")
    config = os.listdir("config")

    for file in os.listdir(video_path):
        if file.startswith(".DS"):
            continue
        video = cv2.VideoCapture(os.path.join(video_path, file))
        result = meterReader(video, [file[:-4] + "_1"])
        print(file, result)
    print("codecov done")


if __name__ == "__main__":
    # serverProcess = multiprocessing.Process(target=startServer)
    # results = multiprocessing.Manager().list()
    # clientProcess = multiprocessing.Process(target=startClient, args=(results,))
    # serverProcess.start()
    # time.sleep(30)
    # clientProcess.start()
    # clientProcess.join()
    # serverProcess.terminate()

    codecov()
    testVideo()
    #
    # for i in range(20):
    #     serverProcess = multiprocessing.Process(target=startServer)
    #     results = multiprocessing.Manager().list()
    #     clientProcess = multiprocessing.Process(target=startClient, args=(results,))
    #     serverProcess.start()
    #     time.sleep(30)
    #     clientProcess.start()
    #     clientProcess.join()
    #     serverProcess.terminate()
    #
    #     codecov()

    # for result in results:
    #     print(result)
    #     if not result:
    #         exit(100)



