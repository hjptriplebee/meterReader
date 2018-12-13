import json
import base64
import cv2
import numpy as np
from Interface import *
from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def meterReaderAPI():
    data = request.get_data().decode("utf-8")
    data = json.loads(data)
    meterIDs = data["meterIDs"]
    imageByte = data["image"].encode("ascii")
    imageByte = base64.b64decode(imageByte)
    imageArray = np.asarray(bytearray(imageByte), dtype="uint8")
    image = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)

    result = meterReader(image, meterIDs)
    sendData = json.dumps(result).encode("utf-8")

    return sendData

if __name__ == '__main__':
    app.run(port=5000, debug=True)
