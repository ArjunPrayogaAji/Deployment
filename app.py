from flask import Flask, jsonify, request
from PIL import Image
from tesserocr import PyTessBaseAPI

import cv2 as cv

import base64
import skimage
import skimage.io

app = Flask(__name__)
@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    api_text = PyTessBaseAPI(path='tessdata', psm=6, lang='eng')

    image = decode(data['file'], grayscale=True)
    text = read_text_with_confidence(image, api_text)
    result = {"message":"success", "result": text}
    return jsonify(result)

# library
def read_text_with_confidence(image, api, whitelist=''):
    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        return '', 0

    image_pil = Image.fromarray(image)

    try:
        api.SetImage(image_pil)

        if whitelist != '':
            api.SetVariable('tessedit_char_whitelist', whitelist)

        api.Recognize()

        text = api.GetUTF8Text()
        confidence = api.MeanTextConf()
    except Exception:
        print("[ERROR] Tesseract exception")
    
    return text, confidence

def decode(base64_string, grayscale=False):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    if base64_string is None:
        return None

    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    if grayscale is True:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

# library

if __name__=='__main__':
    app.run()