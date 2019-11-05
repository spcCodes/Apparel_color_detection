#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:37:46 2019

@author: suman.choudhury
"""

# importing the necessary packages
from flask import request, jsonify, Flask
import time
import utils.utils as ut
import cv2
import numpy as np
from src.apparel_color_detector import ApparelColorDetector

apparel_recog = ApparelColorDetector()

app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome to Vision Api"


@app.route('/recognise', methods=['POST'])
def recognise():
    # the data comes in byte format .
    data = request.get_json()
    encoded_content = data['encodedContent']

    # start the timer
    start_time = time.time()

    image_arr = ut.get_image(encoded_content)
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
    response = apparel_recog.predict(image_arr)
    api_response = response
    print("total API time: ", time.time() - start_time)
    return jsonify(api_response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9790)

