#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 5 21:41:15 2019

@author: suman.choudhury
"""

from PIL import Image
from io import BytesIO
import base64
import unittest
import requests
import cv2
import io
import numpy as np


class TestVisionServer(unittest.TestCase):

    def get_base64_string(self, cropped):
        pil_img = cropped
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        base64_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        return base64_string

    def get_base64(self, cropped):
        base = base64.b64encode(cropped.read())
        return base

    def get_inputs(self):
        image_path = "examples/image_11.jpg"
        img = Image.open(image_path)
        return img

    def test_vision_server(self):
        testimage = self.get_inputs()
        encodedString = self.get_base64_string(testimage)
        payload = {'encodedContent': encodedString}
        r = requests.post("http://0.0.0.0:9790/recognise", json=payload)
        print(r.text)



if __name__ == '__main__':
    unittest.main()
