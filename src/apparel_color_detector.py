#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 5 12:34:36 2019

@author: suman.choudhury
"""

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import imutils
import os
import glob
import json
from src.apparel_detector import ApparelDetector
from src.color_detector import ColorDetector
from src.get_product_details import *
import argparse
import csv

global graph
graph = tf.get_default_graph()


class ApparelColorDetector():

    """
    Apparel and its corresponding color detector

    """

    def __init__(self):

        with open("config/config.json", "r") as f:
            config = json.load(f)
            f.close()
        self.model_apparel_detection = ApparelDetector(config)
        self.model_color_detection = ColorDetector(config)

    def predict(self,image_arr):

        test_image_path = image_arr

        # getting the file name with .jpg extension
        #base_file_name = os.path.basename(test_image_path)

        # reading an image
        image = test_image_path

        # getting the width and height
        [h, w] = image.shape[:2]


        # calling the apparel detector and getting the boxes and scores
        (boxes, scores, classes, num_detections) = self.model_apparel_detection.model_predict(image)

        # getting the total number of products and predictions
        totalcount, product_name, product_confidence, product_image = results(image, h, w, boxes, scores, classes,
                                                                              num_detections)
        resp={}
        resp["number_of_products"] = totalcount

        for num in range(0, totalcount):
            product_number = num + 1
            name = product_name[num]

            conf = product_confidence[num]
            cropped_image = product_image[num]
            with graph.as_default():
                color_class_name, color_class_score = self.model_color_detection.model_predict(cropped_image)
            aa = "Blue : {:.2f}%  Green : {:.2f}%  Red : {:.2f}%".format(color_class_score[0], color_class_score[1],
                                                                          color_class_score[2])

            resp['Product_' + str(product_number)] = name
            resp['Product_' + str(product_number) + '' + '_confidence'] = conf
            resp['Product_' + str(product_number) + '' + '_predicted color'] = color_class_name
            resp['Product_' + str(product_number) + '' + '_color_percentages'] = aa
            resp['Product_' + str(product_number) + '' + '_name'] = name
        result = resp
        return result



