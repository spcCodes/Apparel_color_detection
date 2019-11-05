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

with open("config/config.json", "r") as f:
    config = json.load(f)
    f.close()

model_apparel_detection = ApparelDetector(config)
model_color_detection = ColorDetector(config)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

#path to test image
#test_image_path = config["test_image"]
test_image_path = args["image"]


#getting the file name with .jpg extension
base_file_name = os.path.basename(test_image_path)

#reading an image
image = cv2.imread(test_image_path)

#getting the width and height
[h, w] = image.shape[:2]


#calling the apparel detector and getting the boxes and scores
(boxes, scores, classes, num_detections) = model_apparel_detection.model_predict(image)

#getting the total number of products and predictions
totalcount , product_name,product_confidence,product_image = results(image,h,w,boxes, scores, classes, num_detections)

print("################ PRINTING THE RESULTS ###############")

print("Products found:" , totalcount)

for num in range(0,totalcount):

    product_number = num +1
    name = product_name[num]
    print("Product"+'_'+str(product_number)+":" ,name)
    conf = product_confidence[num]
    print("Confidence score: {:.2f}%" .format(conf))
    cropped_image = product_image[num]

    color_class_name,color_class_score = model_color_detection.model_predict(cropped_image)
    print("Predicted color:" , color_class_name)

    print("Blue : {:.2f}%  Green : {:.2f}%  Red : {:.2f}%".format(color_class_score[0],color_class_score[1],color_class_score[2]))





















