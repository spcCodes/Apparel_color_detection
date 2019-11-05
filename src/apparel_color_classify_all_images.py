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

#taking the image path
image_path = config["test_image_path"]

image_path1 = image_path + "/*.jpg"
#taking all the images files to glob
TEST_IMAGE_PATHS = glob.glob(image_path1)

#creating an empty csv file list
csv_file =[]

for (number,imageTest) in enumerate(TEST_IMAGE_PATHS):

    # getting the file name with .jpg extension
    base_file_name = os.path.basename(imageTest)

    #reading an image
    image = cv2.imread(imageTest)

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

        file_write = [str(base_file_name) , totalcount , name , color_class_score[0],color_class_score[1],color_class_score[2]]

        print("Blue : {:.2f}%  Green : {:.2f}%  Red : {:.2f}%".format(color_class_score[0],color_class_score[1],color_class_score[2]))

        csv_file.append(file_write)

#after writing all the points in csv file , now saving in csv file
with open('output/output.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['Image_Name', 'Number_of_Clothing' , 'Product_Type' , 'Blue' , 'Green' , 'Red' ])
    writer.writerows(csv_file)
csvFile.close()


















