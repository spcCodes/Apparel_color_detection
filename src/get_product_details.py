#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 5 12:34:36 2019

@author: suman.choudhury
"""


import numpy as np
import tensorflow as tf


def results(image, h, w, boxes, scores, classes, num_detections):
    scores_final = []
    boxes_final = []
    class_final = []

    # for dress
    kboxes_dress = boxes[classes == 1.]
    kscores_dress = scores[classes == 1.]

    if len(kscores_dress) != 0:
        scores_dress = kscores_dress[0]
        boxes_dress = kboxes_dress[0]

        if scores_dress > 0.3:
            bbox_dress = boxes_dress
            scores_final.append(scores_dress)
            boxes_final.append(bbox_dress)
            class_final.append('dress')

    # for jeans
    kboxes_jeans = boxes[classes == 2.]
    kscores_jeans = scores[classes == 2.]

    if len(kscores_jeans) != 0:
        scores_jeans = kscores_jeans[0]
        boxes_jeans = kboxes_jeans[0]

        if scores_jeans > 0.3:
            bbox_jeans = boxes_jeans
            scores_final.append(scores_jeans)
            boxes_final.append(bbox_jeans)
            class_final.append('jeans')

    # for tops
    kboxes_tops = boxes[classes == 3.]
    kscores_tops = scores[classes == 3.]

    if len(kscores_tops) != 0:
        scores_tops = kscores_tops[0]
        boxes_tops = kboxes_tops[0]

        if scores_tops > 0.3:
            bbox_tops = boxes_tops
            scores_final.append(scores_tops)
            boxes_final.append(bbox_tops)
            class_final.append('tops')

    # count of the products
    count = len(class_final)

    product_name = []
    product_confidence = []
    product_image = []

    for (num, i) in enumerate(class_final):
        box = boxes_final[num]
        class_score = scores_final[num]
        ymin, xmin, ymax, xmax = box
        aa = np.array(box)
        bb = (aa * np.array([h, w, h, w])).astype("int")
        (startX, startY, endX, endY) = (bb[1], bb[0], bb[3], bb[2])
        class_name = class_final[num]
        #print(class_name)
        confidence = class_score * 100

        image_arr = image[startY:endY, startX:endX]
        # cv2.imshow("Image:",image_arr)
        # cv2.waitKey(0)
        product_name.append(class_name)
        product_confidence.append(confidence)
        product_image.append(image_arr)

    return count, product_name, product_confidence, product_image