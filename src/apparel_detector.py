#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 5 12:34:36 2019

@author: suman.choudhury
"""

from utils import utils as ut
import tensorflow as tf
import cv2
import numpy as np
import time


class ApparelDetector():
    """
    Apparel Detector model
    """

    def __init__(self, config):
        self.kurti_label = config["apparel_detection_label"]
        self.apparel_model = config["apparel_detection_model"]
        self.num_classes = config["number_classes"]
        self.label_map = ut.load_labelmap(self.kurti_label)
        self.categories = ut.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes,
                                                             use_display_name=True)
        self.category_index = ut.create_category_index(self.categories)
        self.threshold = config["confidence_threshold"]

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.apparel_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            configs = tf.ConfigProto()
            configs.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=configs)
            self.windowNotSet = True

    def model_predict(self, image):
        """
        apparel detector model predict function

        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)



