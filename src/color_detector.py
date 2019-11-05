#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 5 12:34:36 2019

@author: suman.choudhury
"""

import tensorflow as tf
import numpy as np
import cv2
import glob
from utils import utils as ut
from keras.models import load_model

class ColorDetector():

    """
    Color detection model

    """

    def __init__(self,config):
        self.color_model = config["color_model"]
        self.color_label = ["blue","green","red"]
        self.num_color_classes = 3
        self.dict_class = {0: "blue", 1: "green", 2: "red"}
        self.loaded_model = load_model(self.color_model)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.color_model, 'rb') as fid:
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
        color detector model predict function

        """

        img = cv2.resize(image, (299, 299))
        img = np.reshape(img, (1, 299, 299, 3))
        predictions = self.loaded_model.predict(img)[0]
        class_name = self.dict_class[np.argmax(predictions)]
        score = np.max(predictions)
        all_scores = predictions.tolist()
        all_scores = [i * 100 for i in all_scores]

        return class_name , all_scores
