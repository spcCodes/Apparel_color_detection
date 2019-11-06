# Apparel_Color_detection
This project focusses on detecting the apparel which may be dress , tops or jeans and following that it would identify the color of the apparel which may be green , blue or red. 

## Table of contents
* [General info](#general-info)
* [Project Structure](#project_str)
* [Dataset](#data)
* [Project Execution Steps](#project)
* [Flask App](#flask)
* [Result Images](#images)
* [Future Work](#future)

<a name="general-info"></a>
## General info
The objective of this project is to develop a working model that can automatically identify the apparel types given an image and also identify its colors. To mantain the sanity and complexity of the dataset , the apparel is restricted to dresss,tops and jeans while color is restricted to red, blue and green. It is targeted to serve as an automatic apparel recognition api.


The dataset used for this model was the **[cloths dataset](https://drive.google.com/open?id=1aj-umzIq9ujQTGnUae__MVgwjTWxviSP)**. This dataset was collected from the net as we wanted aa custom dataset whoch catered to these 3 apparels only.If you face downloading the dataset , do drop me a mail to gain the access credentials.

So, in the image recognition api , at the first stage the apparel is detected and identified. So we have used object detection technique for this module. We used **Tensorflow Object Detection** api to train the model in accordance to our custom dataset. The architecture used for training the model was **SSD - Mobilenet** architecture which gave us a very good IOU and also a very good classification accuracy on the test unseen dataset.

In the next stage after the apparel is identified , the cropped part of the image is taken out which is being extracted using the information of the bounding box givne from the object detection module. This cropped part of the image is then givne toi **image classification module** which identifies the color of the apparel accordingly which may be red , blue or green.

**XceptionNet** was used as a pretrained model which was further fine tuned for our custom image recognition engine. The color recognition module was giving a test accuracy of around **99.1%** which is very promising.

All the codes for the project is kept in the src folder and the common scripts are kept in the scripts folder.

Current validation accuracy for the apparel detection stands at **94%** which can be further improved if trained for a larger period of time. Due to time constraints , further hyper-parameter optimisation for the network could not be done.


<a name="project_str"></a>
## Project Structure

The entire project structure is as follows:

```
├── config
│   └── config.json
├── data
├── examples
│   ├── image_1.jpg
│   ├── image_2.jpg
│   ├── image_3.jpg
│   ├── image_4.jpg
│   ├── image_5.jpg
│   ├── image_6.jpg
│   ├── image_7.jpg
│   ├── image_8.jpg
│   └── image_9.jpg
|   .......
├── models
│   ├── apparel_detector.pb
│   ├── apparel_labels.pbtxt
│   └── color_model.h5
├── output
│   └── output.csv
├── src
│   ├── apparel_color_classify.py
│   ├── apparel_color_classify_all_images.py
│   ├── apparel_color_detector.py
│   ├── apparel_detector.py
│   ├── color_detector.py
│   └── get_product_details.py
├── test
├── utils
│   │   └── utils.cpython-36.pyc
│   ├── string_int_label_map_pb2.py
│   └── utils.py
├── app.py
├── requirements.txt
└── test_apparel_classifier.py

```
As we see from the project structure :
a) all the class related to apparel detector and color detector are kept in **src** folder. 

b)The test images are kept in examples folder. 

c) The configurable parameters like model loading path , threshold , test image path are kept in **config.json** file

d) All the trained models and tits label files are kept in **models** folder

e) The utilities or helper functions which are required throughout the project are kept in utils.py insode **utils** folder

f) **app.py** was written for a flask python wrapper to start the server kernel. Also the **test_classifier.py** is on the client side to test it given a web based api


<a name="data"></a>
## Dataset

The dataset for this challenge was a custom made dataset which was extracted from the net [cloth dataset](https://drive.google.com/open?id=1aj-umzIq9ujQTGnUae__MVgwjTWxviSP)

The Clothing dataset contains 810 images of around 270 images of each class. The data is split into 720 training images and 81 testing images. As the challenge clearly stated just to take the training dataset for preparing the model , only the  containing 720 inages were taken for training our model.
The meta data information are the label annotations that were donw using LabelImg which gaves us the bounding box required to train our object detection module.








