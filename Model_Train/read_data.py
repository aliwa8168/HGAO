#!/user/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import os
import numpy as np
import datetime
# Load images from the dataset
# dataset_path = './LC2500'
# dataset_path = './Plant'
# dataset_path = './AIDER'
# dataset_path = './UC'
dataset_path = './zhongyaocai'  # Chinese medicine dataset
classes = os.listdir(dataset_path)
# print(classes)
Read_time = datetime.datetime.now()
print("Start time for reading images:", Read_time)
x_data = []  # Used to store training set image data
y_data = []  # Used to store training set labels
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path):
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            x_data.append(img)
            y_data.append(class_name)

    print(class_name, 'read successfully')

x_data = np.array(x_data)
y_data = np.array(y_data)
