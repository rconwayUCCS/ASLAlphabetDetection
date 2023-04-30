import cv2
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

read_csv = "ArrayTest.csv"

with open(read_csv, mode='r', newline='') as image_list:
        reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        labels = []
        data = []
        for row in reader:
            labels.append(row[0])
            data.append(row[1])

        print(labels[0])
        print(data[0][0])