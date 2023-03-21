import pandas as pd
import cv2

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

df = pd.read_csv("image_list.csv")

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

#for index, row in df.iterrows():
#    image = cv2.imread(row[1])
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    cv2.imshow("Image", image)
#    print(image.shape)