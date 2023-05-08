import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

read_csv = ["CoordsNormalized2.csv"]
            #, "CoordsBad.csv"]
            #, "CoordsSet3.csv"]

labels = []
data = []

raw_x = []
raw_y = []
raw_z = []

image_size = 16

for csv_list in read_csv:
    print(f"Reading from {csv_list}")
    with open(csv_list, mode='r', newline='') as image_list:
        reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            labels.append(float(row[0]))

            raw_x = np.array(row[1:22]).astype(float)
            raw_y = np.array(row[22:43]).astype(float)
            raw_z = np.array(row[43:]).astype(float)

            hand = np.zeros((image_shape, image_shape, image_shape))

            for i in range(21):
                x_point = round(raw_x[i]*image_size)
                y_point = round(raw_y[i]*image_size)
                z_point = round(raw_z[i]*image_size)

                hand[x_point, y_point, z_point] = 1

            data.append(hand)

images = []
for img in paths:
    images.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))

labels = []
for let in labels_letters:
    labels.append(alphabet.index(let))

train_labels, test_labels, train_images, test_images = train_test_split(labels, images, test_size=0.2, shuffle = True)

model = models.Sequential()
model.add(layers.Conv3D(32, (3, 3), activation='relu', input_shape=(16, 16, 16,)))
model.add(layers.MaxPooling3D((3, 3)))
model.add(layers.Conv3D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2)))
model.add(layers.Conv3D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(np.array(train_images), np.array(train_labels), epochs=10, 
                    validation_data=(np.array(test_images), np.array(test_labels)))

test_loss, test_acc = model.evaluate(np.array(test_images),  np.array(test_labels), verbose=2)
print(test_acc)

model.save_weights("CNN_Weights2.h5")