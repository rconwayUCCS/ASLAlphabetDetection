import csv
#import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from track_with_mediapipe import create_hand_array
import matplotlib.pyplot as plt

read_csv = ["CoordsNormalized2.csv", "CoordsBad.csv", "CoordsSet3.csv"]

labels = []
data = []

raw_x = []
raw_y = []

image_size = 16

for csv_list in read_csv:
    print(f"Reading from {csv_list}")
    with open(csv_list, mode='r', newline='') as image_list:
        reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            labels.append(float(row[0]))

            raw_x = np.array(row[1:22]).astype(float)
            raw_y = np.array(row[22:43]).astype(float)

            hand = np.zeros((image_size, image_size))

            for i in range(21):
                x_point = round(raw_x[i]*(image_size - 1))
                y_point = round(raw_y[i]*(image_size - 1))

                #if hand[x_point, y_point, z_point] == 1:
                    #print("Overlap!")
                hand[x_point, y_point] = 1

            data.append(hand)


train_labels, test_labels, train_images, test_images = train_test_split(labels, data, test_size=0.2, shuffle = True)

input_shape = (32, image_size, image_size, 1)

model = models.Sequential()
model.add(layers.Conv2D(50, 4, activation='relu', input_shape=input_shape[1:]))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(100, 3, activation='relu'))
#model.add(layers.MaxPooling3D((2, 2, 2)))
#model.add(layers.Conv2D(64, (3, 3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(96, activation='relu'))
model.add(layers.Dense(26))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(np.array(train_images), np.array(train_labels), batch_size=32, epochs=10, 
                    validation_data=(np.array(test_images), np.array(test_labels)))

test_loss, test_acc = model.evaluate(np.array(test_images),  np.array(test_labels), verbose=2)
print(test_acc)

model.save_weights("2d_Weights.h5")