import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

df = pd.read_csv("image_list.csv", names = ["labels", "paths"])

labels_letters = df["labels"].tolist()
paths = df["paths"].tolist()

images = []
for img in paths:
    images.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))

labels = []
for let in labels_letters:
    if let == "A":
        labels.append(0)
    elif let == "B":
        labels.append(1)
    elif let == "C":
        labels.append(2)
    elif let == "D":
        labels.append(3)

train_labels, test_labels, train_images, test_images = train_test_split(labels, images, test_size=0.2, shuffle = True)

print(len(train_labels))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(np.array(train_images), np.array(train_labels), epochs=5, 
                    validation_data=(np.array(test_images), np.array(test_labels)))

test_loss, test_acc = model.evaluate(np.array(test_images),  np.array(test_labels), verbose=2)
print(test_acc)

model.save_weights("CNN_Weights.h5")