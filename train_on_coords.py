import csv
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import math


read_csv = ["CoordsNormalized2.csv", "CoordsBad.csv", "CoordsSet3.csv"]

labels = []
data = []

for csv_list in read_csv:
    print(f"Reading from {csv_list}")
    with open(csv_list, mode='r', newline='') as image_list:
        reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            labels.append(float(row[0]))
    
            data.append(np.array(row[1:]).astype(float))

#with open(read_csv2, mode='r', newline='') as image_list:
#    reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#    for row in reader:
#        labels.append(float(row[0]))
#
#        data.append(np.array(row[1:]).astype(float))
    
train_labels, test_labels, train_data, test_data = train_test_split(np.array(labels), np.array(data), test_size=0.2, shuffle = True)
    
model = models.Sequential()
model.add(layers.Dense(200, activation='relu', input_shape=(63,)))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(26))
    
model.summary()
    
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
history = model.fit(np.array(train_data), np.array(train_labels), epochs=10, 
                    validation_data=(np.array(test_data), np.array(test_labels)))
    
test_loss, test_acc = model.evaluate(np.array(test_data),  np.array(test_labels), verbose=2)
print(test_acc)

model.save_weights("200_100_Weights.h5")