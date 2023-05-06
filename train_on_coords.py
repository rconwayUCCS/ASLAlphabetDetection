import csv
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import math


read_csv = "CoordsNormalized2.csv"

with open(read_csv, mode='r', newline='') as image_list:
    reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    labels = []
    data = []
    for row in reader:
        labels.append(float(row[0]))

        data.append(np.array(row[1:]).astype(float))

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #
    #for i in range(21):
    #    print(i)
    #    ax.scatter(raw_x[-1][i], raw_y[-1][i], raw_z[-1][i], marker='o', color='green')
    #
    #thumb_line_x = [raw_x[-1][0], raw_x[-1][4]]
    #thumb_line_y = [raw_y[-1][0], raw_y[-1][4]]
    #thumb_line_z = [raw_z[-1][0], raw_z[-1][4]]
    #
    #palm_line_x = [raw_x[-1][5], raw_x[-1][17]]
    #palm_line_y = [raw_y[-1][5], raw_y[-1][17]]
    #palm_line_z = [raw_z[-1][5], raw_z[-1][17]]
    #
    #left_palm_line_x = [raw_x[-1][0], raw_x[-1][17]]
    #left_palm_line_y = [raw_y[-1][0], raw_y[-1][17]]
    #left_palm_line_z = [raw_z[-1][0], raw_z[-1][17]]
    #
    #right_palm_line_x = [raw_x[-1][0], raw_x[-1][5]]
    #right_palm_line_y = [raw_y[-1][0], raw_y[-1][5]]
    #right_palm_line_z = [raw_z[-1][0], raw_z[-1][5]]
    #
    #ax.plot(thumb_line_x, thumb_line_y, thumb_line_z, color='black')
    #ax.plot(palm_line_x, palm_line_y, palm_line_z, color='black')
    #
    #ax.plot(left_palm_line_x, left_palm_line_y, left_palm_line_z, color='black')
    #ax.plot(right_palm_line_x, right_palm_line_y, right_palm_line_z, color='black')
    #
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    #
    #plt.show()
    
    train_labels, test_labels, train_data, test_data = train_test_split(np.array(labels), np.array(data), test_size=0.2, shuffle = True)
    
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(63,)))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(26))
    
    model.summary()
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    history = model.fit(np.array(train_data), np.array(train_labels), epochs=5, 
                        validation_data=(np.array(test_data), np.array(test_labels)))
    
    test_loss, test_acc = model.evaluate(np.array(test_data),  np.array(test_labels), verbose=2)
    print(test_acc)

    model.save_weights("Norm_Weights.h5")