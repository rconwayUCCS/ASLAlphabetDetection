import cv2
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
#import track_with_mediapipe as tm
from track_with_mediapipe import *


alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

#model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
#model.add(layers.MaxPooling2D((3, 3)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(26))

image_size = 12

input_shape = (32, image_size, image_size, image_size, 1)

model = models.Sequential()
model.add(layers.Conv3D(32, 3, activation='relu', input_shape=input_shape[1:]))
model.add(layers.MaxPooling3D((3, 3, 3)))
model.add(layers.Conv3D(32, 2, activation='relu'))
#model.add(layers.MaxPooling3D((2, 2, 2)))
#model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(96, activation='relu'))
model.add(layers.Dense(26))

model.load_weights('3d_Weights.h5')

model.summary()

def predict_live(model):
    cap = cv2.VideoCapture(0)

    queue = [None for i in range(10)]

    oldest = 0
    while(True):
        sum = [0] * 26
        ret, frame = cap.read()
        output = detect_hand(frame)
        image = cv2.resize(frame,(500,400))
        hand_image = np.zeros_like(image)
        sorted_alpha = ["None", "None", "None", "None", "None"]
        if output is not None:
            output = convert_palm_normal(output[0], output[1], output[2])
            for i in range(21):
                hand_image = cv2.circle(hand_image, (int(output[0][i] * 500), int(output[1][i] * 400)), 1, (0, 255, 0), -1)

            output = normalize(output[0], output[1], output[2])
            output = create_hand_array(output[0], output[1], output[2], image_size)
            output = np.reshape(output, input_shape)
            prediction = model.predict(np.array(output))[0]
        
            queue[oldest] = prediction
            oldest += 1
            if oldest >= 10:
                oldest = 0
            for i in range(10):
                if queue[i] is None:
                    break;
                for j in range(26):
                    sum[j] += queue[i][j]

            sorted_alpha = [x for _, x in sorted(zip(sum, alphabet))]
        
        image = cv2.putText(frame, sorted_alpha[-1], (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                         (0, 0, 0), 2, cv2.LINE_AA, False)
        for i in range(2, 6):
            image = cv2.putText(frame, sorted_alpha[-i], (25, 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                         (0, 0, 0), 2, cv2.LINE_AA, False)

        cv2.imshow("Hand", hand_image)
        cv2.imshow("Video", image)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_from_file(csv_list, model):
    labels = []
    raw_x = []
    raw_y = []
    raw_z = []
    data = []
    for file in csv_list:
        with open(file, mode='r', newline='') as image_list:
            reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                labels.append(float(row[0]))

                raw_x = np.array(row[1:22]).astype(float)
                raw_y = np.array(row[22:43]).astype(float)
                raw_z = np.array(row[43:]).astype(float)

                hand = np.zeros((image_size, image_size, image_size))

                for i in range(21):
                    x_point = round(raw_x[i]*(image_size - 1))
                    y_point = round(raw_y[i]*(image_size - 1))
                    z_point = round(raw_z[i]*(image_size - 1))

                    #if hand[x_point, y_point, z_point] == 1:
                        #print("Overlap!")
                    hand[x_point, y_point, z_point] = 1

                #data.append(hand)

                data.append(create_hand_array(raw_x, raw_y, raw_z, image_size))

    accuracy_dict = dict()
    for i in range(26):
        subset = []
        for j, row in enumerate(data):
            if labels[j] == i:
                subset.append(row)
        predictions = model.predict(np.array(subset))
        predictions = np.argmax(predictions, axis=1)

        accuracy_dict[alphabet[i]] = np.count_nonzero(predictions == i) / len(predictions)

    print(sorted(accuracy_dict.items(), key=lambda x:x[1]))

read_csv = ["CoordsNormalized2.csv", "CoordsBad.csv", "CoordsSet3.csv"]
predict_from_file(read_csv, model)
#predict_live(model)