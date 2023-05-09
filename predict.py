import cv2
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from track_with_mediapipe import detect_hand, convert_palm_normal, normalize


alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

model = models.Sequential()
model.add(layers.Dense(200, activation='relu', input_shape=(63,), batch_size=1))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(26))

model.load_weights('200_100_Weights.h5')

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
            output = np.reshape(output, (1, 63))
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
    data = []
    raw_x = []
    raw_y = []
    raw_z = []
    with open(csv_list, mode='r', newline='') as image_list:
        reader = csv.reader(image_list, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            labels.append(float(row[0]))

            raw_x = np.array(row[1:22]).astype(float)
            raw_y = np.array(row[22:43]).astype(float)
            raw_z = np.array(row[43:]).astype(float)

            data.append(create_hand_array(raw_x, raw_y, raw_z, image_size))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.CategoricalAccuracy())

    test_loss, test_acc = model.evaluate(np.array(data),  np.array(labels))
    print(test_acc)

read_csv = ["CoordsNormalized2.csv", "CoordsBad.csv", "CoordsSet3.csv"]
predict_from_file(read_csv[0], model)
#predict_live(model)