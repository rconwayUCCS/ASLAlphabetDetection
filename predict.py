import cv2
import numpy as np
import tensorflow as tf
#from tensorflow import 
from tensorflow.keras import datasets, layers, models
from track_with_mediapipe import detect_hand

alphabet = ["A", "B", "C", "D"]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), batch_size = 1))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4))

model.load_weights('CNN_Weights.h5')

#model.summary()

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    output = detect_hand(frame, 64)
    frame = cv2.resize(frame,(256,256))
    cv2.imshow("Input", frame)
    if output is None:
        print("No Hand Detected")
        continue
    output = np.reshape(output, (1, 64, 64, 1))
    prediction = model.predict(np.array(output))
    prediction_max = np.argmax(prediction)
    
    print(alphabet[prediction_max])

    if cv2.waitKey(1) & 0xFF == ord(' '):#exiting
        break

cap.release()
cv2.destroyAllWindows()