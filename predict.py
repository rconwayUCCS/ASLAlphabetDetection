import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from track_with_mediapipe import detect_hand

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

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
    text = "None"
    if output is not None:
        output = np.reshape(output, (1, 64, 64, 1))
        prediction = model.predict(np.array(output))
        prediction_max = np.argmax(prediction)
        
        text = alphabet[prediction_max]
        #print(alphabet[prediction_max])
    #else:
        #print("No Hand Detected")

    frame = cv2.resize(frame,(500,400))
    image = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                 (0, 0, 0), 2, cv2.LINE_AA, False)
    cv2.imshow("Video", image)
    if cv2.waitKey(1) & 0xFF == ord(' '):#exiting
        break

cap.release()
cv2.destroyAllWindows()