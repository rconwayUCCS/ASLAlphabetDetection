import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

#Code taken from https://google.github.io/mediapipe/solutions/hands.html

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# For static images:
IMAGE_FILES = ["C100.jpg", "CTest.jpg", "F1.jpg"]

def detect_hand(file, out_size):

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        image_size = 256
        # Read the image and resize it before detecting hands
        image = cv2.imread(file)
        image = cv2.resize(image, (image_size, image_size))
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        blank = np.zeros_like(image)

        if not results.multi_hand_landmarks:
            return None

        circle_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255,255,255))
        line_spec = mp_drawing.DrawingSpec(thickness=2, color=(255,255,255))

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                blank,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                circle_spec,
                line_spec)

            max_x, max_y = 0, 0
            min_x, min_y = image_size, image_size
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmark_x = int(landmark.x * image_size)
                landmark_y = int(landmark.y * image_size)
                if (landmark_x >= max_x):
                    max_x = landmark_x
                elif (landmark_x <= min_x):
                    min_x = landmark_x
                if (landmark_y >= max_y):
                    max_y = landmark_y
                elif (landmark_y <= min_y):
                    min_y = landmark_y

        #Make sure the bounding box is a square, then crop and resize the image.
        x_dim = max_x - min_x
        y_dim = max_y - min_y

        if x_dim > y_dim:
            diff = int((x_dim - y_dim) / 2)
            max_y += diff
            min_y -= diff
        else:
            diff = int((y_dim - x_dim) / 2)
            max_x += diff
            min_x -= diff

        cropped = blank[min_y - 8 :max_y + 8, min_x - 8:max_x + 8]
        cropped = cv2.resize(cropped, (out_size, out_size))
        binary = cv2.threshold(cropped, 32, 255, cv2.THRESH_BINARY)[1]

        

        #plt.title("Resultant Image");plt.axis('off');plt.imshow(binary[:,:,::-1]);plt.show()

        return binary
    
#detect_hand("C100.jpg", 64)