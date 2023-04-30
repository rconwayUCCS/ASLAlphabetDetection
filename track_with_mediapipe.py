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

def detect_hand(file):

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        if isinstance(file, str):
            image = cv2.imread(file)
        else:
            image = file
        #Image dimensions are stored as (y, x, color)
        res = image.shape
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            #print("Hand in %s not recognized" % file)
            return None
        
        landmarks_x = []
        landmarks_y = []
        landmarks_z = []
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmarks_x.append(landmark.x)
                landmarks_y.append(landmark.y)
                landmarks_z.append(landmark.z)
            

        return landmarks_z, landmarks_y, landmarks_x
    
#print(detect_hand("F1.jpg"))