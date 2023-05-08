import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

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
            

        return landmarks_x, landmarks_y, landmarks_z

def convert_palm_normal(xs, ys, zs):
    #Calculate and normalize the original normal vector of palm
    palm_vec1 = [xs[5] - xs[0], ys[5] - ys[0], zs[5] - zs[0]]
    palm_vec2 = [xs[17] - xs[0], ys[17] - ys[0], zs[17] - zs[0]]
    palm_normal = np.cross(palm_vec1, palm_vec2)
    palm_normal = palm_normal / np.linalg.norm(palm_normal)

    #Calculate and normalize the axis of rotation, and extract the angle
    axis_of_rotation = np.cross([0,0,1], palm_normal)
    axis_mag = np.linalg.norm(axis_of_rotation)
    angle_of_rotation = np.arcsin(axis_mag)
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    q = Quaternion(axis=axis_of_rotation, angle=angle_of_rotation)

    x_primes = [xs[0]]
    y_primes = [ys[0]]
    z_primes = [zs[0]]

    for i in range(1, len(xs)):
        x = xs[i] - xs[0]
        y = ys[i] - ys[0]
        z = zs[i] - zs[0]

        x, y, z = q.rotate([x, y, z])
        x_primes.append(x + xs[0])
        y_primes.append(y + ys[0])
        z_primes.append(z + zs[0])

    return x_primes, y_primes, z_primes

def normalize(x_primes, y_primes, z_primes):
    min_x = min(x_primes)
    min_y = min(y_primes)
    min_z = min(z_primes)
    

    x_primes = [x - min_x for x in x_primes]
    y_primes = [x - min_y for x in y_primes]
    z_primes = [x - min_z for x in z_primes]

    
    x_primes = [x / max(x_primes) for x in x_primes]
    y_primes = [x / max(y_primes) for x in y_primes]
    z_primes = [x / max(z_primes) for x in z_primes]

    return x_primes, y_primes, z_primes

    
#print(detect_hand("F1.jpg"))