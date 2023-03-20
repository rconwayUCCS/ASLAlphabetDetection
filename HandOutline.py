import numpy as np
import cv2
from Keypoints import get_points_and_mask, color_and_crop



#This code was taken from https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Image%20Preprocessing/image_processing.py

def hand_to_binary(frame, skinMask, key_mask): 
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    key_mask = cv2.cvtColor(key_mask, cv2.COLOR_BGR2GRAY)

    lowerBoundary = np.array([140,40,30],dtype="uint8")
    upperBoundary = np.array([180,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)

    key_mask = cv2.medianBlur(key_mask, 5)

    #Morphology to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)

    key_mask = cv2.morphologyEx(key_mask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("key_mask", key_mask)

    #Create a mask only where both masks agree
    skinMask = cv2.bitwise_or(skinMask, key_mask)
    #cv2.imshow("Final Mask", skinMask)

    # blur the mask to help remove noise, then apply the
    # mask to the frame, creating a hand cutout
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    #cv2.imshow("masked",skin) # Everything apart from skin is shown to be black

    #Take hand cutout and convert to binary black and white
    bw_image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    bw_image = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY) 
    bw_image = cv2.GaussianBlur(bw_image,(5,5),0)
    bw_image = cv2.threshold(bw_image, 32, 255, cv2.THRESH_BINARY)[1]

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return bw_image
    
def process_image(path, show_images):
    img = cv2.imread(path)
    img = cv2.resize(img,(128,128))
    
    key_points2, skin_mask = get_points_and_mask(img)
    
    key_mask, cropped, skin_mask = color_and_crop(key_points2, img, skin_mask)

    processed = hand_to_binary(cropped, skin_mask, key_mask)
    
    if (show_images):
        cv2.imshow("Original", img)
        cv2.imshow("Keypoints", key_mask)
        cv2.imshow("Final", hand_to_binary(cropped, skin_mask, key_mask))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed
