import numpy as np
import cv2
#from matplotlib import pyplot as plt
from Keypoints import get_points_and_mask, color_and_crop

#This code was taken from https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Image%20Preprocessing/image_processing.py

def hand_to_binary(frame, skinMask, key_mask): 

    #frame = cv2.imread(path)
    #frame = cv2.resize(image,(96,96))
    # downsize it to reduce processing time
    #cv2.imshow("original",frame)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    key_mask = cv2.cvtColor(key_mask, cv2.COLOR_BGR2GRAY)
    key_mask = cv2.medianBlur(key_mask, 5)
    #key_skin = cv2.bitwise_and(frame, frame, mask = key_mask)
    #cv2.imshow("Blended", key_skin)

    #print(frame.shape)
    #tuned settings
    #lowerBoundary = np.array([0,40,30],dtype="uint8")
    #upperBoundary = np.array([43,255,254],dtype="uint8")

    #skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)

    # apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
    #skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)
    #skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    #skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    key_mask = cv2.morphologyEx(key_mask, cv2.MORPH_OPEN, kernel)
    #key_mask = cv2.morphologyEx(key_mask, cv2.MORPH_CLOSE, kernel)
    #key_mask = cv2.erode(key_mask, kernel, iterations = 2)
    #key_mask = cv2.dilate(key_mask, kernel, iterations = 2)
    cv2.imshow("key_mask", key_mask)

    #lowerBoundary = np.array([170,80,30],dtype="uint8")
    #upperBoundary = np.array([180,255,250],dtype="uint8")

    #skinMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)
    #cv2.imshow("Mask2", skinMask2)
    skinMask = cv2.bitwise_and(skinMask, key_mask)
    cv2.imshow("Mask2", skinMask)
    #skinMask = cv2.addWeighted(skinMask,0.5,key_mask,0.5,0.0)
    #print(skinMask.flatten())
    #print(skinMask.shape)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    cv2.imshow("masked",skin) # Everything apart from skin is shown to be black

    h,w = skin.shape[:2]
    bw_image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)  # Convert image from HSV to BGR format
    bw_image = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)  # Convert image from BGR to gray format
    bw_image = cv2.GaussianBlur(bw_image,(5,5),0)  # Highlight the main object
    bw_image = cv2.threshold(bw_image, 32, 255, cv2.THRESH_BINARY)[1]
    #threshold = 1
    #for i in range(h):
    #    for j in range(w):
    #        if bw_image[i][j] > threshold:
    #           bw_image[i][j] = 0
    #        else:
    #           bw_image[i][j] = 255

    #cv2.imshow("thresholded",bw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bw_image

def make_outline(frame):    
    frame = cv2.resize(frame,(128,128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #cv2.imshow("original",converted2)

    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    #cv2.imshow("masked",skinMask)
    
    skinMask = cv2.medianBlur(skinMask, 5)
    
    skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
    
    img2 = cv2.Canny(skin,100,40, L2gradient=True)
    img2 = cv2.resize(img2,(256,256))
    return img2

def fill_outline(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold
    thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    
    # get the (largest) contour
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    
    # draw white filled contour on black background
    result = np.zeros_like(image)
    cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
    
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result
        
img = cv2.imread("C100.jpg")
img = cv2.resize(img,(128,128))

#key_points1= func2(img)
key_points2, skin_mask = get_points_and_mask(img)

#avg_x1, avg_y1 = avg_kp(key_points1)
#avg_x2, avg_y2 = avg_kp(key_points2)


#cropped1 = img[avg_x1 - 64:avg_x1 + 63, avg_y1 - 64:avg_y1 + 63]

key_mask, cropped, skin_mask = color_and_crop(key_points2, img, skin_mask)

cv2.imshow("keypoints", key_mask)
cv2.imshow("Hand to binary", hand_to_binary(cropped, skin_mask, key_mask))

#cropped1 = generate_mask(cropped2)
#cropped1 = cv2.bitwise_not(cropped1)
#cropped3 = make_outline(cropped2)
#cv2.imshow("HD2", cropped3)
#cropped3 = fill_outline(cropped3)
#cropped3 = fill_outline(cropped3)

#cv2.imshow("HD2", cropped3)

#cropped1 = cv2.resize(cropped1,(128,128))
#cropped3 = cv2.resize(cropped3,(128,128))
#dst = cv2.addWeighted(cropped1,1,cropped3,1,0)
#
#cv2.imshow("Output", dst)

#fill_outline(cropped3)

#cv2.imshow("HD1", cropped1)
#cropped2 = cv2.resize(cropped2,(256,256))
#cv2.imshow("Cropped", cropped2)
#cv2.imshow("Original", img)
#cv2.imshow("HD2", cropped3)

cv2.waitKey(0)
cv2.destroyAllWindows()
