import numpy as np
import cv2
from matplotlib import pyplot as plt
from hand_detection1 import generate_mask

#This code was taken from https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition/blob/master/Image%20Preprocessing/image_processing.py

def func(image):    
    #frame = cv2.imread(path)
    frame = cv2.resize(image,(96,96))
    # downsize it to reduce processing time
    #cv2.imshow("original",frame)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #print(frame.shape)
    #tuned settings
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([30,255,254],dtype="uint8")

    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)

    # apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    lowerBoundary = np.array([170,80,30],dtype="uint8")
    upperBoundary = np.array([180,255,250],dtype="uint8")

    skinMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask2,0.5,0.0)
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
    threshold = 1
    for i in range(h):
        for j in range(w):
            if bw_image[i][j] > threshold:
               bw_image[i][j] = 0
            else:
               bw_image[i][j] = 255


    #cv2.imshow("thresholded",bw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bw_image

def func3(frame):    
    #frame = cv2.imread(path)
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
    
    #cv2.imshow("masked2",skin)
    img2 = cv2.Canny(skin,60,60)
    #cv2.imshow("edge detection",img2)
    img2 = cv2.resize(img2,(256,256))
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img2,None)

    #print(len(des2))
    #img2 = cv2.drawKeypoints(img2,kp,None,color=(0,255,0), flags=0)
    #plt.imshow(img2),plt.show()
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return kp
#func("001.jpg")

def make_outline(frame):    
    #frame = cv2.imread(path)
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
    
    #cv2.imshow("masked2",skin)
    img2 = cv2.Canny(skin,60,60)
    #cv2.imshow("edge detection",img2)
    img2 = cv2.resize(img2,(256,256))
    return img2

#Returns the average of all key points
def avg_kp(kp):
    length = len(kp)
    x_avg = 0
    y_avg = 0

    for point in kp:
        cords = point.pt
        x_avg += cords[0]
        y_avg += cords[1]

    x_avg = int(x_avg/length)
    y_avg = int(y_avg/length)
    
    return x_avg, y_avg
        


#image = func("CTest.jpg")

img = cv2.imread("CTest.jpg")

#key_points1= func2(img)
key_points2 = func3(img)

#avg_x1, avg_y1 = avg_kp(key_points1)
avg_x2, avg_y2 = avg_kp(key_points2)

img = cv2.resize(img,(256,256))
#cropped1 = img[avg_x1 - 64:avg_x1 + 63, avg_y1 - 64:avg_y1 + 63]
cropped2 = img[avg_x2 - 64:avg_x2 + 63, avg_y2 - 64:avg_y2 + 63]

#cropped1 = generate_mask(cropped1)
cropped1 = generate_mask(cropped2)
cropped3 = make_outline(cropped2)

cv2.imshow("HD1", cropped1)
cropped2 = cv2.resize(cropped2,(256,256))
cv2.imshow("Cropped", cropped2)
cv2.imshow("Original", img)
cv2.imshow("HD2", cropped3)

#cv2.imshow("First", image)
#cv2.imshow("Second", image1)
#cv2.imshow("Third", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imshow('video', image)
#while True:
    #pass



