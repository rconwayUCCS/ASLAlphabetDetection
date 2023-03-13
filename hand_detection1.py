import cv2
import numpy as np

#This code is taken and modified from https://github.com/sudonitin/Hand_detection_tracking_opencv-

def generate_mask(frame):
    kernelOpen = np.ones((5,5))#if jiggers are present other than yellow area
    kernelClose = np.ones((20,20)) #if jiggers are present in yellow area

    lb = np.array([0,0,0])
    ub = np.array([43,255,255])
    img_width = 200
    img_height = 200

    flipped = cv2.resize(frame,(img_width,img_height))
    
    #use HSV of yellow to detect only yellow color
    imgSeg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    imgSeg = cv2.resize(imgSeg,(img_width,img_height))
    #masking and filtering all shades of yellow
    mask = cv2.inRange(imgSeg, lb, ub)
    mask = cv2.resize(mask,(img_width,img_height))
    
    #apply morphology to avoid jiggers
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskOpen = cv2.resize(maskOpen,(img_width,img_height))
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    maskClose = cv2.resize(maskClose,(img_width,img_height))
    
    final = maskClose
    conts, h = cv2.findContours(maskClose,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    if(len(conts)!=0): #draws the contours of that object which has the highest
        b = max(conts, key=cv2.contourArea)
        c = min(conts, key=cv2.contourArea)
        west = tuple(b[b[:, :, 0].argmin()][0]) #gives the co-ordinate of the left extreme of contour
        east = tuple(b[b[:, :, 0].argmax()][0]) #above for east i.e right
        north = tuple(b[b[:, :, 1].argmin()][0])
        south = tuple(b[b[:, :, 1].argmax()][0])
        centre_cords = [(west[0]+east[0])/2, (north[1]+south[1])/2]

        #centre_y = (north[1]+south[1])/2
    
        cv2.drawContours(flipped, b, -1, (0,255,0), 3)
        cv2.drawContours(flipped, c, -1, (0,255,0), 3)
        cv2.circle(flipped, west, 6, (255,0,255), -1)
        cv2.circle(flipped, east, 6, (0,255,0), -1)
        cv2.circle(flipped, north, 6, (0,0,255), -1)
        cv2.circle(flipped, south, 6, (255,0,0), -1)
        cv2.circle(flipped, (int(centre_cords[0]),int(centre_cords[1])), 6, (40,100,255), -1)#plots centre of the area
    
    #cv2.imshow('video', flipped)
    #cv2.imshow('Closed Mask', maskClose)
    print("Upper:", ub)
    print("Lower:", lb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return maskClose


#img = cv2.imread("CTest.jpg")
#generate_mask(img)

#cap.release()
#cv2.destroyAllWindows()
