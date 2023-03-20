import numpy as np
import cv2



#Uses a skin mask and canny edge detection to find key points.
#This will be used later as the first of 2 masks, and to crop the image
def get_points_and_mask(frame):    
    frame = cv2.resize(frame,(128,128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV

    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    
    skinMask = cv2.medianBlur(skinMask, 5)
    
    skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
    
    img2 = cv2.Canny(skin,60,60)

    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img2,None)

    #img_orb = cv2.drawKeypoints(img2,kp,None,color=(0,255,0), flags=0)
    #cv2.imshow("orb", img_orb)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return kp, skinMask

#Colors a white circle at all keypoints
#Then crops the image to 64x64 pixels around the hand.
#Returns the newly colored image and the cropped original image
def color_and_crop(kp, img, skin_mask):
    length = len(kp)
    avg_x = 0
    avg_y = 0

    #create a black image and convert it to color
    blank = np.zeros_like(img)

    for point in kp:
        cords = point.pt
        avg_x += cords[0]
        avg_y += cords[1]
        blank = cv2.circle(blank, (int(cords[0]),int(cords[1])), radius=3, color=(255,255,255), thickness=-1)
        
    #Calculate the average of all keypoints, and crop image to 64x64 pixels centered around the average
    if length == 0:
        avg_x = 64
        avg_y = 64
    else:
        avg_x = int(avg_x/length)
        avg_y = int(avg_y/length)

    cropped_blank = blank[avg_x - 32:avg_x + 31, avg_y - 32:avg_y + 31]
    cropped_img = img[avg_x - 32:avg_x + 31, avg_y - 32:avg_y + 31]
    cropped_skin = skin_mask[avg_x - 32:avg_x + 31, avg_y - 32:avg_y + 31]

    return cropped_blank, cropped_img, cropped_skin


