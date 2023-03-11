import cv2
import numpy as np
from sklearn.metrics import pairwise
import tkinter as tk
from tkinter import ttk

#This code is taken and modified from https://github.com/sudonitin/Hand_detection_tracking_opencv-

#cap = cv2.VideoCapture(0)
kernelOpen = np.ones((5,5))#if jiggers are present other than yellow area
kernelClose = np.ones((20,20)) #if jiggers are present in yellow area

# Set up the GUI sliders
root = tk.Tk()
root.title('Thresholds')

lb = np.array([140,0,0])
ub = np.array([180,255,255])

# Define the function to update the lower and upper bounds
def update_lb_hue(self):
    global lb
    lb[0] = slider_h_low.get()

def update_lb_saturation(self):
    global lb
    lb[1] = slider_s_low.get()

def update_lb_value(self):
    global lb
    lb[2] = slider_v_low.get()

def update_ub_hue(self):
    global ub
    ub[0] = slider_h_high.get()

def update_ub_saturation(self):
    global ub
    ub[1] = slider_s_high.get()

def update_ub_value(self):
    global ub
    ub[2] = slider_v_high.get()


# Define the sliders for lower bounds
frame_low = ttk.Frame(root)
frame_low.pack(side=tk.LEFT, padx=10, pady=10)
ttk.Label(frame_low, text='Lower bounds').grid(column=0, row=0, columnspan=2, sticky=tk.W)

lb_names = ['H', 'S', 'V']
slider_h_low = ttk.Scale(frame_low, from_=0, to=360, orient=tk.HORIZONTAL, length=150, command=update_lb_hue)
slider_h_low.set(140)
slider_h_low.grid(column=0, row=1, padx=5)
ttk.Label(frame_low, text=lb_names[0]).grid(column=1, row=1)

slider_s_low = ttk.Scale(frame_low, from_=0, to=255, orient=tk.HORIZONTAL, length=150, command=update_lb_saturation)
slider_s_low.set(0)
slider_s_low.grid(column=0, row=2, padx=5)
ttk.Label(frame_low, text=lb_names[1]).grid(column=1, row=2)

slider_v_low = ttk.Scale(frame_low, from_=0, to=255, orient=tk.HORIZONTAL, length=150, command=update_lb_value)
slider_v_low.set(0)
slider_v_low.grid(column=0, row=3, padx=5)
ttk.Label(frame_low, text=lb_names[2]).grid(column=1, row=3)

# Define the sliders for upper bounds
frame_high = ttk.Frame(root)
frame_high.pack(side=tk.LEFT, padx=10, pady=10)
ttk.Label(frame_high, text='Upper bounds').grid(column=0, row=0, columnspan=2, sticky=tk.W)

ub_names = ['H', 'S', 'V']
slider_h_high = ttk.Scale(frame_high, from_=0, to=360, orient=tk.HORIZONTAL, length=150, command=update_ub_hue)
slider_h_high.set(180)
slider_h_high.grid(column=0, row=1, padx=5)
ttk.Label(frame_high, text=ub_names[0]).grid(column=1, row=1)

slider_s_high = ttk.Scale(frame_high, from_=0, to=255, orient=tk.HORIZONTAL, length=150, command=update_ub_saturation)
slider_s_high.set(255)
slider_s_high.grid(column=0, row=2, padx=5)
ttk.Label(frame_high, text=ub_names[1]).grid(column=1, row=2)

slider_v_high = ttk.Scale(frame_high, from_=0, to=255, orient=tk.HORIZONTAL, length=150, command=update_ub_value)
slider_v_high.set(255)
slider_v_high.grid(column=0, row=3, padx=5)
ttk.Label(frame_high, text=ub_names[2]).grid(column=1, row=3)

#HSV color range which should be detected
lb = np.array([140,0,0])
ub = np.array([180,255,255])

sat_dir = 1
val_dir = 1


img_width = 200
img_height = 200

def generate_mask():
    #update_bounds()
    frame = cv2.imread(cv2.samples.findFile("C100.jpg"))
    #frame = cv2.imread(cv2.samples.findFile("CTest.jpg"))
    flipped = cv2.flip(frame, 1)
    flipped = cv2.resize(flipped,(img_width,img_height))
    
    #use HSV of yellow to detect only yellow color
    imgSeg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    imgSegFlipped = cv2.flip(imgSeg, 1)
    imgSegFlipped = cv2.resize(imgSegFlipped,(img_width,img_height))
    #masking and filtering all shades of yellow
    mask = cv2.inRange(imgSegFlipped, lb, ub)
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
        other_west = tuple(c[c[:, :, 0].argmin()][0]) #gives the co-ordinate of the left extreme of contour
        other_east = tuple(c[c[:, :, 0].argmax()][0]) #above for east i.e right
        other_north = tuple(c[c[:, :, 1].argmin()][0])
        other_south = tuple(c[c[:, :, 1].argmax()][0])
        centre_cords = [(west[0]+east[0])/2, (north[1]+south[1])/2]

        inner_west = [(west[0] + centre_cords[0]) / 2, (west[1] + centre_cords[1]) / 2]
        inner_east = [(east[0] + centre_cords[0]) / 2, (east[1] + centre_cords[1]) / 2]
        inner_north = [(north[0] + centre_cords[0]) / 2, (north[1] + centre_cords[1]) / 2]
        inner_south = [(south[0] + centre_cords[0]) / 2, (south[1] + centre_cords[1]) / 2]

        #centre_y = (north[1]+south[1])/2
    
        cv2.drawContours(flipped, b, -1, (0,255,0), 3)
        cv2.drawContours(flipped, c, -1, (0,255,0), 3)
        cv2.circle(flipped, west, 6, (255,0,255), -1)
        cv2.circle(flipped, east, 6, (0,255,0), -1)
        cv2.circle(flipped, north, 6, (0,0,255), -1)
        cv2.circle(flipped, south, 6, (255,0,0), -1)
        cv2.circle(flipped, (int(centre_cords[0]),int(centre_cords[1])), 6, (40,100,255), -1)#plots centre of the area
        cv2.circle(flipped, (int(inner_west[0]),int(inner_west[1])), 6, (40,100,255), -1)
        cv2.circle(flipped, (int(inner_east[0]),int(inner_east[1])), 6, (40,100,255), -1)
        cv2.circle(flipped, (int(inner_north[0]),int(inner_north[1])), 6, (40,100,255), -1)
        cv2.circle(flipped, (int(inner_south[0]),int(inner_south[1])), 6, (40,100,255), -1)
    
        hues = [imgSeg[int(inner_west[0]), int(inner_west[1]), 0], imgSeg[int(inner_east[0]), int(inner_east[1]), 0], imgSeg[int(inner_north[0]), int(inner_north[1]), 0], imgSeg[int(inner_south[0]), int(inner_south[1]), 0]]
        sats = [imgSeg[int(inner_west[0]), int(inner_west[1]), 1], imgSeg[int(inner_east[0]), int(inner_east[1]), 1], imgSeg[int(inner_north[0]), int(inner_north[1]), 1], imgSeg[int(inner_south[0]), int(inner_south[1]), 1]]
        vals = [imgSeg[int(inner_west[0]), int(inner_west[1]), 2], imgSeg[int(inner_east[0]), int(inner_east[1]), 2], imgSeg[int(inner_north[0]), int(inner_north[1]), 2], imgSeg[int(inner_south[0]), int(inner_south[1]), 2]]

    cv2.imshow('video', flipped)
    cv2.imshow('Closed Mask', maskClose)
    print("Upper:", ub)
    print("Lower:", lb)
    root.after(500, generate_mask)
    #cv2.destroyAllWindows()



root.update()
root.deiconify()
root.after(500, generate_mask)
root.mainloop()

#cap.release()
cv2.destroyAllWindows()
