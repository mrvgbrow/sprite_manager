
import math
import cv2
import numpy as np
import os
import sys

def color_distance(image1,image2):
    im1=image1.astype('int')
    im2=image2.astype('int')
    color_dist=(im1[:,:,0]-im2[:,:,0])**2+(im1[:,:,1]-im2[:,:,1])**2+(im1[:,:,2]-im2[:,:,2])**2
    color_dist=np.sqrt(color_dist)
    return color_dist

def color_combine(image):
    im=np.zeros([image.shape[0],image.shape[1]],'int')
    im=1000000*image[...,0].astype('int')+1000*image[...,1].astype('int')+image[...,2].astype('int')
    return im

def color_expand(image):
    im=np.zeros([image.shape[0],image.shape[1],3],'uint8')
    im[:,:,0]=image[:,:]/1000000
    im[:,:,1]=(image[:,:]-1000000*im[:,:,0].astype('int'))/1000
    im[:,:,2]=image[:,:]-1000000*im[:,:,0]-1000*im[:,:,1]
    return im

def click_color(event,x,y,flags,param):
    global refPt,clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt=(x,y)
        clicked=1

def select_color(image,color1,fuzz,invert=False):
    blue=image[:,:,0].astype('int')
    green=image[:,:,1].astype('int')
    red=image[:,:,2].astype('int')
    if invert==False:
        indices=np.nonzero((color1[0]-blue)**2+(color1[1]-green)**2+(color1[2]-red)**2<=fuzz**2)
    else:
        indices=np.nonzero((color1[0]-blue)**2+(color1[1]-green)**2+(color1[2]-red)**2>fuzz**2)
    return indices

def imshow_get_color(image,title,exit_char):
    global refPt,clicked

    clicked=0
    color=[0,0,0]
    refPt=(0,0)
    cv2.namedWindow(title)
    cv2.setMouseCallback(title,click_color)
    while True:
        cv2.imshow(title,image)
        key = cv2.waitKey(1) & 0xFF
        
        if key==ord(exit_char):
            break
        if clicked==1:
            color= image[refPt[1],refPt[0],:]
            break
    cv2.destroyAllWindows()
    return color
            
