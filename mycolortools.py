
import math
import cv2
import numpy as np
import os
import sys

colorsRGB = {
        'white': (255,255,255),
        'black': (0,0,0),
        'red': (255,0,0),
        'green': (0,255,0),
        'blue': (0,0,255)
    }

def color_distance(color1,color2):
    color1=color1.astype('int')
    color2=color2.astype('int')
    color_dist=(color1[0]-color2[0])**2+(color1[1]-color2[1])**2+(color1[2]-color2[2])**2
    color_dist=np.sqrt(color_dist)
    return color_dist

def color_distance_2d(image1,image2):
    im1=image1.astype('int')
    im2=image2.astype('int')
    color_dist=(im1[:,:,0]-im2[:,:,0])**2+(im1[:,:,1]-im2[:,:,1])**2+(im1[:,:,2]-im2[:,:,2])**2
    color_dist=np.sqrt(color_dist)
    return color_dist

def color_distance_1d(var1,var2):
    var1=var1.astype('int')
    var2=var2.astype('int')
    color_dist=(var1[:,0]-var2[:,0])**2+(var1[:,1]-var2[:,1])**2+(var1[:,2]-var2[:,2])**2
    color_dist=np.sqrt(color_dist)
    return color_dist

def parse_color(string):
    carr=string.split(" ")
    color=np.zeros([4],'uint8')
    for i in range(len(carr)):
        color[i]=int(carr[i])
    return color


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

def flood_select(image,mask,position,ref_color,threshold):
    shape=image.shape
    if mask[position]==1:
        return mask
    else:
        if color_distance(image[position],ref_color)>threshold:
            return mask
    mask[position]=1
    if shape[1]>position[1]+1:
        mask=flood_select(image,mask,(position[0],position[1]+1),ref_color,threshold)
    if position[0]>0:
        mask=flood_select(image,mask,(position[0]-1,position[1]),ref_color,threshold)
    if shape[0]>position[0]+1:
        mask=flood_select(image,mask,(position[0]+1,position[1]),ref_color,threshold)
    if position[1]>0:
        mask=flood_select(image,mask,(position[0],position[1]-1),ref_color,threshold)
    return mask

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
            
def color_to_RGB(string):
    return colorsRGB.get(string)

def color_to_BGR(string):
    color=colorsRGB.get(string)
    return (color[2],color[1],color[0])

def mean_color(array):
    array=np.array(array)
    mean=np.mean(array,axis=0)
    return mean

def median_color(array):
    array=np.array(array)
    median=np.median(array,axis=0)
    return median

def color_spread(array):
    color_mean=np.array([mean_color(array)])
    distances=[]
    for i in range(len(array)):
        distances.append(color_distance_1d(np.array([array[i]]),color_mean))
    color_spread=np.mean(distances)
    return color_spread
