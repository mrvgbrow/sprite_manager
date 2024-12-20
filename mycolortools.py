
import json
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

class palette:
    def __init__(self,data,game='None',filename='None'):
        self.root_dir='C:/Users/sp4ce/Google Drive/Documents/Palettes'
        if game != 'None':
            path=self.root_dir+'/'+game+'/palette.json'
            color_dict=self.read_palette(path)
            self.data=color_dict
        else:
            self.data=data

    def save_palette(self,game,filename='None',append=0):
        gamedir=self.root_dir+'/'+game+'/'
        os.makedirs(gamedir,exist_ok=True)
        with open(gamedir+'palette.txt','w') as file:
            file.write(json.dumps(self.data))

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

def click_point(event,x,y,flags,param):
    global refPt,clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt=(x,y)
        clicked=1

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

def bucket_select(image,title='Image',threshold=25):
    global refPt,clicked,clicked2

    refPt=(0,0)
    clicked=0
    inds=[]
    info_window=np.zeros((40,500,3))
    cv2.namedWindow(title,flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title,click_point)
    while True:
        info_window*=0
        info_string="(x,y) = "+str(refPt)
        cv2.putText(info_window,info_string,(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow(title,image.astype('uint8'))
        cv2.imshow('Info',info_window)
        key = cv2.waitKey(1) & 0xFF
        if clicked==1:
            mask=np.zeros((image.shape[0],image.shape[1]),'uint8')
            ref_color=image[refPt[1],refPt[0]]
            mask=flood_select(image,mask,(refPt[1],refPt[0]),ref_color,threshold)
            inds=np.nonzero(mask==1)
            image[inds]=[255,255,255]
            clicked=0
        if key==ord('x'):
            cv2.destroyAllWindows()
            return 0
        if key==ord('s'):
            cv2.destroyAllWindows()
            return np.nonzero(mask==1)


def color_combine(image,color_dict='None'):
    im=np.zeros([image.shape[0],image.shape[1]],'int')
    im=1000000*image[...,0].astype('int')+1000*image[...,1].astype('int')+image[...,2].astype('long')
    if color_dict!='None':
        for k in color_dict:
            im[np.nonzero(im==int(k))]=color_dict[k]
    return im

def color_expand(image,d='None'):
    image=image.astype('int')
    im=np.zeros([image.shape[0],image.shape[1],3],'uint8')
    if d!= 'None':
        for k in d:
            image[np.nonzero(image==k)]=int(d[k])
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

def find_barrier(image,start,direction,threshold=25,bcolor='0'):
    ref_color=image[(start[1],start[0])]
    cdiff=0
    if bcolor != '0':
        ref_color=parse_color(bcolor)
        threshold=-threshold
        cdiff=-1000
    position=start
    while cdiff<threshold and position[0]>0 and position[1]>0 and position[0]<image.shape[1] and position[1]<image.shape[0]:
        position=np.add(direction,position)
        cdiff=color_distance(ref_color,image[(position[1],position[0])])
        if bcolor != '0':
            cdiff=-cdiff
    return (position[0],position[1])

def make_color_dict(images):
    d={}
    val=0
    imlist2=[]
    for i in range(len(images)):
        im2=np.empty((images[0].shape[0],images[0].shape[1]),'uint8')
        image=images[i]
        im=1000000*image[...,0].astype('int')+1000*image[...,1].astype('int')+image[...,2].astype('long')
        for n in range(im.shape[1]):
            for m in range(im.shape[0]):
                try: 
                    im2[m,n]=d[int(im[m,n])]
                except:
                    im2[m,n]=val
                    d[int(im[m,n])]=val
                    val+=1
        imlist2.append(im2)
    return imlist2,d

            
def read_palette(game):
    root_dir='C:/Users/sp4ce/Google Drive/Documents/Palettes'
    if game != 'None':
        path=root_dir+'/'+game+'/palette.txt'
    if os.path.isfile(path)==False:
        print("Palette not found.")
        return
    with open(path,'r') as file:
        d=json.loads(file.read())
    return d

