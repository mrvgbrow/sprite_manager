#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python37/python

import math
import cv2
import numpy as np
import os
import imutils
import sys
import myimutils
from scipy import ndimage

class sprite_path:
    def __init__(self,path,sizes=[],angles=[]):
        self.path=np.array(path)
        if len(sizes)!=len(path):
            self.sizes=self.path*0.0
        else:
            self.sizes=np.array(sizes)
        if len(angles)!=len(angles):
            self.angles=self.path*0.0
        else:
            self.angles=np.array(angles)

class Sprite:
    def __init__(self,game,object,frame,pace=1,size=1,rotate=0,path=sprite_path([0.0])):
        self.data=[]
        self.visible=[]
        self.center=[]
        self.pace=pace
        self.size=size
        self.rotate=rotate
        self.path=path
        self.full_path=sprite_path(game,object,frame)
        if frame == "all":
            self.read_dir(self.full_path)
        else:
            self.read_image(self.full_path)
        self.resize(size)
        
    def read_image(self,path):
        if os.path.isfile(path)==False:
            print("Sprite not found.")
            return
        data=cv2.imread(self.full_path,cv2.IMREAD_UNCHANGED)
        self.data.append(data)
        visible=np.nonzero(data[:,:,3])
        self.visible.append(visible)
        self.center.append([np.mean(visible[0]),np.mean(visible[1])])
        self.n_image=1

    def read_dir(self,path):
        if os.path.isdir(path)==False:
            print("No such sprite directory")
            return
        files=os.listdir(path)
        for thisim in files:
            image=cv2.imread(path+"/"+thisim,cv2.IMREAD_UNCHANGED)
            self.data.append(image)
            visible=np.nonzero(image[:,:,3])
            self.visible.append(visible)
            self.center.append([np.mean(visible[0]),np.mean(visible[1])])
        self.n_image=len(files)

    def resize(self,size):
        self.size=size
        if size != 1.0:
            self.data_overlay=[]
            for i in range(self.n_image):
                s_image=self.data[i]
                center=self.center[i]
                self.center[i]=(float(center[0])*size,float(center[1])*size)
                self.data_overlay.append(cv2.resize(s_image,(0,0),fx=size,fy=size,interpolation=cv2.INTER_AREA))
        else:
            self.data_overlay=self.data

    def overlay(self,background,position):
        overlay=[]
        pace_count=0
        s_index=0
        for i in range(len(background)):
            pace_count+=1
            img=self.data_overlay[s_index]
            if pace_count==self.pace:
                pace_count=0
                s_index+=1
            if s_index==self.n_image:
                s_index=0
            if self.rotate>0:
                img_new=myimutils.rotate_image(img,15*self.rotate*i)
                center=self.center[s_index]
                center=(center[0]+(img_new.shape[1]-img.shape[1])/2,center[1]+(img_new.shape[0]-img.shape[0])/2)
            else:
                img_new=self.data_overlay[s_index]
                center=self.center[s_index]
            true_pos=(position[0]-int(center[0]),position[1]-int(center[1]))
            overlaid=myimutils.add_images(img_new,background[i],true_pos[0],true_pos[1])
            overlay.append(overlaid)
        return overlay

def add_sprite_image(image,game,object,frame):
    visible=np.nonzero(image[:,:,3])
    mins=np.amin(visible,axis=1)
    maxes=np.amax(visible,axis=1)
    sprite_im=image[mins[0]:maxes[0]+1,mins[1]:maxes[1]+1,:]
    spath=sprite_path(game,object,frame)
    dirname,filename=os.path.split(spath)
    print(dirname,filename)
    os.makedirs(dirname,exist_ok=True)
    cv2.imwrite(spath,sprite_im)

def sprite_path(game,object,frame):
    root_dir='C:/Users/sp4ce/OneDrive/Documents/Sprites/'
    if frame == "all":
        full_path=root_dir+game+"/"+object
    else:
        full_path=root_dir+game+"/"+object+"/"+frame+".png"

    return full_path

