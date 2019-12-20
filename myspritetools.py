#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python37/python

import math
import cv2
import numpy as np
import os
import sys
import myimutils
from scipy import ndimage

class Sprite:
    def __init__(self,game,object,frame):
        self.data=[]
        self.visible=[]
        self.center=[]
        self.full_path=sprite_path(game,object,frame)
        if frame == "all":
            self.read_dir(self.full_path)
        else:
            self.read_image(self.full_path)
        
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
        self.pace=2

    def overlay(self,background,position):
        overlay=[]
        pace_count=0
        s_index=0
        for i in range(len(background)):
            pace_count+=1
            if pace_count==self.pace:
                pace_count=0
                s_index+=1
            if s_index==self.n_image:
                s_index=0
            center=self.center[s_index]
            true_pos=(position[0]-int(center[0]),position[1]-int(center[1]))
            overlaid=myimutils.add_images(self.data[s_index],background[i],true_pos[0],true_pos[1])
            overlay.append(overlaid)
        return overlay

def add_sprite_image(image,game,object,frame):
    visible=np.nonzero(image[:,:,3])
    mins=np.amin(visible,axis=1)
    maxes=np.amax(visible,axis=1)
    sprite_im=image[mins[0]:maxes[0]+1,mins[1]:maxes[1]+1,:]
    spath=sprite_path(game,object,frame)
    dirname,filename=os.path.split(spath)
    os.makedirs(dirname)
    cv2.imwrite(spath,sprite_im)

def sprite_path(game,object,frame):
    root_dir="../Sprites/"
    if frame == "all":
        full_path=root_dir+game+"/"+object
    else:
        full_path=root_dir+game+"/"+object+"/"+frame+".png"

    return full_path
