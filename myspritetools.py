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
    def __init__(self,game,object,frame,pace=1,size=1,rotate=0,directory=''):
        self.data=[]
        self.visible=[]
        self.center=[]
        self.pace=pace
        self.size=size
        self.rotate=rotate
        if directory != '':
            self.full_path=directory
            frame='all'
        else:
            self.full_path=sprite_fullpath(game,object,frame)
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
        self.recenter()
        self.n_image=1

    def read_dir(self,path):
        if os.path.isdir(path)==False:
            print("No such sprite directory")
            return
        files=os.listdir(path)
        for thisim in files:
            image=cv2.imread(path+"/"+thisim,cv2.IMREAD_UNCHANGED)
            self.data.append(image)
        self.n_image=len(files)
        self.recenter()

    def resize(self,size):
        if size != 1.0:
            for i in range(self.n_image):
                s_image=self.data[i]
                image=cv2.resize(s_image,(0,0),fx=size,fy=size,interpolation=cv2.INTER_LINEAR)
                self.data[i]=image
            self.recenter()

    def pad(self,x,y):
        for i in range(self.n_image):
            new_im=np.zeros([y,x,4],'uint8')
            old_im=self.data[i]
            xdiff=x-old_im.shape[1]
            ydiff=y-old_im.shape[0]
            new_im[int(ydiff/2):int(ydiff/2)+old_im.shape[0],int(xdiff/2):int(xdiff/2)+old_im.shape[1]]=old_im
            self.data[i]=new_im
            self.recenter()

    def recenter(self):
        self.visible=[]
        self.center=[]
        for i in range(self.n_image):
            image=self.data[i]
            visible=np.nonzero(image[:,:,3])
            self.visible.append(visible)
            self.center.append([np.mean(visible[1]),np.mean(visible[0])])

    def compute_fit(self,image,i):
        visible=self.visible[i]
        sprite_image=self.data[i][visible].astype('float')
        sprite_image=sprite_image[...,0:3]
        fit_quality=np.sum((image[visible].astype('float')-sprite_image)**2)
        fit_quality=math.sqrt(fit_quality)/len(visible[0]/3)
        return fit_quality

    def fit(self,background,position,xyrange):
        chimin=1.0e30
        for i in range(self.n_image):
            sprite_image=self.data[i]
            center0=self.center[i]
            shape=sprite_image.shape

            for x in range(xyrange):
                for y in range(xyrange):
                    center=(center0[0]+(int((x-xyrange/2))),center0[1]+(int((y-xyrange/2))))
                    true_pos=(position[0]-int(center[0]),position[1]-int(center[1]))
                    back_sub=background[true_pos[1]:true_pos[1]+shape[0],true_pos[0]:true_pos[0]+shape[1]]
                    chival=self.compute_fit(back_sub,i)
                    if chival<chimin:
                        chimin=chival
                        dxmin=int(x-xyrange/2)
                        dymin=int(y-xyrange/2)
                        imin=i
        return [chimin,dxmin,dymin,imin]

    def maxsize(self):
        maxsize=0
        for i in range(self.n_image):
            maxhere=np.max(self.visible[i])
            if maxhere>maxsize:
                maxsize=maxhere
        return maxsize

    def overlay(self,background,position,frames=0,path=0):
        overlay=[]
        pace_count=0
        s_index=0
        if frames==0:
            frames=len(background)
        else:
            background=[background]*frames
        if path == 0:
            path=sprite_path([(position[0],position[1])]*frames)
        for i in range(frames):
            pace_count+=1
            img=self.data[s_index]
            position=path.path[i]
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
                img_new=self.data[s_index]
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
    spath=sprite_fullpath(game,object,frame)
    dirname,filename=os.path.split(spath)
    os.makedirs(dirname,exist_ok=True)
    cv2.imwrite(spath,sprite_im)

def sprite_fullpath(game,object,frame):
    root_dir='C:/Users/sp4ce/OneDrive/Documents/Sprites/'
    if frame == "all":
        full_path=root_dir+game+"/"+object
    else:
        full_path=root_dir+game+"/"+object+"/"+frame+".png"

    return full_path

