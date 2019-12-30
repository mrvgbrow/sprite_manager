#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import math
import cv2
import numpy as np
import os
import imutils
import sys
import genutils as genu
import myimutils as myim
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
    def smooth(self,scale):
        new_path=[]
        for i in range(len(self.path)):
            xav=0.0
            yav=0.0
            functot=0.0
            funcref=0.0
            for k in range(2*scale+1):
                funcref+=genu.gaussian_function(k/scale)
            for j in range(max(0,i-2*scale),min(len(self.path),i+2*scale+1)):
                if self.path[j][0]>=0: 
                    xav+=genu.gaussian_function((i-j)/scale)*self.path[j][0]
                    yav+=genu.gaussian_function((i-j)/scale)*self.path[j][1]
                    functot+=genu.gaussian_function((i-j)/scale)
            if functot>0.8*funcref:
                xval=int(round(xav/functot))
                yval=int(round(yav/functot))
            else:
                xval=-1
                yval=-1
            new_path.append((xval,yval))
        self.path=new_path
    
    def overlay(self,background,width=1):
        for i in range(len(background)):
            for j in range(i+1,len(background)):
                myim.fill_square(background[j],(self.path[i][0],self.path[i][1]),width)
        return background

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

    def save_rotations(self,interval):
        for o in range(interval,360,interval):
            for i in range(self.n_image):
                self.data.append(myim.rotate_image(self.data[i],o))
        self.n_image=len(self.data)
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
        interval=int(xyrange/2)
        imin=0
        dxmin=0
        dymin=0
        for i in range(self.n_image):
            sprite_image=self.data[i]
            center0=self.center[i]
            shape=sprite_image.shape
            if chimin==0:
                break

            for dx in range(-interval,interval+1):
                if chimin==0:
                    break
                for dy in range(-interval,interval+1):
                    center=(center0[0]+dx,center0[1]+dy)
                    true_pos=(position[0]-int(center[0]),position[1]-int(center[1]))
                    if myim.check_in_image(background,true_pos,shape):
                        back_sub=myim.window_from_image(background,true_pos,shape)
                    else:
                        continue
                    chival=self.compute_fit(back_sub,i)
                    if chival<chimin:
                        chimin=chival
                        dxmin=dx
                        dymin=dy
                        imin=i
                        if chimin==0:
                            break
        center=np.array([int(self.center[imin][0])+dxmin,int(self.center[imin][1])+dymin])
        feathered_indices=self.feather_visible(0,imin)
        back_indices=myim.shift_indices(feathered_indices,np.array(position)-center)
        return [chimin,dxmin,dymin,imin,back_indices]

    def maxsize(self):
        maxsize=0
        maxsize_x=0
        maxsize_y=0
        for i in range(self.n_image):
            maxhere_x=np.max(self.visible[i][1])
            maxhere_y=np.max(self.visible[i][0])
            maxhere=np.max(self.visible[i])
            if maxhere>maxsize:
                maxsize=maxhere
            if maxhere_x>maxsize_x:
                maxsize_x=maxhere_x
            if maxhere_y>maxsize_y:
                maxsize_y=maxhere_y
        return (maxsize,maxsize_x,maxsize_y)

    def center_to_corner(position,center):
        return (position[0]-center[0],position[1]-center[1])

    def overlay(self,background,path,frames=0):
        overlay=[]
        pace_count=0
        s_index=0
        if frames==0:
            frames=len(background)
        else:
            background=[background]*frames
        if type(path) is tuple:
            path=sprite_path([(path[0],path[1])]*frames)
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
                img_new=myim.rotate_image(img,15*self.rotate*i)
                center=self.center[s_index]
                center=(center[0]+(img_new.shape[1]-img.shape[1])/2,center[1]+(img_new.shape[0]-img.shape[0])/2)
            else:
                img_new=self.data[s_index]
                center=self.center[s_index]
            true_pos=(position[0]-int(center[0]),position[1]-int(center[1]))
            if path.path[i][0]==-1:
                overlay.append(background[i])
            else:
                overlay.append(myim.add_images(img_new,background[i],true_pos[0],true_pos[1]))
        return overlay

    def feather_visible(self,pixels,i):
        data=self.data[i]
        mask=np.zeros([data.shape[0]+pixels,data.shape[1]+pixels],'float')
        new_visible=myim.shift_indices(self.visible[i],[int(pixels/2),int(pixels/2)])
        mask[new_visible]=1.0
        threshold=genu.gaussian_function(pixels)
        blurred=cv2.GaussianBlur(mask,(7,7),1)
        indices_new=np.where(blurred > threshold)
        indices_new_shifted=myim.shift_indices(indices_new,[-int(pixels/2),-int(pixels/2)])
        return indices_new_shifted

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
    root_dir='C:/Users/sp4ce/Google Drive/Documents/Sprites/'
    if frame == "all":
        full_path=root_dir+game+"/"+object
    else:
        full_path=root_dir+game+"/"+object+"/"+frame+".png"

    return full_path

def add_sprite(images,game,object,frame="all",size=1.0,rotate=0.0,pace=1,path=[0]):
    if (images[0].shape[2]==3):
        images=myim.add_alpha_channel(images)
    mysprite=Sprite(game,object,frame,pace=pace,size=size,rotate=rotate)
    if path==[0]:
        path=myim.capture_point(images[0])
    return mysprite.overlay(images,path)
