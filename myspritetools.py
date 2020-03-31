#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import time
import re
import multiprocessing
import glob
import math
import cv2
import numpy as np
import os
import imutils
import sys
import mycolortools as mycolor
import genutils as genu
import myimutils as myim
from scipy import ndimage

class sprite_path:
    def __init__(self,path,sizes=[],angle1=[],angle2=[],angle3=[],flipx=[],flipy=[],opacity=[]):
        self.path=np.array(path)
        if len(sizes)!=len(path):
            self.sizes=[(1.0,1.0)]*len(self.path)
        else:
            self.sizes=np.array(sizes)
        if len(angle1)!=len(path):
            self.angle1=[0.0]*len(self.path)
        else:
            self.angle1=np.array(angle1)
        if len(angle2)!=len(path):
            self.angle2=[0.0]*len(self.path)
        else:
            self.angle2=np.array(angle2)
        if len(angle3)!=len(path):
            self.angle3=[0.0]*len(self.path)
        else:
            self.angle3=np.array(angle3)
        if len(flipx)!=len(path):
            self.flipx=[0]*len(self.path)
        else:
            self.flipx=np.array(flipx)
        if len(flipy)!=len(path):
            self.flipy=[0]*len(self.path)
        else:
            self.flipy=np.array(flipy)
        if len(opacity)!=len(path):
            self.opacity=[1.0]*len(self.path)
        else:
            self.opacity=np.array(opacity)

    def smooth(self,scale,ref_angle=-999):
        new_path=[]
        new_speed=[]
        new_xspeed=[]
        new_yspeed=[]
        new_angle=[]
        angleold=0
        for i in range(len(self.path)):
            xav=0.0
            yav=0.0
            speedav=0.0
            xspeedav=0.0
            yspeedav=0.0
            angleav=0.0
            functot=0.0
            funcref=0.0
            for k in range(2*scale+1):
                funcref+=genu.gaussian_function(k/scale)
            angle0=math.atan2((self.path[i][0]-self.path[i-1][0]),(self.path[i][1]-self.path[i-1][1]))*180/math.pi
            for j in range(max(0,i-2*scale),min(len(self.path)-1,i+2*scale+1)):
                if self.path[j][0]>=0: 
                    gfunc=genu.gaussian_function((i-j)/scale)
                    xav+=gfunc*self.path[j][0]
                    yav+=gfunc*self.path[j][1]
                    speedav+=gfunc*self.speed[j]
                    xspeedav+=gfunc*self.xspeed[j]
                    yspeedav+=gfunc*self.yspeed[j]
                    if ref_angle==-999:
                        angleav+=gfunc*self.angle1[j]
                    else:
                        # Avoid averaging over angles between different cycles
                        anglehere=math.atan2((self.path[j+1][0]-self.path[j][0]),(self.path[j+1][1]-self.path[j][1]))*180/math.pi
                        if anglehere-angle0>180:
                            anglehere-=360
                        if angle0-anglehere>180:
                            anglehere+=360

                        angleav+=gfunc*anglehere
                    functot+=gfunc
            if functot>0.8*funcref:
                xval=int(round(xav/functot))
                yval=int(round(yav/functot))
                speedval=speedav/functot
                yspeedval=yspeedav/functot
                xspeedval=xspeedav/functot
                angleav=angleav/functot
                if ref_angle!=-999:
                    angleav+=ref_angle
                angleold=angleav
            else:
                xval=-1
                yval=-1
                speedval=-1
                yspeedval=-1
                xspeedval=-1
                angleav=-1
            new_path.append((xval,yval))
            new_speed.append(speedval)
            new_xspeed.append(xspeedval)
            new_yspeed.append(yspeedval)
            new_angle.append(angleav)
        self.path=new_path
        self.speed=new_speed
        self.xspeed=new_xspeed
        self.yspeed=new_yspeed
        self.angle1=new_angle

    def determine_angles(self,ref_angle=0):
        self.angle1[0]=math.atan2((self.path[1][0]-self.path[0][0]),(self.path[1][1]-self.path[0][1]))*180/math.pi+ref_angle
        for i in range(1,len(self.path)):
            self.angle1[i]=math.atan2((self.path[i][0]-self.path[i-1][0]),(self.path[i][1]-self.path[i-1][1]))*180/math.pi+ref_angle

    
    def overlay(self,background,width=2,blend=0.6,color=[255,255,255]):
        for i in range(len(background)):
            for j in range(i+1,len(background)):
                if myim.check_in_image(background[j],(self.path[i][0],self.path[i][1]),(width,width)):
                    myim.fill_square(background[j],(self.path[i][0],self.path[i][1]),width,blend=blend,color=color)
        return background

    def path_length(self,ind1,ind2):
        return (math.sqrt((self.path[ind1][0]-self.path[ind2][0])**2+(self.path[ind1][1]-self.path[ind2][1])**2))

    def input_path(self,images,ref_angle=-999):
        self.path,self.sizes,self.angle1,self.angle2,self.angle3,self.flipx,self.flipy,self.opacity,smooth=myim.capture_path_full(images)
        if ref_angle != -999 and smooth==0:
                smooth=1
        if smooth!=0:
            self.find_speeds()
            self.smooth(smooth,ref_angle=ref_angle)
        return 0

    def calc_g(self,x0,y0,angle,speed,scale,frametime):
        new_path=[]
        x=x0
        y=y0
        vx=speed*math.cos(angle)/scale*frametime
        vy=speed*math.sin(angle)/scale*frametime
        acceleration=9.8*frametime**2/scale
        new_path.append((int(x),int(y)))
        for i in range(1,len(self.path)):
           vy+=acceleration
           x+=vx
           y+=vy
           new_path.append((int(x),int(y)))
        self.path=new_path
        return 0

    def input_trajectory(self,images,speed,scale=1.0,frametime=1.0):
        (point1,point2)=myim.capture_line(images[0])
        angle_init=math.atan2((point2[1]-point1[1]),(point2[0]-point1[0]))
        self.calc_g(point1[0],point1[1],angle_init,speed,scale,frametime)
        self.angle1=[0.0]*len(self.path)
        self.angle2=[0.0]*len(self.path)
        self.angle3=[0.0]*len(self.path)
        self.flipx=[0.0]*len(self.path)
        self.flipy=[0.0]*len(self.path)
        self.opacity=[1.0]*len(self.path)
        return 0

    def find_speeds(self):
        self.speed=[]
        self.xspeed=[]
        self.yspeed=[]
        self.speed.append(self.path_length(1,0))
        self.xspeed.append(self.path[1][0]-self.path[0][0])
        self.yspeed.append(self.path[1][1]-self.path[0][1])
        for i in range(1,len(self.path)-1):
            if self.path[i+1][0]==-1 or self.path[i-1][0]==-1:
                self.speed.append(-1)
                self.xspeed.append(-1)
                self.yspeed.append(-1)
                continue
            self.speed.append(self.path_length(i+1,i-1)/2)
            self.xspeed.append((self.path[i+1][0]-self.path[i-1][0])/2)
            self.yspeed.append((self.path[i+1][1]-self.path[i-1][1])/2)
        self.speed.append(self.path_length(len(self.path)-1,len(self.path)-2))
        self.xspeed.append(self.path[len(self.path)-1][0]-self.path[len(self.path)-2][0])
        self.yspeed.append(self.path[len(self.path)-1][1]-self.path[len(self.path)-2][1])
        return (self.speed,self.xspeed,self.yspeed)

    def write_path(self,outfile):
        f=open(outfile,'w')
        for i in range(len(self.path)):
            f.write(str(self.path[i][0])+","+str(self.path[i][1])+"\n")
        f.close()
            


class Sprite:
    def __init__(self,game,object,frame,pace=1,size=1,rotate=0,directory='',anchor=0):
        self.data=[]
        self.visible=[]
        self.center=[]
        self.pace=pace
        self.size=size
        self.rotate=rotate
        self.anchor=anchor
        if directory != '':
            self.full_path=directory
            frame='all'
        else:
            self.full_path=sprite_fullpath(game,object,frame)
        if frame == "all":
            self.read_dir(self.full_path)
        else:
            pathparts=os.path.splitext(self.full_path)
            if pathparts[1]=='.vdat':
                self.read_vector(self.full_path)
            else:
                self.read_image(self.full_path)
        self.sequence=range(len(self.data))
        self.seqflips=[]
        self.seqoffx=[]
        self.seqoffy=[]
        self.resize(size)
        self.read_colors()

    def read_colors(self):
        colorfile=self.full_path+'/colors.txt'
        if os.path.isfile(colorfile)==0:
            self.colors={}
            return
        f=open(colorfile,'r')
        fl=f.readlines()
        self.colors={}
        i=0
        for line in fl:
            name,vals=line.split('-')
            name=name.rstrip()
            colors=vals.split(',')
            for j in range(len(colors)):
                colors[j]=int(colors[j])
            self.colors[name]=colors
            if i==0:
                self.colors['Active']=self.colors[name]
            i+=1
        
    def read_image(self,path):
        if os.path.isfile(path)==False:
            print("Sprite not found.")
            return
        data=cv2.imread(self.full_path,cv2.IMREAD_UNCHANGED)
        self.data.append(data)
        self.n_image=1
        self.recenter()

    def read_vector(self,path):
        if os.path.isfile(path)==False:
            print("Sprite not found.")
            return
        positions=np.loadtxt(path,dtype='int')
        data=vector2image(positions[:,0],positions[:,1])
        self.data.append(data)
        self.n_image=1
        self.recenter()

    def list_colors(self):
        color_list=[]
        for i in range(self.n_image):
            frame_comb=mycolor.color_combine(self.data[i])
            print(np.unique(frame_comb))

    def swap_colors(self,newcolor):
        color_list=[]
        for i in range(self.n_image):
            frame_comb=mycolor.color_combine(self.data[i])
            for j in range(len(self.colors[newcolor])):
                inds=np.nonzero(frame_comb==self.colors['Active'][j])
                frame_comb[inds]=self.colors[newcolor][j]
            self.data[i]=mycolor.color_expand(frame_comb)
        self.colors['Active']=self.colors[newcolor]

    def read_dir(self,path):
        if os.path.isdir(path)==False:
            print("No such sprite directory")
            return
        files=glob.glob(path+'/'+'*.png')
        files_txt=glob.glob(path+'/'+'*.vdat')
        if len(files_txt)>0:
            self.vector=1
            self.positions=[]
            for thisim in files_txt:
                positions=np.loadtxt(thisim,dtype='int')
                image=vector2image(positions[:,0],positions[:,1])
                self.data.append(image)
                self.positions.append(positions)
                self.n_image=len(files_txt)
        else:
            self.vector=0
            self.positions=[]
            for thisim in files:
                image=cv2.imread(thisim,cv2.IMREAD_UNCHANGED)
                self.data.append(image)
                self.n_image=len(files)
        self.recenter()

    def resize(self,size):
        if size != 1.0:
            if len(self.positions)>0:
                for i in range(self.n_image):
                    self.data[i]=vector2image(self.positions[i][:,0],self.positions[i][:,1],scale=size)
            else:
                for i in range(self.n_image):
                    s_image=self.data[i]
                    self.data[i]=cv2.resize(s_image,(0,0),fx=size,fy=size,interpolation=cv2.INTER_NEAREST)
            self.recenter()
            if len(self.seqoffx)>0:
                for i in range(len(self.seqoffx)):
                    self.seqoffx[i]=int(self.seqoffx[i]*size)
            if len(self.seqoffy)>0:
                for i in range(len(self.seqoffy)):
                    self.seqoffy[i]=int(self.seqoffy[i]*size)

    def save_rotations(self,interval):
        for o in range(interval,360,interval):
            for i in range(self.n_image):
                self.data.append(myim.rotate_image(self.data[i],o))
        self.n_image=len(self.data)
        self.recenter()

    def save_flips(self,ftype=2):
        flip_arr=[0,1,-1]
        if ftype!=2:
            flip_arr=[ftype]
        for o in flip_arr:
            for i in range(self.n_image):
                self.data.append(cv2.flip(self.data[i],flipCode=o))
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
            self.center.append(self.compute_center(visible))
    
    def compute_center(self,visible):
        if self.anchor==0:
            return [np.mean(visible[1]),np.mean(visible[0])]
        elif self.anchor==1:
            return [np.mean(visible[1]),np.max(visible[0])]
        elif self.anchor==2:
            return [np.min(visible[1]),np.mean(visible[0])]
        elif self.anchor==3:
            return [np.mean(visible[1]),np.min(visible[0])]
        elif self.anchor==4:
            return [np.max(visible[1]),np.mean(visible[0])]
        elif self.anchor==5:
            return [np.min(visible[1]),np.min(visible[0])]

    def compute_fit(self,image,i):
        visible=self.visible[i]
        sprite_image=self.data[i][visible].astype('float')
        sprite_image=sprite_image[...,0:3]
        fit_quality=np.sum((image[visible].astype('float')-sprite_image)**2)
        fit_quality=math.sqrt(fit_quality)/len(visible[0]/3)
        return fit_quality

    def fit_one_im(self,imnum,background,position,interval):
        sprite_image=self.data[imnum]
        center0=self.center[imnum]
        shape=sprite_image.shape
        chimin=1.0e30
        dxmin=0
        dymin=0
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
                chival=self.compute_fit(back_sub,imnum)
                if chival<chimin:
                    chimin=chival
                    dxmin=dx
                    dymin=dy
                    if chimin==0:
                        break
        return (chimin,dxmin,dymin)



    def fit(self,background,position,xyrange):
        interval=int(xyrange/2)
        starttime=time.time()
        dxmin0=0
        dymin0=0
        imin=0
        chimin0=1e30
        for i in range(self.n_image):
            (chimin,dxmin,dymin)=self.fit_one_im(i,background,position,interval)
            if chimin<chimin0:
                 chimin0=chimin
                 dxmin0=dxmin
                 dymin0=dymin
                 imin=i
            if chimin0==0:
                 break
        center=np.array([int(self.center[imin][0])+dxmin,int(self.center[imin][1])+dymin])
        feathered_indices=self.feather_visible(2,imin)
        back_indices=myim.shift_indices(feathered_indices,np.array(position)-center)
        return [chimin0,dxmin0,dymin0,imin,back_indices]

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
        if len(self.seqoffx)>0:
            maxsize_x+=max(self.seqoffx)-min(self.seqoffx)
        if len(self.seqoffy)>0:
            maxsize_y+=max(self.seqoffy)-min(self.seqoffy)
        return (maxsize,maxsize_x,maxsize_y)

    def read_sequence(self,name):
        sequence_file=self.full_path+'/sequence.txt'
        if os.path.isfile(sequence_file)==0:
            return
        f=open(sequence_file,'r')
        found=0
        for line in f:
            tag=line.split(":")
            if found==1 and tag[0].strip()=='':
                break
            if tag[0]=='Name':
                namehere=tag[1].strip()
                if namehere==name:
                    found=1
                    continue
            if tag[0]=='Sequence' and found==1:
                sequence=tag[1].split(',')
                self.sequence=[]
                for im in sequence:
                    self.sequence.append(int(im))
            if tag[0]=='Rotation' and found==1:
                sequence=tag[1].split(',')
                self.seqrots=[]
                for rot in sequence:
                    self.seqrots.append(int(rot))
            if tag[0]=='Offset_x' and found==1:
                sequence=tag[1].split(',')
                self.seqoffx=[]
                for offset in sequence:
                    self.seqoffx.append(int(offset))
            if tag[0]=='Offset_y' and found==1:
                sequence=tag[1].split(',')
                self.seqoffy=[]
                for offset in sequence:
                    self.seqoffy.append(int(offset))
            if tag[0]=='Flip' and found==1:
                sequence=tag[1].split(',')
                self.seqflips=[]
                for flip in sequence:
                    self.seqflips.append(int(flip))
        if (found==0):
            print('Sequence '+name+' not found')
        print(self.sequence)

    def center_to_corner(self,position,center):
        return (position[0]-center[0],position[1]-center[1])

    def rotate_data(self,theta):
        for i in range(len(self.data)):
            self.data[i]=myim.rotate_image(self.data[i],theta)
        self.recenter()

    def flip_data(self,axis):
        for i in range(len(self.data)):
            shape=self.data[0].shape
            new_arr=np.zeros(shape,'uint8')
            if axis==1:
                new_arr[0:shape[0],0:shape[1],0:4]=self.data[i][0:shape[0],-1::-1,0:4]
            if axis==2:
                new_arr[0:shape[0],0:shape[1],0:4]=self.data[i][-1::-1,0:shape[1],0:4]
            self.data[i]=new_arr
        self.recenter()

    def overlay_frame(self,image,pos,frame,center=1):
        if center==1:
            centerpos=myim.force_in_image(image,(pos[0]-int(self.center[frame][0]),pos[1]-int(self.center[frame][1])))
            image=myim.add_images(self.data[frame],image,centerpos[0],centerpos[1])
        else:
            image=myim.add_images(self.data[frame],image,pos[0],pos[1])
        return image

    def overlay(self,background,path,frames=0):
        pace_count=-1
        s_index=0
        if frames==0:
            frames=len(background)
        else:
            background=[background]*frames
        if type(path) is tuple:
            path=sprite_path([(path[0],path[1])]*frames)
        for i in range(frames):
            pace_count+=1
            position=path.path[i]
            if pace_count==self.pace:
                pace_count=0
                s_index+=1
            if s_index==len(self.sequence):
                s_index=0
            img_new=self.data[self.sequence[s_index]]
            img0=img_new.copy()
            if path.angle1[i] != 0:
                img_new=myim.rotate_image(img_new,path.angle1[i])
            if path.angle2[i] > 0:
                img_new=myim.rotate3d(img_new,path.angle2[i],0.0)
            if path.angle3[i] > 0:
                img_new=myim.rotate3d(img_new,0.0,path.angle3[i])
            if path.flipx[i] > 0:
                img_new=cv2.flip(img_new,0)
            if path.flipy[i] > 0:
                img_new=cv2.flip(img_new,1)
            if path.opacity[i] < 1.0:
                img_new=myim.apply_opacity(img_new,path.opacity[i])
            center=self.center[self.sequence[s_index]]
            # Correct for rotation offset in center
            imcenter=(int(img_new.shape[1]/2),int(img_new.shape[0]/2))
            imcenter0=(int(img0.shape[1]/2),int(img0.shape[0]/2))
            center=(center[0]+imcenter[0]-imcenter0[0],center[1]+imcenter[1]-imcenter0[1])
            if len(self.seqflips)>0: 
                if self.seqflips[s_index]!=0:
                    if self.seqflips[s_index]>0:
                        flipCode=self.seqflips[s_index]%2
                    else:
                        flipCode=-1
                    img_new=cv2.flip(img_new,flipCode=flipCode)
            if len(self.seqoffx)>0: 
                offset_x=self.seqoffx[s_index]
            else:
                offset_x=0
            if len(self.seqoffy)>0: 
                offset_y=self.seqoffy[s_index]
            else:
                offset_y=0
            if path.sizes[i] != (1.0,1.0) and path.sizes[i] != (-2,-2):
                img_new=cv2.resize(img_new,(0,0),fx=path.sizes[i][0],fy=path.sizes[i][1],interpolation=cv2.INTER_NEAREST)
                center=(center[0]*path.sizes[i][0],center[1]*path.sizes[i][1])
                offset_x=int(offset_x*path.sizes[i][0])
                offset_y=int(offset_y*path.sizes[i][1])
            true_pos=(position[0]-int(center[0])+offset_x,position[1]-int(center[1])+offset_y)
            if path.path[i][0]>=0:
                background[i]=myim.add_images(img_new,background[i],true_pos[0],true_pos[1])
        return background

    def nframes(self):
        return len(self.sequence)*self.pace

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

def add_sprite_vector(positions,game,object,frame):
    positions=np.swapaxes(positions,0,1)
    spath=sprite_fullpath(game,object,frame,vector=1)
    dirname,filename=os.path.split(spath)
    os.makedirs(dirname,exist_ok=True)
    np.savetxt(spath,positions,fmt='%d')

def sprite_fullpath(game,object,frame,vector=0):
    root_dir='C:/Users/sp4ce/Google Drive/Documents/Sprites/'
    if frame == "all":
        full_path=root_dir+game+"/"+object
    else:
        path=root_dir+game+"/"+object+"/"+frame
        if vector==1:
            full_path=path+'.vdat'
        else:
            thisfile=glob.glob(path+'*')
            if len(thisfile)>0:
                full_path=thisfile[0]
            else:
                full_path=path+'.png'

    return full_path

def add_sprite(images,game,object,frame="all",size=1.0,rotate=0.0,pace=1,path=[0],sequence='None',anchor=0,center=0,flip=0,color='Default'):
    if (images[0].shape[2]==3):
        images=myim.add_alpha_channel(images)
    mysprite=Sprite(game,object,frame,pace=pace,size=size,anchor=anchor)
    if rotate>0:
        mysprite.rotate_data(rotate)
    if flip>0:
        mysprite.flip_data(flip)
    if sequence != 'None':
        mysprite.read_sequence(sequence)
    if color != 'Default':
        mysprite.swap_colors(color)
    if path==[0]:
        path=myim.capture_point(images[0])
    return mysprite.overlay(images,path)

def add_sprite_blank(game,object,frame="all",size=1.0,rotate=0.0,pace=1,path=[0],sequence='None',anchor=0,center=0,flip=0,text='',bgcolor='48 40 36',bright=1.0,color='Default'):
    mysprite=Sprite(game,object,frame,pace=pace,anchor=anchor)
    if rotate>0:
        mysprite.rotate_data(rotate)
    if flip>0:
        mysprite.flip_data(flip)
    if sequence != 'None':
        mysprite.read_sequence(sequence)
    if color != 'Default':
        mysprite.swap_colors(color)
    if size!=1.0:
        mysprite.resize(size)
    size,size_x,size_y=mysprite.maxsize()
    bgcolor=mycolor.parse_color(bgcolor)
    blank_size=int(size*1.5)
    blank_size_x=int(size_x*1.5)
    blank_size_y=int(size_y*1.5)
    new_img=np.zeros([blank_size_y,blank_size_x,4],'uint8')
    new_img[:,:,0:3]=bgcolor[0:3]
    new_img=myim.add_texture(new_img,bright=bright)
    #new_img=myim.add_border(new_img,width=3)
    images=[new_img]*mysprite.nframes()
    if text != '':
        fontsize=blank_size/200
        text_pos_x=int(blank_size/2)-int(len(text)*8*fontsize)
        text_pos_y=int(blank_size/6)
        for i in range(len(images)):
            cv2.putText(images[i],text,(text_pos_x,text_pos_y),cv2.FONT_HERSHEY_SIMPLEX,fontsize,(255,255,255),1,cv2.LINE_AA)
    if path==[0] and center==0:
        path=myim.capture_point(images[0])
    if path==[0] and center==1:
        if len(mysprite.seqoffy)>0:
            posy=int(blank_size_y/2)-int(np.mean(mysprite.seqoffy))
        else:
            posy=int(blank_size_y/2)
        if len(mysprite.seqoffx)>0:
            posx=int(blank_size_x/2)-int(np.mean(mysprite.seqoffx))
        else:
            posx=int(blank_size_x/2)
        path=(posx,posy)
    return mysprite.overlay(images,path)

def vector2image(x,y,color=[255,255,255,255],width=1,scale=1):
    xsize=int((np.max(x)-np.min(x))*scale)+int(width/2+1)
    ysize=int((np.max(y)-np.min(y))*scale)+int(width/2+1)
    xmin=np.min(x)
    ymin=np.min(y)
    image=np.zeros((ysize,xsize,4),dtype='uint8')
    i=0
    for xp in x:
        i+=1
        if i==len(x):
            break
        image=cv2.line(image,(int((x[i-1]-xmin)*scale),int((y[i-1]-ymin)*scale)),(int((x[i]-xmin)*scale),int((y[i]-ymin)*scale)),color,width)
    return image
