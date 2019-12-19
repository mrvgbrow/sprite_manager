
import math
import cv2
import numpy as np
import os
import sys
import time
import shutil

def add_images(image2,image1,x,y):
    x1,x2,y1,y2=get_overlap(image1,image2,x,y)
    alpha_2=image2[:y2-y1,:x2-x1,3]/255.0
    alpha_1=1.0-alpha_2
    for c in range(0,3):
        image1[y1:y2,x1:x2,c]=(alpha_2*image2[:y2-y1,:x2-x1,c]+alpha_1*image1[y1:y2,x1:x2,c])
    return image1

def add_alpha_channel(image):
    image=np.dstack((image,np.ones((image.shape[0],image.shape[1]),'byte')*255))
    return image

def get_overlap(image1,image2,x,y):
    xlim=image1.shape[1]
    ylim=image1.shape[0]
    y1,y2=y,y+image2.shape[0]
    x1,x2=x,x+image2.shape[1]
    if xlim<x2:
        x2=xlim-1
    if ylim<y2:
        y2=ylim-1
    return [x1,x2,y1,y2]

def read_imdir(dir):
    name,extension=os.path.splitext(dir)
    if extension=='.gif':
        make_outdir('temp',1)
        deanimate_gif(dir,'temp')
        dir='temp'
    if extension=='.png':
        return [cv2.imread(dir,cv2.IMREAD_UNCHANGED)]
    elif os.path.isdir(dir):
        files=os.listdir(dir)
        imlist=[]
        
        i=0
        for thisim in files:
            image=cv2.imread(dir+"/"+thisim,cv2.IMREAD_UNCHANGED)
            imlist.append(image)
            i+=1
        imlist=np.stack((imlist[:]))
        return imlist

def click_point(event,x,y,flags,param):
    global refPt,clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt=(x,y)
        clicked=1

def clicksave(event,x,y,flags,param):
    global clicked,pointlist
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked=1
    if event == cv2.EVENT_MOUSEMOVE and clicked==1:
        if len(pointlist)>0:
            previous_point=pointlist[len(pointlist)-1]
        else:
            previous_point=(0,0)
        if previous_point[0]!=x or previous_point[1]!=y:
            pointlist.append((x,y))
    if event == cv2.EVENT_LBUTTONUP:
        clicked=0

def click_rectangle(event,x,y,flags,param):
    global boxcorners,clicked,cornertemp
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked=1
        boxcorners=[(x,y)]
        cornertemp=[]
    if event == cv2.EVENT_MOUSEMOVE and clicked==1:
        cornertemp=[(x,y)]
    if event == cv2.EVENT_LBUTTONUP:
        clicked=0
        boxcorners.append((x,y))

def capture_point(image):
    global clicked,refPt

    clicked=0
    refPt=(0,0)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",click_point)

    while True:
        cv2.imshow("image",image[:,:,:3])
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break
        if clicked==1:
            break

    cv2.destroyAllWindows()
    return refPt

def capture_path(image):
    global clicked,pointlist
    clicked=0
    pointlist=[]

    image = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
    h1,w1=image.shape[:2] 
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",clicksave)

    while True:
        cv2.imshow("image",image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    return pointlist

def capture_box(image):
    global clicked,boxcorners,cornertemp
    clicked=0
    boxcorners=[]
    cornertemp=[]

    cv2.namedWindow("image")
    cv2.setMouseCallback("image",click_rectangle)

    while True:     
        clone=image.copy()
        if len(cornertemp)>0:
            cv2.rectangle(clone, boxcorners[0], cornertemp[0], (0, 255, 0), 2)
        cv2.imshow("image",clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    return boxcorners

def make_outdir(dir,force):
    if force==1 and os.path.isdir(dir):
        shutil.rmtree(dir)
    try:
        os.mkdir(dir)
    except OSError as e:
        print("Error in making directory "+dir+": "+e.strerror)

def sprite_center(mask):
    indices=np.asarray(np.nonzero(mask))
    return indices.mean(axis=1)

def maketransparent_withmask(image,mask):
    b_channel, g_channel, r_channel=cv2.split(image)
    image=cv2.merge((b_channel,g_channel,r_channel,mask.astype(b_channel.dtype)*255))
    return image

def do_grabcut_rect(image,corner1,corner2,iter):
    mask=np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect=(corner1[0],corner1[1],corner2[0]-corner1[0],corner2[1]-corner1[1])
    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,iter,cv2.GC_INIT_WITH_RECT)
    mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = image*mask2[:,:,np.newaxis]

    return [image,mask2]

def imshow_loop(image,title,exit_char):
    cv2.namedWindow(title)
    while True:
        cv2.imshow(title,image)
        key = cv2.waitKey(1) & 0xFF
        if key==ord(exit_char):
            break
    cv2.destroyAllWindows()

def imfile_name(index,indir):
    numstr="{0:04d}".format(index)
    filename=numstr+".png"
    if indir != "":
        return indir+"/"+filename
    else:
        return filename

def animate_dir(dirname,delay,outfile):
    cmd="magick convert -dispose previous -set delay "+str(delay)+" "+dirname+"/*.png "+outfile
    returned_value=os.system(cmd)

def deanimate_gif(infile,outdir):
    cmd="magick convert -coalesce "+infile+" "+outdir+"/%04d.png"
    returned_value=os.system(cmd)

def make_mask(indices,shape):
    mask=np.zeros((shape[0],shape[1]),np.bool)
    mask[indices]=1
    return mask

def apply_mask(image,mask):
    image_new = image*mask[:,:,np.newaxis]
    return image_new

def mask_replace(image,mask,color):
    mask_indices=np.nonzero(mask)
    image[mask_indices[0],mask_indices[1],0] = color[0]
    image[mask_indices[0],mask_indices[1],1] = color[1]
    image[mask_indices[0],mask_indices[1],2] = color[2]
    return image