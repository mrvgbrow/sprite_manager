import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import time
import math

def get_mask_box(mask,center,sizex,sizey):
    ylim1=int(round(center[0]))-int(sizey/2)
    ylim2=int(round(center[0]))+int(sizey/2)
    xlim1=int(round(center[1]))-int(sizex/2)
    xlim2=int(round(center[1]))+int(sizex/2)
    return mask[ylim1:ylim2,xlim1:xlim2]

def get_mask_diff(mask1,mask2):
    box_area=float(mask1.shape[0]*mask1.shape[1])
    diff=np.sum(np.absolute(mask1.astype('float')-mask2.astype('float')))/box_area
    return diff

def iter_grabcut(im,corner1,corner2,iter,sizex,sizey,old_mask):
    [image,mask]=myimutils.do_grabcut_rect(im,corner1,corner2,iter)
    newcenter=myimutils.sprite_center(mask)      
    new_mask=get_mask_box(mask,newcenter,sizex,sizey) 
    diff=get_mask_diff(old_mask,new_mask)   
    return [image,mask,new_mask,diff,newcenter]

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

iter=5
threshold=0.07
imlist1 = myimutils.read_imdir(sys.argv[1])
image=imlist1[0]

if len(imlist1)>1:
    myimutils.make_outdir(sys.argv[2],1)

while True:
    box=myimutils.capture_box(image)
    [image,mask]=myimutils.do_grabcut_rect(image,box[0],box[1],iter)
    image_trans=myimutils.maketransparent_withmask(image,mask)

    cv2.namedWindow("image2")
    cv2.imshow("image2",image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("a") or key == ord("r"):
            break
    if key == ord("a"):
        break

i=0
corner1=box[0]
corner2=box[1]
sizex=corner2[0]-corner1[0]
sizey=corner2[1]-corner1[1]
center=myimutils.sprite_center(mask) 
old_mask=get_mask_box(mask,center,sizex,sizey)
for im in imlist1:
    numstr="{0:04d}".format(i)
    [image,mask,new_mask,diff,newcenter]=iter_grabcut(im,corner1,corner2,iter,sizex,sizey,old_mask)
    if diff>threshold:        
        [image,mask,new_mask,diff,newcenter]=iter_grabcut(im,corner1+(-1,0),corner2+(1,0),iter,sizex,sizey,old_mask)
    if diff>threshold:        
        [image,mask,new_mask,diff,newcenter]=iter_grabcut(im,corner1+(0,-1),corner2+(0,1),iter,sizex,sizey,old_mask)
    if diff>threshold:        
        [image,mask,new_mask,diff,newcenter]=iter_grabcut(im,corner1+(-1,-1),corner2+(-1,-1),iter,sizex,sizey,old_mask)
    if diff>threshold:        
        [image,mask,new_mask,diff,newcenter]=iter_grabcut(im,corner1+(1,1),corner2+(1,1),iter,sizex,sizey,old_mask)

    print(str(diff),numstr)
    corner1=(int(round(newcenter[1]-sizex/2)),int(round(newcenter[0]-sizey/2)))
    corner2=(int(round(newcenter[1]+sizex/2)),int(round(newcenter[0]+sizey/2)))
    if corner2[0]>image.shape[1]-1 or corner2[1]>image.shape[0]-1:
        break
    if corner1[0]<0 or corner1[1]<0:
        break
    image_trans=myimutils.maketransparent_withmask(image,mask)
    cv2.rectangle(image, corner1, corner2, (0, 255, 0), 2)
    cv2.imwrite(sys.argv[2]+"/"+numstr+".png",image)
    i+=1
    if diff<threshold:
        old_mask=new_mask

cv2.destroyAllWindows()
sys.exit()