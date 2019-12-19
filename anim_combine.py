import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import myargutils
import time

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

imlist1 = myimutils.read_imdir(sys.argv[1])
imlist2 = myimutils.read_imdir(sys.argv[2])
image1=imlist1[0]
image2=imlist2[0]
side=int(myargutils.check_arg(sys.argv,3,1))
outfile=myargutils.check_arg(sys.argv,4,'temp.gif')
separator_size=myargutils.check_arg(sys.argv,5,6)

outdir="temp"
myimutils.make_outdir(outdir)
nframes=min(len(imlist1),len(imlist2))

xsize=max(image1.shape[1],image2.shape[1])
ysize=max(image1.shape[0],image2.shape[0])
if side==1:
    xsize=image1.shape[1]+image2.shape[1]+separator_size
if side==2:
    ysize=image1.shape[0]+image2.shape[0]+separator_size

for i in range(nframes):
    image1=imlist1[i]
    image2=imlist2[i]
    newim=np.zeros((ysize,xsize,image1.shape[2]),image1.dtype)

    newim[:image1.shape[0],:image1.shape[1],:image1.shape[2]]=image1
    if side==2:
        newim[image1.shape[0]+separator_size:ysize,:image2.shape[1],:image1.shape[2]]=image2
        newim[image1.shape[0]+1:image1.shape[0]+separator_size,:,0]=255
    if side==1:
        newim[image2.shape[0],image1.shape[1]+separator_size:xsize,:image1.shape[2]]=image2
        newim[:,image1.shape[1]+1:image1.shape[1]+separator_size,0]=255
    cv2.imwrite(myimutils.imfile_name(i,outdir),newim)

myimutils.animate_dir(outdir,1,outfile)
cv2.destroyAllWindows()
sys.exit()