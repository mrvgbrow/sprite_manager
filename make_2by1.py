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
image1=imlist1[0]
outfile=myargutils.check_arg(sys.argv,2,'temp.png')

outdir="temp"
myimutils.make_outdir(outdir,1)

y_size=max(image1.shape[0],image1.shape[1]/2)
x_size=2*y_size
if image1.shape[0]==y_size:
    dimentopad=1
    pad=x_size-image1.shape[1]
else:
    dimentopad=0
    pad=y_size-image1.shape[0]

newim=np.zeros((y_size,x_size,image1.shape[2]),image1.dtype)
if dimentopad==0:
    newim[int(pad/2):int(pad/2)+image1.shape[0],:image1.shape[1],:image1.shape[2]]=image1
else:
    newim[:image1.shape[0],int(pad/2):int(pad/2)+image1.shape[1],:image1.shape[2]]=image1
cv2.imwrite(outfile,newim)
sys.exit()

for i in range(len(imlist1)):
    image1=imlist1[i]
    newim=np.zeros((y_size,x_size,image1.shape[2]),image1.dtype)

    if dimentopad==0:
        newim[int(pad/2):int(pad/2)+image1.shape[0],:image1.shape[1],:image1.shape[2]]=image1
    else:
        newim[:image1.shape[0],int(pad/2):int(pad/2)+image1.shape[1],:image1.shape[2]]=image1

    cv2.imwrite(myimutils.imfile_name(i,outdir),newim)

myimutils.animate_dir(outdir,1,outfile)
cv2.destroyAllWindows()