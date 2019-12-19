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
outfile=myargutils.check_arg(sys.argv,2,'temp.gif')

outdir="temp"
myimutils.make_outdir(outdir,1)

yim=image1.shape[0]
xim=image1.shape[1]
if xim>yim and xim>16*yim/9:
    x_size=xim
    y_size=int(xim*16/9)
elif xim>yim and xim<16*yim/9:
    y_size=yim
    x_size=int(yim*9/16)
elif yim>=xim and yim>16*xim/9:
    y_size=yim
    x_size=int(yim*9/16)
else:
    y_size=int(xim*16/9)
    x_size=xim

xpad=x_size-xim
ypad=y_size-yim

for i in range(len(imlist1)):
    image1=imlist1[i]
    newim=np.zeros((y_size,x_size,image1.shape[2]),image1.dtype)
    newim[int(ypad/2):int(ypad/2)+image1.shape[0],int(xpad/2):int(xpad/2)+image1.shape[1],:image1.shape[2]]=image1
    cv2.imwrite(myimutils.imfile_name(i,outdir),newim)

myimutils.animate_dir(outdir,1,outfile)
cv2.destroyAllWindows()
sys.exit()