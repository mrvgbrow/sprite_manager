#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import myargutils
import mycolortools
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("infile",help="Name of the input animation",type=str)
ap.add_argument("-o","--outfile",required=False,help="Name of the output background file",default='test.png')
ap.add_argument("-g","--outgif",required=False,help="Name of the background-subtracted animation file",default='test.gif')
ap.add_argument("-f","--fuzz",required=False,help="Color proximity to black that should be made transparent",type=int,default=5)
args=vars(ap.parse_args())

imlist1,durations = myimutils.read_gif(args['infile'])
i=0
for im in imlist1:
    imlist1[i]=mycolortools.color_combine(im)
    i+=1
imagemode=stats.mode(imlist1)
image=np.squeeze(imagemode[0],axis=0)
image_mode=mycolortools.color_expand(image)
myimutils.imshow_loop(image_mode,'Background Image','x')

i=0
for im in imlist1:
    imlist1[i]=mycolortools.color_expand(imlist1[i])
    imlist1[i]=np.subtract(imlist1[i],image_mode)
    indices=mycolortools.select_color(imlist1[i],[0,0,0],args['fuzz'],invert=True)
    mask=myimutils.make_mask(indices,imlist1[i].shape)
    imlist1[i]=myimutils.maketransparent_withmask(imlist1[i],mask)
    i+=1
cv2.imwrite(args['outfile'],image)
myimutils.write_animation(imlist1,durations,args['outgif'])
