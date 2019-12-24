#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python37/python

import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import myargutils
import mycolortools
from scipy import stats

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

imlist1,durations = myimutils.read_gif(sys.argv[1])
outfile=myargutils.check_arg(sys.argv,2,'temp.png')
imlist_combined=[]
for im in imlist1:
    imlist_combined.append(mycolortools.color_combine(im))
imagemode=stats.mode(imlist_combined)
image=np.squeeze(imagemode[0],axis=0)
image_mode=mycolortools.color_expand(image)
myimutils.imshow_loop(image_mode,'Background Image','x')
cv2.imwrite(outfile,image)
