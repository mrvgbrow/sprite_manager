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

imlist1 = myimutils.read_imdir(sys.argv[1])
threshold=int(myargutils.check_arg(sys.argv,2,3))
outfile=myargutils.check_arg(sys.argv,3,'temp.gif')
myimutils.make_outdir('temp_out',1)
image0=stats.mode(imlist1)
image=np.squeeze(image0[0],axis=0)
myimutils.imshow_loop(image,'Background Image','x')

i=0
for im in imlist1:
    new_img=im
    diff_img=mycolortools.color_distance(im,image)
    near_indices=np.where(diff_img<threshold)
    new_img[near_indices]=0
    cv2.imwrite(myimutils.imfile_name(i,"temp_out"),new_img)
    i+=1

myimutils.animate_dir("temp_out",1,outfile)