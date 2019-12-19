import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import myargutils
import mycolortools

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

imlist1 = myimutils.read_imdir(sys.argv[1])
image1=imlist1[0]
fuzz=int(myargutils.check_arg(sys.argv,2,0))
outfile=myargutils.check_arg(sys.argv,3,'temp.gif')


color_in=mycolortools.imshow_get_color(image1,'Select Color to Replace','x')
color_out=mycolortools.imshow_get_color(image1,'Select Color to Replace With','x')
indices=mycolortools.select_color(image1,color_in,fuzz)
mask=myimutils.make_mask(indices,image1.shape)
image_masked=myimutils.mask_replace(image1,mask,color_out)
myimutils.imshow_loop(image_masked,'Masked Image','x')