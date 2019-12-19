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
outfile=myargutils.check_arg(sys.argv,2,'temp.png')
image0=stats.mode(imlist1)
image=np.squeeze(image0[0],axis=0)
myimutils.imshow_loop(image,'Background Image','x')
cv2.imwrite(outfile,image)