import os
import sys
import cv2
import numpy as np
import argparse
import myanimutils
import time

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

imlist1 = myanimutils.read_animgif(sys.argv[1])

if len(sys.argv)>2:
    delay=sys.argv[2]
else:
    delay=10

cv2.namedWindow("image")
i=0
for im in imlist1:
    cv2.waitKey(delay)&0xFF
    cv2.imshow("image",im)
    i+=1

cv2.destroyAllWindows()
sys.exit()