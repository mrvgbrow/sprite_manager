import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import time

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

imlist1 = myimutils.read_imdir(sys.argv[1])
image=imlist1[0]

cv2.namedWindow("Image Browser")

current=0
while True:
    image=imlist1[current]
    cv2.imshow("Image Browser",image)
    key = cv2.waitKey(1) & 0xFF
    if key==ord("x"):
        break
    if key==ord("n"):
        current+=1
    if key==ord("p"):
        current-=1
    if key==ord("s"):
        current+=10
    if key==ord("r"):
        current-=10
    if current>len(imlist1):
        current=current%len(imlist1)
    if current<0:
        current=len(imlist1)+current

cv2.destroyAllWindows()
sys.exit()