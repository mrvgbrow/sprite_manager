import subprocess
import os
import math
import sys
import cv2
import imutils
import numpy as np

img_in=sys.argv[1]
img_ext=".png"
filename_prefix=os.path.splitext(img_in)
dir_out="rot_frames_"+filename_prefix[0]

try:
    os.mkdir(dir_out)
except OSError as e:
    print("Error in making directory "+dir_out+": "+e.strerror)

if len(sys.argv) > 2:
    interval=int(sys.argv[2])

img=cv2.imread(img_in,cv2.IMREAD_UNCHANGED)

i=0
for angle in np.arange(0,360,interval):
    numstr="{0:03d}".format(i)
    img_rotate = imutils.rotate(img,angle)
    cv2.imwrite(dir_out+"/"+numstr+'.png', img_rotate)
    i+=1

delay=1
outfile=filename_prefix[0]+"_rotated.gif"
cmd="magick convert -dispose previous -set delay "+str(delay)+" "+dir_out+"/*.png "+outfile
returned_value=os.system(cmd)
returned_value=os.system(outfile)