import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import myargutils
import math

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

imlist1 = myimutils.read_imdir(sys.argv[1])
final_speed=float(myargutils.check_arg(sys.argv,2,3))
outfile=myargutils.check_arg(sys.argv,3,'temp.gif')
myimutils.make_outdir('temp_out',1)
scalefac=math.sqrt(float(imlist1.shape[0]))/final_speed

i=0
j=0
nskip=0
for frame in imlist1:
    if nskip>1:
        nskip-=1
        i+=1
        continue
    speed=math.sqrt(float(i)+0.5)/scalefac
    if speed<=1.0:
        ncopies=int(round(1/speed))
        for copy in range(ncopies):
            cv2.imwrite(myimutils.imfile_name(j,"temp_out"),frame)
            j+=1
    else:
        cv2.imwrite(myimutils.imfile_name(j,"temp_out"),frame)
        j+=1
        nskip=int(round(speed))
    i+=1

myimutils.animate_dir("temp_out",1,outfile)