import subprocess
import os
import math
import sys
import cv2
import numpy as np
import argparse
import myimutils

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

imlist1 = myimutils.read_imdir(sys.argv[1])
imlist2 = myimutils.read_imdir(sys.argv[2])

path=myimutils.capture_path(imlist1[0])
print(path)

outdir=sys.argv[3]
myimutils.make_outdir(outdir)

foreground_count=0
path_count=0
i=0
for im_background in imlist1:
    image2=imlist2[foreground_count]
    numstr="{0:03d}".format(i)
    point=path[path_count]
    clone=im_background.copy()
    print(point)
    myimutils.add_images(clone,image2,point[0],point[1])
    cv2.imwrite(outdir+"/"+numstr+".png",clone)
    if foreground_count==len(imlist2):
        foreground_count=0
    else:
        foreground_count+=1
    path_count+=1
    i+=1

sys.exit()

delay=1
outfile=filename_prefix[0]+"_rotated.gif"
cmd="magick convert -dispose previous -set delay "+str(delay)+" "+dir_out+"/*.png "+outfile
returned_value=os.system(cmd)
returned_value=os.system(outfile)