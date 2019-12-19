import subprocess
import os
import math
import sys
import myimutils
import cv2

indir=sys.argv[1]
if not os.path.isdir(indir):
    print("No such directory: "+indir)
    sys.exit()

if len(sys.argv)>2:
    delay=sys.argv[2]
else:
    delay=1

if len(sys.argv)>3:
    outfile=sys.argv[3]
else:
    outfile=indir+".gif"

#cv2.imwrite("testing0.png",imlist[0])
cmd="magick convert -dispose previous -set delay "+str(delay)+" "+indir+"/*.png "+outfile
print(cmd)
returned_value=os.system(cmd)
returned_value=os.system(outfile)