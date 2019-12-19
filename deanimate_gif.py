import subprocess
import os
import math
import sys
import cv2

infile=sys.argv[1]
if not os.path.exists(infile):
    print("No such file: "+infile)
    sys.exit()

if len(sys.argv)>2:
    outdir=sys.argv[2]
else:
    outdir=os.path.splitext(infile)
    outdir=outdir[0]

try:
    os.mkdir(outdir)
except OSError as e:
    print("Error in making directory "+outdir+": "+e.strerror)

cmd="magick convert -coalesce "+infile+" "+outdir+"/%04d.png"
print(cmd)
returned_value=os.system(cmd)