#!/C/uSERs/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import time
from PIL import ImageSequence, Image, ImageOps

ap = argparse.ArgumentParser()
ap.add_argument("infile",help="Name of the input animation",type=str)
ap.add_argument("-o","--outfile",required=False,help="Name of the output animation file",default='temp.gif')
args=vars(ap.parse_args())

im=Image.open(args['infile'])

outfile=args['outfile']
dimens=im.size

square_size=max(dimens)
if dimens[0]<square_size:
    padding=(int((square_size-dimens[0])/2),0,int((square_size-dimens[0])/2),0)
else:
    padding=(0,int((square_size-dimens[1])/2),0,int((square_size-dimens[1])/2))

frames=[]
durations=[]
print(padding,square_size)
for frame in range(0,im.n_frames):
    im.seek(frame)
    new_im=im.convert('RGBA')
    durations.append(new_im.info['duration'])
    new_im2=ImageOps.expand(new_im,padding)
    frames.append(new_im2)

frames[0].save(outfile,save_all=True,append_images=frames[1:],duration=durations,loop=0)
