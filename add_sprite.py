import os
import sys
import cv2
import numpy as np
import argparse
import myimutils
import myargutils
import myspritetools
import time
from PIL import ImageSequence, Image, ImageOps

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

background=cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)

background_trans=myimutils.add_alpha_channel(background)
mysprite=myspritetools.Sprite(sys.argv[2],sys.argv[3],sys.argv[4])
print(type(mysprite.center[0]))
position=myimutils.capture_point(background)
new_im=mysprite.overlay(background_trans,position)
cv2.imwrite('test.png',new_im)
#myspritetools.get_sprite(sys.argv[2],sys.argv[3],sys.argv[4])
sys.exit()
im=Image.open(sys.argv[1])
pix=np.array(im.convert('RGB'))
print(pix.shape)

outfile=myargutils.check_arg(sys.argv,2,'temp.gif')
dimens=im.size

square_size=max(dimens)
if dimens[0]<square_size:
    padding=(int((square_size-dimens[0])/2),0,int((square_size-dimens[0])/2),0)
else:
    padding=(0,int((square_size-dimens[1])/2),0,int((square_size-dimens[1])/2))

frames=[]
print(padding,square_size)
for frame in range(0,im.n_frames):
    im.seek(frame)
    new_im=im.convert('RGBA')
    new_im2=ImageOps.expand(new_im,padding)
    frames.append(new_im2)

frames[0].save(outfile,save_all=True,append_images=frames[1:],duration=10,loop=0)
