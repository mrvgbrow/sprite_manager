#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

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

ap = argparse.ArgumentParser()
ap.add_argument("infile",help="Name of the input background file",type=str)
ap.add_argument("-o","--outfile",required=False,help="Name of the output file",default='test.gif')
ap.add_argument("-p","--pace",required=False,help="Speed at which the sprite sequence moves",default=1,type=int)
ap.add_argument("game",help="Name of the game the sprite is from",type=str)
ap.add_argument("object",help="Name of the object the sprite represents",type=str)
ap.add_argument("-f","--frame",required=False,help="The particular frame in the sprite sequence",type=str,default='all')
ap.add_argument("-c","--center",required=False,help="Center the sprite in the background image",type=int,default=0)
ap.add_argument("-s","--size",required=False,help="The sprite size scale factor",type=float,default=1.0)
ap.add_argument("-q","--sequence",required=False,help="Name of the animation sequence to use",type=str,default='None')
ap.add_argument("-a","--anchor",required=False,help="Where to anchor the sprite sequence (0: center, 1: bottom, 3: top)",type=int,default=0)
ap.add_argument("-r","--rotate",required=False,help="The sprite rotation rate",type=float,default=0.0)
ap.add_argument("-l","--flip",required=False,help="Flip axis (0:None,1:horizontal,2:vertical)",type=int,default=0)
ap.add_argument("-t","--text",required=False,help="Text string to label sprite with",type=str,default='')
args=vars(ap.parse_args())

game=args['game']
object=args['object']
frame=args['frame']
pace=args['pace']
outfile=args['outfile']
size=args['size']
rotate=args['rotate']
sequence=args['sequence']
anchor=args['anchor']
center=args['center']
flip=args['flip']
text=args['text']

if args['infile'] != 'blank':
    (background,durations)=myimutils.read_imdir(args['infile'])
    path=myspritetools.sprite_path([0])
    path.input_path(background)
    if len(background)==1:
        background=[background[0]]*100

if args['infile']=='blank':
    new_frames=myspritetools.add_sprite_blank(game,object,size=size,pace=pace,rotate=rotate,frame=frame,sequence=sequence,anchor=anchor,center=center,flip=flip,text=text)
    durations=[10]*len(new_frames)
else:
    new_frames=myspritetools.add_sprite(background,game,object,size=size,pace=pace,rotate=rotate,frame=frame,sequence=sequence,anchor=anchor,center=center,flip=flip,path=path)
dum=myimutils.gif_viewer(new_frames,durations,'Result')
new_frames=myimutils.convert_to_PIL(new_frames)
myimutils.write_animation(new_frames,durations,outfile)
