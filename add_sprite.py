#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python37/python


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
ap.add_argument("-s","--size",required=False,help="The sprite size scale factor",type=float,default=1.0)
ap.add_argument("-r","--rotate",required=False,help="The sprite rotation rate",type=float,default=0.0)
args=vars(ap.parse_args())

game=args['game']
object=args['object']
(background,durations)=myimutils.read_imdir(args['infile'])
frame=args['frame']
pace=args['pace']
outfile=args['outfile']
size=args['size']
rotate=args['rotate']

background_trans=myimutils.add_alpha_channel(background)
mysprite=myspritetools.Sprite(game,object,frame,pace=pace,size=size,rotate=rotate)
position=myimutils.capture_point(background[0])
new_frames=mysprite.overlay(background_trans,position)
myimutils.write_animation(new_frames,durations,outfile)
dum=myimutils.gif_viewer(new_frames,durations,'Result')
