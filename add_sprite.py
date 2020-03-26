#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import os
import sys
import cv2
import numpy as np
import mycolortools as mycolor
import argparse
import myimutils
import myargutils
import myspritetools as myspr
import time
from PIL import ImageSequence, Image, ImageOps

ap = argparse.ArgumentParser()
ap.add_argument("infile",help="Name of the input background file",type=str)
ap.add_argument("-o","--outfile",required=False,help="Name of the output file",default='test.gif')
ap.add_argument("-p","--pace",required=False,help="Number of frames of the final animation per sprite frame",default=1,type=int)
ap.add_argument("-d","--duration",required=False,help="Duration of each frame",default=1,type=int)
ap.add_argument("game",help="Name of the game the sprite is from",type=str)
ap.add_argument("object",help="Name of the object the sprite represents",type=str)
ap.add_argument("-f","--frame",required=False,help="The particular frame in the sprite sequence",type=str,default='all')
ap.add_argument("-c","--color",required=False,help="Color scheme for the sprite",type=str,default='Default')
ap.add_argument("-s","--size",required=False,help="The sprite size scale factor",type=float,default=1.0)
ap.add_argument("-q","--sequence",required=False,help="Name of the animation sequence to use",type=str,default='None')
ap.add_argument("-a","--anchor",required=False,help="Where to anchor the sprite sequence (0: center, 1: bottom, 3: top)",type=int,default=0)
ap.add_argument("-r","--rotate",required=False,help="The sprite rotation rate",type=float,default=0.0)
ap.add_argument("-l","--flip",required=False,help="Flip axis (0:None,1:horizontal,2:vertical)",type=int,default=0)
ap.add_argument("-b","--bucketexclude",required=False,help="Threshold for bucket select exclusion",type=int,default=0)
ap.add_argument("-i","--pil",required=False,help="Use PIL?",type=int,default=0)
ap.add_argument("-t","--text",required=False,help="Text string to label sprite with",type=str,default='')
ap.add_argument("-g","--bright",required=False,help="Brightness factor of padding.",type=float,default=1.0)
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
flip=args['flip']
text=args['text']
pil=args['pil']

sys.setrecursionlimit(1000000)
if args['infile'] != 'blank':
    (background,durations)=myimutils.read_imdir(args['infile'])
    if args['bucketexclude']>0:
        background0=background[0][:,:,0:3].copy()
        inds=mycolor.bucket_select(background[0],threshold=args['bucketexclude'])
    path=myspr.sprite_path([0])
    if len(background)==1:
        tempsprite=myspr.Sprite(game,object,frame)
        tempsprite.read_sequence(sequence)
        background=[background[0]]*tempsprite.nframes()
        durations=[10*args['duration']]*len(background)
        pil=0
    path.input_path(background)

if args['infile']=='blank':
    center=1
    if game != 'blank':
        new_frames=myspr.add_sprite_blank(game,object,size=size,pace=pace,rotate=rotate,frame=frame,sequence=sequence,anchor=anchor,center=center,flip=flip,text=text,bright=args['bright'],color=args['color'])
    else:
        new_frames=[myimutils.add_texture(np.zeros([int(size*30),int(size*30),4],'uint8'))]*100
    durations=[10*args['duration']]*len(new_frames)
else:
    new_frames=myspr.add_sprite(background,game,object,size=size,pace=pace,rotate=rotate,frame=frame,sequence=sequence,anchor=anchor,center=0,flip=flip,path=path,color=args['color'])
    if args['bucketexclude']>0:
        for i in range(len(new_frames)):
            new_frames[i]=new_frames[i][:,:,0:3]
            new_frames[i][inds]=background0[inds]
dum=myimutils.gif_viewer(new_frames,durations,'Result')
new_frames=myimutils.convert_to_PIL(new_frames)
myimutils.write_animation(new_frames,durations,outfile,pil=pil)
