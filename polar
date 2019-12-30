#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import cv2
import myimutils as myim
import sys
import argparse
import numpy as np
import genutils as genu


ap = argparse.ArgumentParser()
ap.add_argument("infile",help="Name of the input animation",type=str)
ap.add_argument("-o","--outfile",required=False,help="Name of the output background file",default='test.gif')
args=vars(ap.parse_args())

imlist1,durations = myim.read_gif(args['infile'])

x=np.linspace(0,1,imlist1[0].shape[1])
y=np.linspace(0,1,imlist1[0].shape[0])
xv,yv=np.meshgrid(x,y)
cart_indices=np.where(imlist1[0]>-1)
yv=cart_indices[0].astype('float')
xv=cart_indices[1].astype('float')
xv,yv=genu.cart_to_polar(xv,yv,maxrad=100,minrad=50)
polar_indices=myim.grid_to_indices(xv,yv)
polar_indices=(polar_indices[0],polar_indices[1],cart_indices[2])

i=0
imlist2=[]
for i in range(len(imlist1)):
    new_image=np.zeros([np.max(polar_indices[0])+1,np.max(polar_indices[1])+1,3],'uint8')
    new_image[polar_indices]=imlist1[i][cart_indices]
    imlist2.append(new_image)

imlist1=None
myim.gif_viewer(imlist2,durations,'Result')
myim.write_animation(imlist2,durations,args['outfile'])
