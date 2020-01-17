#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import sys
import math
import numpy as np

def position_in_circle(radius,angle,center):
    circlepos=(math.cos(angle)*radius,math.sin(angle)*radius)
    abspos=(circlepos[0]+center[0],circlepos[1]+center[1])
    return abspos

def gaussian_function(norm_distance):
    return 1.0/math.sqrt(2*math.pi)*math.exp(-1.0/2.0*norm_distance**2)

def cart_to_polar(xv,yv,maxrad=1,minrad=0,axis=0,phase=0):
    theta_v=math.pi*2*xv/np.max(xv)
    r_v=yv/np.max(yv)*maxrad+minrad
    xnew_v=r_v*np.cos(theta_v+phase)
    ynew_v=r_v*np.sin(theta_v+phase)
    return((xnew_v,ynew_v))

