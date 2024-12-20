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

def sample_line(pos1,pos2,nsample):
    inc_x=(pos2[0]-pos1[0])/(nsample-1)
    inc_y=(pos2[1]-pos1[1])/(nsample-1)
    pos_all=[pos1]
    for i in range(nsample-2):
        pos_all.append((int(pos1[0]+(i+1)*inc_x),int(pos1[1]+(i+1)*inc_y)))
    pos_all.append(pos2)
    return pos_all

def sample_quadratic(pos1,pos2,nsample,sign=1):
    samples=np.linspace(0,1,num=nsample)
    if sign==1:
        x=np.multiply(samples,samples*((pos2[0]-pos1[0])))+pos1[0]
        y=np.multiply(samples,samples*((pos2[1]-pos1[1])))+pos1[1]
    else:
        x=pos2[0]-np.multiply(samples,samples*((pos2[0]-pos1[0])))
        y=pos2[1]-np.multiply(samples,samples*((pos2[1]-pos1[1])))
        x=np.flip(x)
        y=np.flip(y)
    return x,y

def rotate_points(origin,points,angle):

    ox,oy=origin
    points_new=[]
    for i in range(len(points)):
        qx=int(ox+math.cos(angle)*(points[i][0]-ox)-math.sin(angle)*(points[i][1]-oy))
        qy=int(oy+math.sin(angle)*(points[i][0]-ox)+math.cos(angle)*(points[i][1]-oy))
        points_new.append((qx,qy))
    return points_new
