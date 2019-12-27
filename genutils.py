#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import sys
import math

def position_in_circle(radius,angle,center):
    circlepos=(math.cos(angle)*radius,math.sin(angle)*radius)
    abspos=(circlepos[0]+center[0],circlepos[1]+center[1])
    return abspos

def gaussian_function(norm_distance):
    return 1.0/math.sqrt(2*math.pi)*math.exp(-1.0/2.0*norm_distance**2)
