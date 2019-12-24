#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python37/python

import sys
import math

def position_in_circle(radius,angle,center):
    circlepos=(math.cos(angle)*radius,math.sin(angle)*radius)
    abspos=(circlepos[0]+center[0],circlepos[1]+center[1])
    return abspos
