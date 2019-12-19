
import math
import cv2
import numpy as np
import os
import sys
import imageio

def read_animgif(file):
    gif=imageio.mimread(file)
    nums=len(gif)
    print("{} frames read".format(nums))
    imgs = [cv2.cvtColor(img,cv2.COLOR_RGB2BGR) for img in gif]
    return imgs

