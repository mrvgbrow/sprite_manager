#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import math
import cv2
import numpy as np
import os
import sys

def check_arg(argv,argnum,default):
    if len(argv)>argnum:
        return argv[argnum]
    else:
        return default
