
import math
import cv2
import numpy as np
import os
import sys
import myimutils
from scipy import ndimage

class Sprite:
    def __init__(self,game,object,frame):
        root_dir="../Sprites/"
        self.full_path=root_dir+game+"_"+object+"/"+frame+".png"
        if os.path.isfile(self.full_path)==False:
            print("Sprite not found.")
            return
        self.data=cv2.imread(self.full_path,cv2.IMREAD_UNCHANGED)
        self.visible=np.nonzero(self.data[:,:,3])
        self.center=[np.mean(self.visible[0]),np.mean(self.visible[1])]

    def overlay(self,background,position):
        return myimutils.add_images(self.data,background,position[0]-int(self.center[0]),position[1]-int(self.center[1]))