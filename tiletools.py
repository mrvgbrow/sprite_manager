import math
import numpy as np
import sys
import os
import cv2

class tilemap:
    def __init__(self,data,tilesize=(8,8),tilesep=(0,0),ntile=(0,0),init_skip=(0,0),game=''):
        self.root_dir='C:/Users/sp4ce/Google Drive/Documents/Tiles'
        if type(data) is str:
            filename=self.root_dir+'/'+game+'/'+data+'.png'
            self.data0=cv2.imread(filename)
        else:
            self.data0=data
        self.tilesize=tilesize
        self.tilesep=tilesep
        self.init_skip=init_skip
        self.ntile=ntile
        self.extract_tiles()

    def extract_tiles(self):
        shape=self.data0.shape
        increment_x=self.tilesize[0]+self.tilesep[0]
        increment_y=self.tilesize[1]+self.tilesep[1]
        self.nx=int((shape[1]-self.init_skip[0])/increment_x)
        self.ny=int((shape[0]-self.init_skip[1])/increment_y)
        self.tiles=[]
        for j in range(self.init_skip[1],shape[0]-increment_y+1,increment_y):
            for i in range(self.init_skip[0],shape[1]-increment_x+1,increment_x):
                self.tiles.append(self.data0[j:j+self.tilesize[1],i:i+self.tilesize[0]])

    def resize(self,size_x,size_y=0):
        if size_y==0:
            size_y=size_x
        if size_x == 1.0 and size_y ==1.0:
            return
        else:
            for i in range(len(self.tiles)):
                s_tile=self.tiles[i]
                s_tile=cv2.resize(s_tile,(size_x,size_y),interpolation=cv2.INTER_NEAREST)
                self.tiles[i]=s_tile
        self.tilesize=(size_x,size_y)

    def orient(self,tile,orientation):
        if orientation<4:
            nrot=orientation
            flip=0
        else: 
            nrot=orientation-4
            flip=1
        newtile=np.rot90(tile,nrot)
        if flip==1:
            newtile=np.flip(newtile,axis=0)
        return newtile

    def place_tile(self,image,tile,pos,orientation=0):
        shape=image.shape
        x0=int(pos[0]/self.tilesize[0])*self.tilesize[0]
        y0=int(pos[1]/self.tilesize[1])*self.tilesize[1]
        newtile=self.tiles[tile]
        if orientation !=0:
            newtile=self.orient(newtile,orientation)
        image[y0:y0+self.tilesize[1],x0:x0+self.tilesize[0],0:3]=newtile
        return image

    def display_map(self,display_fac=3):
        new_im=cv2.resize(self.data0,(0,0),fx=display_fac,fy=display_fac,interpolation=cv2.INTER_NEAREST)
        myim.img_viewer(new_im)

    def save_tiles(self,game,append=0,filename='0'):
        gamedir=self.root_dir+'/'+game+'/'
        os.makedirs(gamedir,exist_ok=True)
        n_per_file=300
        nx=10
        ny=int(n_per_file/nx)
        nfiles=int(self.nx*self.ny/n_per_file)+1
        tile0=0
        for file in range(nfiles):
            endpos=min(n_per_file,len(self.tiles)-tile0)
            grid=np.zeros((ny*self.tilesize[0],nx*self.tilesize[1],3),dtype='uint8')
            i=0
            for tile in range(tile0,endpos):
                xpos=(i%nx)*self.tilesize[0]
                ypos=int(i/nx)*self.tilesize[1]
                grid[ypos:ypos+self.tilesize[1],xpos:xpos+self.tilesize[0]]=self.tiles[tile][:,:,0:3]
                i+=1
            tile0+=n_per_file
            if filename=='0':
                cv2.imwrite(gamedir+str(file)+'.png',grid)
            else:
                cv2.imwrite(gamedir+str(filename)+'.png',grid)
        return grid

    def pos_to_tile(self,pos):
        increment_x=self.tilesize[0]+self.tilesep[0]
        increment_y=self.tilesize[1]+self.tilesep[1]
        tx=int((pos[0]-self.init_skip[0])/increment_x)
        ty=int((pos[1]-self.init_skip[1])/increment_y)
        tilenum=tx+ty*self.nx
        return tilenum

