import math
import mycolortools as mycolor
import numpy as np
import myimutils as myim
import sys
import os
import cv2

class tileset:
    def __init__(self,data,tilesize=(8,8),tilesep=(0,0),init_skip=(0,0),game=''):
        self.root_dir='C:/Users/sp4ce/Google Drive/Documents/Tiles'
        if type(data) is str:
            filename=self.root_dir+'/'+game+'/'+data+'.png'
            data,dum=myim.read_imdir(filename)
            self.data0=data[0]
        else:
            self.data0=data
        self.tilesize=tilesize
        self.tilesep=tilesep
        self.init_skip=init_skip
        self.extract_tiles()
        print(len(self.tiles))

    def append(self,tiles):
        if tiles[0].shape[0]==self.tilesize[0]:
            for tile in tiles:
                self.tiles.append(tile)

    def extract_tiles(self):
        shape=self.data0.shape
        print(self.tilesep,self.tilesize,self.init_skip)
        increment_x=self.tilesize[0]+self.tilesep[0]
        increment_y=self.tilesize[1]+self.tilesep[1]
        self.nx=int((shape[1]-self.init_skip[0]+self.tilesep[0])/increment_x)
        self.ny=int((shape[0]-self.init_skip[1]+self.tilesep[1])/increment_y)
        self.tiles=[]
        for j in range(self.init_skip[1],shape[0],increment_y):
            for i in range(self.init_skip[0],shape[1],increment_x):
                tile=self.data0[j:j+self.tilesize[1],i:i+self.tilesize[0]]
                tilemean=np.mean(tile)
                if (tilemean != 255):
                    self.tiles.append(tile)

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

    def flip_all(self,direction=1):
        shape=self.tiles[0].shape
        if direction==1:
            for i in range(len(self.tiles)):
                self.tiles[i]=self.tiles[i][-1::-1,0:shape[1],0:3]
        else:
            for i in range(len(self.tiles)):
                self.tiles[i]=self.tiles[0:shape[0],-1::-1,0:3]


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

    def place_tile(self,image,tile,pos,orientation=0,force=0):
        shape=image.shape
        print('tilenum ',tile)
        if force==0:
            x0=int(pos[0]/self.tilesize[0])*self.tilesize[0]
            y0=int(pos[1]/self.tilesize[1])*self.tilesize[1]
        else:
            x0=pos[0]
            y0=pos[1]
        newtile=self.tiles[tile]
        if orientation !=0:
            newtile=self.orient(newtile,orientation)
        image[y0:y0+self.tilesize[1],x0:x0+self.tilesize[0],0:3]=newtile
        return image

    def make_tileset_img(self,nx,ny,first=0,last=-1,scale=1,border=0):
        if last==-1:
            last=min(len(self.tiles),nx*ny)
        tilesize=(self.tilesize[0]*scale+border,self.tilesize[1]*scale+border)
        grid=np.ones((ny*tilesize[0],nx*tilesize[1],3),dtype='uint8')*255
        i=0
        for tile in range(first,last):
            s_tile=cv2.resize(self.tiles[tile][:,:,0:3],(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            xpos=(i%nx)*tilesize[0]
            ypos=int(i/nx)*tilesize[1]
            grid[ypos:ypos+tilesize[1]-border,xpos:xpos+tilesize[0]-border]=s_tile
            i+=1
        return grid

    def save_tiles(self,game,append=0,filename='0'):
        gamedir=self.root_dir+'/'+game+'/'
        os.makedirs(gamedir,exist_ok=True)
        n_per_file=900
        nx=30
        nfiles=int(len(self.tiles)/n_per_file)+1
        ny=int(len(self.tiles)/nx)+1
        tile0=0
        for file in range(nfiles):
            length=min(n_per_file,len(self.tiles)-tile0)
            endpos=tile0+length
            grid=self.make_tileset_img(nx,ny,tile0,endpos,border=0)
            tile0+=n_per_file
            if filename=='0':
                cv2.imwrite(gamedir+str(file)+'.png',grid)
            else:
                cv2.imwrite(gamedir+str(filename)+'.png',grid)
        return grid

    def pos_to_tile(self,pos,nx=0,border=0):
        if nx==0:
            nx=self.nx
        increment_y=self.tilesize[1]+self.tilesep[1]+border
        increment_x=self.tilesize[0]+self.tilesep[0]+border
        tx=int((pos[0]-self.init_skip[0])/increment_x)
        ty=int((pos[1]-self.init_skip[1])/increment_y)
        tilenum=tx+ty*nx
        newPt=(increment_x*tx,increment_y*ty)
        return tilenum,newPt

    def tile_to_pos(self,tile,nx=0):
        if nx==0:
            nx=self.nx
        increment_y=self.tilesize[1]+self.tilesep[1]
        increment_x=self.tilesize[0]+self.tilesep[0]
        xpos=(tile%nx)*increment_x
        ypos=int(tile/nx)*increment_y
        return (xpos,ypos)

    def lay_tiles(self,image,tile_list,init_pos=(0,0)):
        posx=init_pos[0]
        posy=init_pos[1]
        for i in range(len(tile_list)):
            if tile_list[i]==-1:
                posy+=self.tilesize[1]
                continue
            image[posy:posy+self.tilesize[1],posx:posx+self.tilesize[0]]=self.tiles[tile_list[i]]
            posx+=self.tilesize[0]
        return image

def read_tile_file(filename):
    f=open(filename,'r')
    fl=f.readlines()
    f.close()
    all_tiles=[]
    tiles=[]
    for line in fl:
        ltiles=line.split(',')
        if ltiles[0].rstrip()=='FRAME':
            all_tiles.append(tiles.copy())
            tiles=[]
            continue
        for tile in ltiles:
            tiles.append(int(tile))
        tiles.append(-1)
    all_tiles.append(tiles.copy())
    return all_tiles
