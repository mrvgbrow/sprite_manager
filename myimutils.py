import math
import imageio
import genutils as genu
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mycolortools as mycolor
import os
import sys
import time
import shutil
import plotfunctions as pfunc
from PIL import Image

def add_images(image2,image1,x,y):
    x1,x2,y1,y2=get_overlap(image1,image2,x,y)
    shape=image2.shape
    image3=image1.copy()
    xmin=0
    xmax=shape[1]
    ymin=0
    ymax=shape[0]
    if y1==0:
        ymin=shape[0]-y2+y1
        ymax=shape[0]
    if y2==image1.shape[0]-1:
        ymin=0
        ymax=y2-y1
    if x1==0:
        xmin=shape[1]-x2+x1
        xmax=shape[1]
    if x2==image1.shape[1]-1:
        xmin=0
        xmax=x2-x1
    if x1==0 and x2==0 and y1==0 and y2==0:
        return image3.astype('uint8')
    for c in range(0,3):
        image3[y1:y2,x1:x2,c]=(image2[ymin:ymax,xmin:xmax,3]/255.0*image2[ymin:ymax,xmin:xmax,c]+(1.0-image2[ymin:ymax,xmin:xmax,3]/255.0)*image1[y1:y2,x1:x2,c])
    return image3.astype('uint8')

def overlay_two(image2,image1,im2frac):
    image3=image1.copy()
    for c in range(0,3):
        image3[:,:,c]=(im2frac*image2[:,:,c]+(1.0-im2frac)*image1[:,:,c])
    return image3.astype('uint8')

def add_alpha_channel(images):
    x_size=images[0].shape[0]
    y_size=images[0].shape[1]
    for i in range(len(images)):
        images[i]=np.dstack((images[i],np.ones((x_size,y_size),'uint8')*255))
    return images

def fade_ims(ims1,ims2):
    nframes=len(ims1)
    alphas=genu.sample_line((0.,255.),(255.,0.),nframes)
    ims3=[]
    for i in range(nframes):
        ims3.append(overlay_two(ims1[i],ims2[i],alphas[i][1]/255.))
    return ims3

def swipe_ims(ims1,ims2,shift=0):
    nframes=len(ims1)
    xarr=genu.sample_line((ims1[0].shape[1]-1,0),(0,0),nframes)
    ims3=[]
    for i in range(nframes):
        print(xarr[i][0])
        overlap=ims1[0].shape[1]-xarr[i][0]
        if shift==1:
            ims1[i][:,0:xarr[i][0],:]=ims1[i][:,overlap:ims1[0].shape[1],:]
        ims3.append(add_images(ims2[i],ims1[i],xarr[i][0],0))
    return ims3

def get_overlap(image1,image2,x,y):
    xlim=image1.shape[1]
    ylim=image1.shape[0]
    y1,y2=y,y+image2.shape[0]
    x1,x2=x,x+image2.shape[1]
    if xlim<x2:
        x2=xlim-1
    if x1<0:
        x1=0
    if y1<0:
        y1=0
    if x2<0 or y2<0 or xlim<x1 or ylim<y1:
        y1=0
        y2=0
        x1=0
        x2=0
    if ylim<y2:
        y2=ylim-1
    return [x1,x2,y1,y2]

def read_gif(infile):
    im=Image.open(infile)
    allims=[]
    durations=[]
    for i in range(im.n_frames):
        im.seek(i)
        if im.n_frames>1:
            if 'duration' in im.info.keys():
                durations.append(im.info['duration'])
            else:
                durations.append(10)
        pix=np.array(im.convert('RGB'))
        pix=pix[:,:,[2,1,0]]
        allims.append(pix)
        
    return (allims,durations)

def read_imdir(dir):
    name,extension=os.path.splitext(dir)
    if extension=='.gif':
        gifarr,durations=read_gif(dir)
        return (gifarr,durations)
#        make_outdir('temp',1)
#        deanimate_gif(dir,'temp')
#        dir='temp'
    if extension=='.png':
        return ([cv2.imread(dir,cv2.IMREAD_UNCHANGED)],[0])
    elif os.path.isdir(dir):
        files=os.listdir(dir)
        imlist=[]
        
        i=0
        for thisim in files:
            image=cv2.imread(dir+"/"+thisim,cv2.IMREAD_UNCHANGED)
            imlist.append(image)
            i+=1
        imlist=np.stack((imlist[:]))
        return imlist

def click_point(event,x,y,flags,param):
    global refPt,clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt=(x,y)
        clicked=1

def click_line(event,x,y,flags,param):
    global refPt1,refPt2,clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt1=(x,y)
        clicked=1
    if event == cv2.EVENT_MOUSEMOVE:
        refPt2=(x,y)
    if event == cv2.EVENT_LBUTTONUP:
        refPt2=(x,y)
        clicked=0


def click_mouseover(event,x,y,flags,param):
    global refPt,clicked
    
    if event == cv2.EVENT_MOUSEMOVE:
        refPt=(x,y)

def click_edit(event,x,y,flags,param):
    global refPt,clicked,clicked2
    
    if event == cv2.EVENT_MOUSEMOVE:
        refPt=(x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked=1
    if event == cv2.EVENT_RBUTTONDOWN:
        clicked2=1

def clicksave(event,x,y,flags,param):
    global clicked,pointlist
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked=1
    if event == cv2.EVENT_MOUSEMOVE and clicked==1:
        if len(pointlist)>0:
            previous_point=pointlist[len(pointlist)-1]
        else:
            previous_point=(0,0)
        if previous_point[0]!=x or previous_point[1]!=y:
            pointlist.append((x,y))
    if event == cv2.EVENT_LBUTTONUP:
        clicked=0

def click_rectangle(event,x,y,flags,param):
    global boxcorners,clicked,cornertemp
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked=1
        boxcorners=[(x,y)]
        cornertemp=[]
    if event == cv2.EVENT_MOUSEMOVE and clicked==1:
        cornertemp=[(x,y)]
    if event == cv2.EVENT_LBUTTONUP:
        clicked=0
        boxcorners.append((x,y))

def click_rectangle_fixed(event,x,y,flags,param):
    global boxcorner,clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked=1
        boxcorner=[(x,y)]
    if event == cv2.EVENT_MOUSEMOVE and clicked==1:
        boxcorner=[(x,y)]
    if event == cv2.EVENT_LBUTTONUP:
        clicked=0

def capture_line(image,mode=0):
    global clicked,refPt1,refPt2

    clicked=0
    refPt1=(0,0)
    refPt2=(0,0)
    if mode==1:
        cv2.namedWindow("image",flags=cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow("image")

    cv2.setMouseCallback("image",click_line)

    while True:
        clone=image.copy()
        if clicked==1:
            cv2.line(clone,refPt1,refPt2,(0,255,0),1)
        cv2.imshow("image",clone[:,:,:3])
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break
        if key == ord("x"):
            print(refPt1,refPt2)
            break

    cv2.destroyAllWindows()
    return refPt1,refPt2

def capture_point(image,mode=0):
    global clicked,refPt

    clicked=0
    refPt=(0,0)
    if mode==1:
        cv2.namedWindow("image",flags=cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow("image")

    cv2.setMouseCallback("image",click_point)

    while True:
        cv2.imshow("image",image[:,:,:3])
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break
        if clicked==1:
            break

    cv2.destroyAllWindows()
    return refPt

def capture_path(image):
    global clicked,pointlist
    clicked=0
    pointlist=[]

    image = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
    h1,w1=image.shape[:2] 
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",clicksave)

    while True:
        cv2.imshow("image",image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    return pointlist

def capture_box_fixed(image,dims):
    global clicked,boxcorner
    clicked=0
    boxcorner=[(0,0)]

    cv2.namedWindow("image",flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image",click_rectangle_fixed)

    while True:     
        clone=image.copy()
        cv2.rectangle(clone, boxcorner[0], (boxcorner[0][0]+dims[0],boxcorner[0][1]+dims[1]), (0, 255, 0), 1)
        cv2.imshow("image",clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            break

    cv2.destroyAllWindows()
    return boxcorner[0]

def capture_box(image):
    global clicked,boxcorners,cornertemp
    clicked=0
    boxcorners=[]
    cornertemp=[]

    cv2.namedWindow("image",flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image",click_rectangle)

    while True:     
        clone=image.copy()
        if len(cornertemp)>0:
            cv2.rectangle(clone, boxcorners[0], cornertemp[0], (0, 255, 0), 1)
        cv2.imshow("image",clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            break

    cv2.destroyAllWindows()
    return boxcorners

def make_outdir(dir,force):
    if force==1 and os.path.isdir(dir):
        shutil.rmtree(dir)
    try:
        os.mkdir(dir)
    except OSError as e:
        print("Error in making directory "+dir+": "+e.strerror)

def sprite_center(mask):
    indices=np.asarray(np.nonzero(mask))
    return indices.mean(axis=1)

def maketransparent_withmask(image,mask):
    if image.shape[2]==4:
        b_channel, g_channel, r_channel,test=cv2.split(image)
    else:
        b_channel, g_channel, r_channel=cv2.split(image)
    image=cv2.merge((b_channel,g_channel,r_channel,mask.astype(b_channel.dtype)*255))
    return image

def do_grabcut_rect(image,corner1,corner2,iter):
    mask=np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect=(corner1[0],corner1[1],corner2[0]-corner1[0],corner2[1]-corner1[1])
    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,iter,cv2.GC_INIT_WITH_RECT)
    mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = image*mask2[:,:,np.newaxis]

    return [image,mask2]

def imshow_loop(image,title,exit_char):
    cv2.namedWindow(title)
    while True:
        cv2.imshow(title,image)
        key = cv2.waitKey(1) & 0xFF
        if key==ord(exit_char):
            break
    cv2.destroyAllWindows()

def imfile_name(index,indir):
    numstr="{0:04d}".format(index)
    filename=numstr+".png"
    if indir != "":
        return indir+"/"+filename
    else:
        return filename

def animate_dir(dirname,delay,outfile):
    cmd="magick convert -dispose previous -set delay "+str(delay)+" "+dirname+"/*.png "+outfile
    returned_value=os.system(cmd)

def deanimate_gif(infile,outdir):
    cmd="magick convert -coalesce "+infile+" "+outdir+"/%04d.png"
    returned_value=os.system(cmd)

def make_mask(indices,shape):
    mask=np.zeros((shape[0],shape[1]),np.bool)
    mask[indices]=1
    return mask

def apply_mask(image,mask):
    image_new = image*mask[:,:,np.newaxis]
    return image_new

def mask_replace(image,mask,color):
    mask_indices=np.nonzero(mask)
    image[mask_indices[0],mask_indices[1],0] = color[0]
    image[mask_indices[0],mask_indices[1],1] = color[1]
    image[mask_indices[0],mask_indices[1],2] = color[2]
    if len(color)>3:
        image[mask_indices[0],mask_indices[1],3] = color[3]
    return image

def convert_to_PIL(list_np_array):
    new_array=[]
    if len(list_np_array[0].shape)==2:
        for im in list_np_array:
            new_array.append(Image.fromarray(im.astype('uint8')))
        return new_array
    if list_np_array[0].shape[2]==4:
        color_dims=[2,1,0,3]
    else:
        color_dims=[2,1,0]
    for im in list_np_array:
        im=im[:,:,color_dims]
        new_array.append(Image.fromarray(im.astype('uint8')))
    return new_array

def write_gif(np_image,outfile):
    im=convert_to_PIL([np_image])
    im[0].save(outfile,palette='P')

def write_animation(pil_array,durations,outfile,pil=1):
    if pil==0:
        imageio.mimsave(outfile,pil_array,duration=np.mean(durations)/1000)
        return
    for i in range(len(pil_array)):
        pil_array[i]=pil_array[i].convert("P")
    if len(durations)>0:
        pil_array[0].save(outfile,save_all=True,append_images=pil_array[1:],duration=durations,loop=0,palette='P')
    else:
        pil_array[0].save(outfile,save_all=True,append_images=pil_array[1:],loop=0,palette='P')

def img_viewer(image,title='Image',mode=0):
    global refPt

    refPt=(0,0)
    info_window=np.zeros((40,500,3))
    print(image.shape)
    if mode==1:
        cv2.namedWindow(title,flags=cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(title)
    cv2.setMouseCallback(title,click_mouseover)
    while True:
        info_window*=0
        if check_in_image(image,refPt,(1,1)):
            colorstring=str(image[(refPt[1],refPt[0])])
        else:
            colorstring=''
        info_string='(x,y) = '+str(refPt)+'  RGB='+colorstring
        cv2.putText(info_window,info_string,(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow(title,image.astype('uint8'))
        cv2.imshow('Info',info_window)
        wait=1
        key = cv2.waitKey(wait) & 0xFF
        if key==ord('x'):
            break
    cv2.destroyAllWindows()

def gif_plot(y,images,durations,title='',pause=0,xlabel='x',ylabel='y',xmin='None',ymin='None',xmax='None',ymax='None'):
#    xarr=pfunc.setup_plot(y,xlabel='frame',title=title)
    dpi=200
    figsize=(images[0].shape[0]/dpi*1.4,images[0].shape[0]/dpi)
    fig=plt.figure(figsize=figsize,dpi=dpi)
    x=np.array(range(len(y)))
    y=np.array(y)
#    cv2.namedWindow('plot',flags=cv2.WINDOW_NORMAL)
    cv2.namedWindow('plot')
    line1,=plt.plot(x,y,'ko-',markersize=0)
    plt.xticks(size=3)
    plt.yticks(size=3)
    plt.tight_layout()
    if type(xmin)==str:
        xmin=np.min(x)
    if type(xmax)==str:
        xmax=np.max(x)
    if type(ymin)==str:
        ymin=np.min(y)
    if type(ymax)==str:
        ymax=np.max(y)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.ylabel(xlabel,size=5)
    plt.xlabel(ylabel,size=5)
    i=0
    while True:
        line1.set_xdata(x[:i])
        line1.set_ydata(y[:i])
        fig.canvas.draw()

        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        cv2.imshow('plot',img)
        cv2.imshow("image",images[i])
        wait=durations[i]
        key = cv2.waitKey(wait) & 0xFF
        if key==ord(']'):
            i=(i+1)%len(images)
        if key==ord('['):
            i=(i-1)%len(images)
        if key==ord('p'):
            pause=(pause+1)%2
        if pause==0:
            i=(i+1)%len(images)
        if key==ord('x'):
            break


def gif_viewer(images,durations,title,pause=0):
    global refPt

    refPt=(0,0)
    i=0
    speed_factor=1.0
    info_window=np.zeros((40,500,3))
    print(images[0].shape,len(images))
    cv2.namedWindow(title)
    cv2.setMouseCallback(title,click_mouseover)
    while True:
        info_window*=0
        info_string='Frame: '+str(i)+', delay = '+str(speed_factor)+', (x,y) = '+str(refPt)
        cv2.putText(info_window,info_string,(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        thisim=images[i]
        cv2.imshow(title,thisim.astype('uint8'))
        cv2.imshow('Info',info_window)
        if len(durations)>0 and durations[i]>0:
            wait=int(durations[i]*speed_factor)
        else:
            wait=5
        key = cv2.waitKey(wait) & 0xFF
        if key==ord('x'):
            break
        if key==ord('+'):
            speed_factor-=0.3
        if key==ord('-'):
            speed_factor+=0.3
        if key==ord(']'):
            i=(i+1)%len(images)
        if key==ord('['):
            i=(i-1)%len(images)
        if key==ord('}'):
            i=(i+10)%len(images)
        if key==ord('{'):
            i=(i-10)%len(images)
        if key==ord('p'):
            pause=(pause+1)%2
        if pause==0:
            i=(i+1)%len(images)
    cv2.destroyAllWindows()
    return i

def capture_path_full(images):
    global refPt

    refPt=(0,0)
    i=0
    info_window=np.zeros((120,1000,3))
    cv2.namedWindow('Define Sprite Path')
    cv2.setMouseCallback('Define Sprite Path',click_mouseover)
    path=[(-2,-2)]*len(images)
    size=[(-2,-2)]*len(images)
    angle=[-2]*len(images)
    angle2=[-2]*len(images)
    angle3=[-2]*len(images)
    flipx=[-2]*len(images)
    flipy=[-2]*len(images)
    opacity=[-2]*len(images)
    n_trans=20
    opac_step=1/n_trans
    angle_increment=10
    ref_angle=0
    accel=[-2]*len(images)
    loop=-1
    while True:
        info_window*=0
        info_string='Frame: '+str(i)+', (x,y) = '+str(refPt)
        if path[i]!=(-2,-2):
            info_string+=' Frame Marked: '+str(path[i])
        if loop>-1:
            info_string+='   Looping at frame '+str(i)
        if loop<-1:
            info_string+='   Reverse looping at frame '+str(i)
        info_string2=''
        if size[i]!=(-2,-2):
            info_string2+=' Size: '+'%.2f'%size[i][0]+' %.2f'%size[i][1]
        if angle[i]!=-2:
            info_string2+='  Angle: '+str(angle[i])
        if angle2[i]!=-2:
            info_string2+='  ZX Angle: '+str(angle2[i])
        if angle3[i]!=-2:
            info_string2+='  ZY Angle: '+str(angle3[i])
        if opacity[i]!=-2:
            info_string2+='  Opacity: '+str(opacity[i])
        info_string3=''
        if flipx[i]!=-2:
            info_string3+='Flipx: '+str(flipx[i])
        if flipy[i]!=-2:
            info_string3+='  Flipy: '+str(flipy[i])
        if accel[i]!=-2:
            info_string3+='  Acceleration: '+str(accel[i]-1)
        cv2.putText(info_window,info_string,(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(info_window,info_string2,(10,65),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(info_window,info_string3,(10,95),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        thisim=images[i]
        cv2.imshow('Define Sprite Path',thisim.astype('uint8'))
        cv2.imshow('Info',info_window)
        wait=5
        key = cv2.waitKey(wait) & 0xFF
        if key==ord('x'):
            break
        if key==ord(']'):
            i=(i+1)%len(images)
        if key==ord('}'):
            i=(i+10)%len(images)
        if key==ord('['):
            i=(i-1)%len(images)
        if key==ord('{'):
            i=(i-10)%len(images)
        if key==ord('a'):
            path[i]=refPt
        if key==ord('A'):
            accel[i]=(accel[i]+1)%3
        if key==ord('n'):
            if opacity[i]<0:
                opacity[i]=1.0
            else:
                opacity[i]=(round((opacity[i]/opac_step+1))%(n_trans+1))/(n_trans)
        if key==ord('N'):
            if opacity[i]<0:
                opacity[i]=1.0
            else:
                opacity[i]=(round((opacity[i]/opac_step-1))%(n_trans+1))/(n_trans)
        if key==ord('r'):
            if angle[i]==-2:
                angle[i]=ref_angle
            else:
                angle[i]=(angle[i]+angle_increment)
        if key==ord('R'):
            if angle[i]==-2:
                angle[i]=ref_angle
            else:
                angle[i]=(angle[i]-angle_increment)
        if key==ord('z'):
            if angle2[i]==-2:
                angle2[i]=ref_angle
            else:
                angle2[i]=(angle2[i]+angle_increment)
        if key==ord('Z'):
            if angle2[i]==-2:
                angle2[i]=ref_angle
            else:
                angle2[i]=(angle2[i]-angle_increment)
        if key==ord('f'):
            if flipx[i]==-2:
                flipx[i]=1
            else:
                flipx[i]=(flipx[i]+1)%2
        if key==ord('F'):
            if flipy[i]==-2:
                flipy[i]=1
            else:
                flipy[i]=(flipy[i]+1)%2
        if key==ord('y'):
            if angle3[i]==-2:
                angle3[i]=ref_angle
            else:
                angle3[i]=(angle3[i]+angle_increment)
        if key==ord('Y'):
            if angle3[i]==-2:
                angle3[i]=ref_angle
            else:
                angle3[i]=(angle3[i]-angle_increment)
        if key==ord('s'):
            if size[i]==(-2,-2):
                size[i]=(1,1)
            else:
                size[i]=(size[i][0]*1.1,size[i][1]*1.1)
        if key==ord('S'):
            if size[i]==(-2,-2):
                size[i]=(1,1)
            else:
                size[i]=(size[i][0]/1.1,size[i][1]/1.1)
        if key==ord('l'):
            path[i]=(-1,-1)
        if key==ord('L'):
            if loop!=-1:
                loop=-loop
            else:
                loop=i
    position=path[0]
    nowsize=size[0]
    increment=(0,0)
    size_increment=(0,0)
    path=fill_path(path)
    path=[(int(p[0]),int(p[1])) for p in path]
    size=fill_path(size)
    angle=fill_val(angle,default=0.0)
    angle2=fill_val(angle2,default=0.0)
    angle3=fill_val(angle3,default=0.0)
    opacity=fill_val(opacity,default=1.0)
    flipx=fill_discrete(flipx)
    flipy=fill_discrete(flipy)
    accel=fill_discrete(accel)
    if loop<-1:
        for i in range(-loop,-2*loop):
            ref_index=-2*loop-i-1
            path[i]=path[ref_index]
            size[i]=size[ref_index]
            angle[i]=angle[ref_index]
            angle2[i]=angle2[ref_index]
            angle3[i]=angle3[ref_index]
            flipx[i]=flipx[ref_index]
            flipy[i]=flipy[ref_index]
            accel[i]=accel[ref_index]
            opacity[i]=opacity[ref_index]
        loop=-2*loop
    if loop>-1:
        for i in range(loop,len(images)):
            ref_index=i%loop
            path[i]=path[ref_index]
            size[i]=size[ref_index]
            angle[i]=angle[ref_index]
            angle2[i]=angle2[ref_index]
            angle3[i]=angle3[ref_index]
            flipx[i]=flipx[ref_index]
            flipy[i]=flipy[ref_index]
            accel[i]=accel[ref_index]
            opacity[i]=opacity[ref_index]
    cv2.destroyAllWindows()
    return path,size,angle,angle2,angle3,flipx,flipy,opacity

def fill_discrete(array,nullval=-2):
    nowarr=array[0]
    for i in range(len(array)):
        if (array[i]==nullval):
            array[i]=int(nowarr)
        else:
            nowarr=array[i]
    return array

def fill_val(array,nullval=-2,default=0.0):
    nowarr=default
    increment=0
    for i in range(len(array)):
        if (array[i]==nullval):
            nowarr=nowarr+increment
            array[i]=float(nowarr)
        else:
            nowarr=array[i]
            for j in range(i+1,len(array)):
                if array[j]!=nullval:
                    arrdiff=array[j]-array[i]
                    #if angle2[j]>=angle2[i]:
                    #    anglediff2=angle2[j]-angle2[i]
                    #else:
                    #    anglediff2=angle2[j]+360-angle2[i]
                    increment=arrdiff/(j-i)
                    break
                increment=0
    return array

def fill_path(array,nullval=(-2,-2)):
    nowarr=array[0]
    increment=(0,0)
    for i in range(len(array)):
        if (array[i]==nullval):
            nowarr=(nowarr[0]+increment[0],nowarr[1]+increment[1])
            array[i]=(nowarr[0],nowarr[1])
        else:
            nowarr=array[i]
            for j in range(i+1,len(array)):
                if array[j]!=nullval:
                    arrdiff=(array[j][0]-array[i][0],array[j][1]-array[i][1])
                    increment=(arrdiff[0]/(j-i),arrdiff[1]/(j-i))
                    break
                increment=(0,0)
    return array

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def check_in_image(image,refPt,shape):
    if (refPt[0] < 0 or refPt[0]+shape[1] > image.shape[1]):
        return False
    if (refPt[1] < 0 or refPt[1]+shape[0] > image.shape[0]):
        return False
    return True

def force_in_image(image,refPt):
    xnew=refPt[0]
    ynew=refPt[1]
    if (xnew<0):
        xnew=0
    if ynew<0:
        ynew=0
    return (xnew,ynew)

def remove_near_boundary(image,indices,offset_range):
    dimens=image.shape
    indices_of_indices=np.nonzero((indices[0]>offset_range) & (indices[0]<dimens[0]-offset_range) & (indices[1]>offset_range) & (indices[1]<dimens[1]-offset_range))
    return (indices[0][indices_of_indices],indices[1][indices_of_indices])

def indices_in_box(indices,xmin,xmax,ymin,ymax):
    indices_of_indices=np.nonzero((indices[0]>ymin) & (indices[0]<ymax) & (indices[1]>xmin) & (indices[1]<xmax))
    return (indices[0][indices_of_indices],indices[1][indices_of_indices])

def window_from_image(image,position,shape):
    return image[position[1]:position[1]+shape[0],position[0]:position[0]+shape[1]]

def shift_indices(indices,shift):
    return (indices[0]+shift[1],indices[1]+shift[0])

def trim_to_fit(image,indices):
    index_indices=np.where((indices[0]>=0) & (indices[0]<image.shape[0]) & (indices[1]>=0) & (indices[1]<image.shape[1]))
    y=indices[0][index_indices]
    x=indices[1][index_indices]
    return (y,x)

def select_pixels(image,title='Image',color=[255,255,255,255],threshold=25):
    global refPt,clicked,clicked2

    image2=image.copy()
    sys.setrecursionlimit(1000000)
    refPt=(0,0)
    clicked=0
    clicked2=0
    empty=np.array([0,0,0,0])
    info_window=np.zeros((40,500,3))
    cv2.namedWindow(title,flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title,click_edit)
    mask=np.zeros((image2.shape[0],image.shape[1]),'uint8')
    mode='Bucket'
    while True:
        info_window*=0
        info_string="(x,y) = "+str(refPt)+" Mode = "+mode
        cv2.putText(info_window,info_string,(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow(title,image2.astype('uint8'))
        cv2.imshow('Info',info_window)
        key = cv2.waitKey(1) & 0xFF
        if mode == 'Bucket':
            if clicked==1:
                ref_color=image2[refPt[1],refPt[0]]
                mask=mycolor.flood_select(image2,mask,(refPt[1],refPt[0]),ref_color,threshold)
                inds=np.nonzero(mask==1)
                image2[inds]=color
                clicked=0
        if mode == 'Pixel':
            if clicked==1 and not np.array_equal(image2[refPt[1],refPt[0]],color):
                image2[refPt[1],refPt[0]]=color
                mask[refPt[1],refPt[0]]=1
                clicked=0
            elif clicked==1 and np.array_equal(image2[refPt[1],refPt[0]],color):
                image2[refPt[1],refPt[0]]=empty
                mask[refPt[1],refPt[0]]=0
                clicked=0
        if key==ord('b'):
            if mode=='Bucket':
                mode='Pixel'
            else:
                mode='Bucket'
        if key==ord('x'):
            cv2.destroyAllWindows()
            return 0
        if key==ord('s'):
            cv2.destroyAllWindows()
            return mask

def pix_edit(image,title='Image',color=[255,255,255,255]):
    global refPt,clicked,clicked2

    refPt=(0,0)
    clicked=0
    clicked2=0
    color=np.array(color)
    empty=np.array([0,0,0,0])
    info_window=np.zeros((40,500,3))
    cv2.namedWindow(title,flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title,click_edit)
    bucketmode=0
    while True:
        info_window*=0
        info_string="(x,y) = "+str(refPt)+' color = '+str(color)
        cv2.putText(info_window,info_string,(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow(title,image.astype('uint8'))
        cv2.imshow('Info',info_window)
        key = cv2.waitKey(1) & 0xFF
        if bucketmode==1:
            if clicked==1:
                mask=np.zeros((image.shape[0],image.shape[1]),'uint8')
                ref_color=image[refPt[1],refPt[0]]
                mask=mycolor.flood_select(image,mask,(refPt[1],refPt[0]),ref_color,25)
                inds=np.nonzero(mask==1)
                image[inds]=color
                clicked=0
        if bucketmode==0:
            if clicked==1 and not np.array_equal(image[refPt[1],refPt[0]],color):
                image[refPt[1],refPt[0]]=color
                clicked=0
            elif clicked==1 and np.array_equal(image[refPt[1],refPt[0]],color):
                image[refPt[1],refPt[0]]=empty
                clicked=0
        if clicked2==1:
            color=np.array(image[refPt[1],refPt[0]])
            clicked2=0
        if key==ord('b'):
            bucketmode=(bucketmode+1)%2
        if key==ord('x'):
            cv2.destroyAllWindows()
            return 0
        if key==ord('s'):
            cv2.destroyAllWindows()
            return image

def grid_to_indices(xv,yv):
    xv=np.round(xv-np.min(np.round(xv)))
    yv=np.round(yv-np.min(np.round(yv)))
    return([xv.astype('int'),yv.astype('int')])

#def image_map(image,x1,y1,x2,y2):
#    image_new=np.zeros([

def fill_square(image,position,halfside,blend=0,color=[255,255,255]):
    for i in range(-halfside,halfside+1):
        for j in range(-halfside,halfside+1):
            for c in range(0,3):
                image[position[1]+i,position[0]+j,c]=image[position[1]+i,position[0]+j,c]*blend+(1-blend)*color[c]

def find_offset(array1,array2,indices,direction=1,offset_range=3):
    most_matches=0
    for offset in range(-offset_range,offset_range+1):
        if direction==1:
            indices_comp=(indices[0]+offset,indices[1])
        else:
            indices_comp=(indices[0]+offset,indices[1])
        compare=np.nonzero(array1[indices]==array2[indices_comp])
        nmatch=(compare[0].shape)[0]
        if nmatch>most_matches:
            most_matches=nmatch
            offset_ref=offset
    return offset_ref

def axis_ratio_34(inrat):
    dy,dx=inrat.split('x')
    if int(dy)>int(dx):
        fx=int(dy)/int(dx)*3/4
        fy=1
    else:
        fy=int(dx)/int(dy)*3/4
        fx=1
    return (fx,fy)

def add_border(image,color=[30,39,41],width=1):
    #[56,139,191]
    shape=image.shape
    image[0:width,:,0:3]=color
    image[:,0:width,0:3]=color
    image[:,shape[0]-width-1:shape[0],0:3]=color
    image[shape[0]-width-1:shape[0],:,0:3]=color
    return image

def add_texture(image,color1=[100,100,100],color2=[150,150,150],scale=2,bright=1.0):
    shape=image.shape
    scale2=scale*3.8
    if bright!=1.0:
        color1=np.multiply(color1,bright)
        color2=np.multiply(color2,bright)
    if shape[2]>3:
        color1=np.append(color1,0)
        color2=np.append(color2,0)
    grid=np.indices(shape[0:2])
    indices1=np.nonzero(np.logical_or((grid[0]%(scale*2)<scale) & (grid[1]%(scale*2)<scale),(grid[0]%(scale*2)>=scale) & (grid[1]%(scale*2)>=scale)))
    indices2=np.nonzero(np.logical_or((grid[0]%(scale2*2)<scale2) & (grid[1]%(scale2*2)<scale2),(grid[0]%(scale2*2)>=scale2) & (grid[1]%(scale2*2)>=scale2)))
    image[:,:]=color1
    image[indices1]=color2
    image[indices2]=np.multiply(color2,0.7)
    return image

def rotate3d(image,angle1,angle2):
    shape=image.shape
    angle1*=math.pi/180.
    angle2*=math.pi/180.
    pt1_1=[0,0]
    pt2_1=[shape[1]-1,0]
    pt3_1=[0,shape[0]-1]
    src_tri=np.array([pt1_1,pt2_1,pt3_1]).astype(np.float32)

    center=int(shape[1]/2)
    halfwidth=int(shape[1]*math.cos(angle1)/2)
    center2=int(shape[0]/2)
    halfwidth2=int(shape[0]*math.cos(angle2)/2)
    pt1_2=[center-halfwidth,center2-halfwidth2]
    pt2_2=[center+halfwidth,center2-halfwidth2]
    pt3_2=[center-halfwidth,center2+halfwidth2]
    dst_tri=np.array([pt1_2,pt2_2,pt3_2]).astype(np.float32)

    warp_mat=cv2.getAffineTransform(src_tri,dst_tri)
    image=cv2.warpAffine(image,warp_mat,(shape[1],shape[0]))
    return image

def apply_opacity(image,opacity):
    new_img=image.copy()
    new_img[:,:,3]=new_img[:,:,3].astype('float')*opacity
    return new_img.astype('uint8')

def overlay_grid(image,xscale,yscale,xoffset,yoffset,color=[100,100,100],label=0):
    new_img=image.copy()
    for x in range(xoffset,image.shape[1],xscale):
        new_img=cv2.line(new_img,(x,0),(x,image.shape[0]-1),color,1)
    for y in range(yoffset,image.shape[0],yscale):
        new_img=cv2.line(new_img,(0,y),(image.shape[1]-1,y),color,1)
    if label==1:
        new_img=label_grid(new_img,xscale,yscale,xoffset,yoffset,color=color)
    return new_img

def label_grid(image,xscale,yscale,xoffset,yoffset,color=[100,100,100]):
    i=0
    for x in range(xoffset-int(xscale/2)-3,image.shape[1],xscale):
        if x<0:
            continue
        char='A'
        for y in range(yoffset-int(yscale/2)+3,image.shape[0],yscale):
            if y<0:
                continue
            string=char+str(i)
            image=cv2.putText(image,string,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1,cv2.LINE_AA)
            char=chr(ord(char)+1)
        i+=1
    return image
