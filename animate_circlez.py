import subprocess
import os
import math

cmd="magick composite "
options="-gravity center -geometry "
options2=" -rotate "
img_ext=".png"
indir2="..\\Pacman\\img_Pacman_small\\"
indir="..\\Pacman\\img_ghost\\"
outdir="img_out"
outfile="circlepacz.gif"
background="blank.png"
nin2=4
nin=2
nout=20
center_x=0
center_y=0
t_start=-20
t_separation=20
t_step=10
radius=200
py=1000

img_index=0
img_index2=0
t=t_start
for num in range(nout):
    numstr="{0:04d}".format(img_index)
    numstr2="{0:04d}".format(img_index2)
    numstr_out="{0:04d}".format(num)
    filename=numstr+img_ext
    filename2=numstr2+img_ext
    filename_out=numstr_out+img_ext
    pos_x=radius*math.cos(math.pi*t/180.0)
    pos_y=radius*math.sin(math.pi*t/180.0)
    offsets=[pos_x,0]
    if offsets[0]>=0:
        sign1="+"
    else:
        sign1=""
    if offsets[1]>=0:
        sign2="+"
    else:
        sign2=""
    offsets_str=sign1+str(offsets[0])+sign2+str(offsets[1])
    t2=t+t_separation
    pos_x2=radius*math.cos(math.pi*t2/180.0)
    pos_y2=radius*math.sin(math.pi*t2/180.0)
    offsets=[pos_x2,0]
    if offsets[0]>=0:
        sign1="+"
    else:
        sign1=""
    if offsets[1]>=0:
        sign2="+"
    else:
        sign2=""
    offsets2_str=sign1+str(offsets[0])+sign2+str(offsets[1])
    fullcmd=cmd+indir+filename+" "+background+" "+options+offsets_str+" "+"tempor.png"
    print(fullcmd)
    returned_value=os.system(fullcmd)
    fullcmd=cmd+indir2+filename2+" "+"tempor.png"+" "+options+offsets2_str+" "+outdir+"\\"+filename_out
    print(fullcmd)
    returned_value=os.system(fullcmd)
    if img_index==nin-1:
        img_index=0
    else:
        img_index+=1
    if img_index2==nin2-1:
        img_index2=0
    else:
        img_index2+=1
    t+=t_step

animate_cmd="magick convert -dispose previous "+outdir+"\\*.png "+outfile
print(animate_cmd)
returned_value=os.system(animate_cmd)

