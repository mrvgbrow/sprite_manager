import subprocess
import os
import math

cmd="magick composite "
options="-gravity center -geometry "
options2=" -rotate "
outdir="img_out"
outfile="circlepac.gif"
background="blank.png"
nin2=4
nin=2
nout=60
center_x=0
center_y=0
t_start=0
t_separation=30
t_step=6
radius=200
rotate1_start=-90
rotate2_start=-90

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
    rotate1=rotate1_start+t
    returned_value=os.system("magick convert -rotate \""+str(rotate1)+"\" "+indir+filename+" -transparent white temp1.png")
    offsets=[pos_x,pos_y]
    print(pos_x,pos_y)
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
    rotate2=rotate2_start+t2
    returned_value=os.system("magick convert -rotate \""+str(rotate2)+"\" "+indir2+filename2+" -transparent white temp2.png")
    offsets=[pos_x2,pos_y2]
    print(pos_x,pos_y)
    if offsets[0]>=0:
        sign1="+"
    else:
        sign1=""
    if offsets[1]>=0:
        sign2="+"
    else:
        sign2=""
    offsets2_str=sign1+str(offsets[0])+sign2+str(offsets[1])
    fullcmd=cmd+"temp1.png"+" "+background+" "+options+offsets_str+" "+"tempor.png"
    print(fullcmd)
    returned_value=os.system(fullcmd)
    fullcmd=cmd+"temp2.png"+" "+"tempor.png"+" "+options+offsets2_str+" "+outdir+"\\"+filename_out
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

