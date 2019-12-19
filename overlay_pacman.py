import subprocess
import os

dir="img"

cmd="magick composite "
options="-gravity center -geometry "
img_start=""
img_ext=".png"
indir="img_out_mod\\"
indir2="..\\Pacman\\"
filename_in="soserious_2.jpg"
outdir="img_out2\\"
outfile="seriouspac.gif"
nin=30
pos_x=-105
pos_y=-260

step=0
for num in range(nin):
    numstr="{0:04d}".format(num)
    filename=img_start+numstr+img_ext
    offsets=[pos_x,pos_y]
    if offsets[0]>=0:
        sign1="+"
    else:
        sign1=""
    if offsets[1]>=0:
        sign2="+"
    else:
        sign2=""
    offsets_str=sign1+str(offsets[0])+sign2+str(offsets[1])
    fullcmd=cmd+indir+filename+" "+indir2+filename_in+" "+options+offsets_str+" "+outdir+"\\"+filename
    print(fullcmd)
    returned_value=os.system(fullcmd)

animate_cmd="magick convert -dispose previous "+outdir+"\\*.png "+outfile 
print(animate_cmd)
returned_value=os.system(animate_cmd)
