#!/c/Users/sp4ce/AppData/Local/Programs/Python/Python38-32/python

import os

map=['777888999','777888999','777888999','777888999','777888999','777888999','777888999','777888999','777888999']
x_0=45
y_0=50
x_increment=107
y_increment=106

i=0
for x in range(9):
    for y in range(9):
        x_pos=x_0+x_increment*x
        y_pos=y_0+y_increment*y 
        if i%2==0: 
            infile="test.png"
            outfile="test2.png"
        else:
            outfile="test.png"
            infile="test2.png"       
        fullcmd="magick composite splatter-23660_640.png "+infile+" -geometry +"+str(x_pos)+"+"+str(y_pos)+" "+outfile
        print(fullcmd)
        returned_value=os.system(fullcmd)
        i+=1
