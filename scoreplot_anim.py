import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy
import time
import re
import sys
import os

def init():
    point.set_data([],[])
    point2.set_data([],[])
    point3.set_data([],[])
    point4.set_data([],[])
    point5.set_data([],[])
    text.set_text('')
    return point,point2,point3,point4,point5,text

def animate(i,games,scores,highscores_index,highscores):
    point.set_data([games[0:i]],[scores[0:i]])
    point2_index=numpy.where(highscores_index<i+2)
    point2.set_data([highscores_index[numpy.where(highscores_index<i+2)]],[highscores[numpy.where(highscores_index<i+2)]])

    # Animate the high score symbols, expanding stars
    anim1=1
    anim2=2
    anim3=3
    anim4=4
    hs_index=numpy.where((highscores_index<i-anim1) & (highscores_index>=i-anim2))
    point3.set_data([highscores_index[hs_index]],[highscores[hs_index]])
    hs_index=numpy.where((highscores_index<i-anim2) & (highscores_index>=i-anim3))
    point4.set_data([highscores_index[hs_index]],[highscores[hs_index]])
    hs_index=numpy.where((highscores_index<i-anim3) & (highscores_index>=i-anim4))
    point5.set_data([highscores_index[hs_index]],[highscores[hs_index]])

    allhighs=numpy.where(highscores_index<i+1)
    text.set_text(str(allhighs[0].size)+' High Scores')
    return point,point2,point3,point4,point5,text

def MaxSoFar(arr,i):
    if i == 0:
        return True
    if max(arr[:i]) < arr[i]:
        return True
    else:
        return False

# Number of game plays to average over in mean progression
average_interval=15

scorefile=open("C:/Users/sp4ce/OneDrive/Documents/"+sys.argv[1]+"/"+sys.argv[1]+".txt","r")
if scorefile.mode == 'r':
    contents = scorefile.readlines()

# Parse the file header giving information about the scoring
m1=re.match('Starting Lives: ([0-9]+)',contents[0])
if m1:
    lives=m1.groups()
    starting_lives=int(lives[0])
else:
    starting_lives=1
m2=re.match('Extra Lives: ',contents[1])
extra_lives=numpy.array([])
if m2:
    extras=re.findall('[0-9]+',contents[1])
    for i in range(0,len(extras)):
        extra_lives=numpy.append(extra_lives,int(extras[i]))

# Step through the file, storing scores and markers
i=0
score_index=numpy.array([])
scores=numpy.array([])
highscores_index=numpy.array([])
highscores=numpy.array([])
for line in contents:
    line=line.strip()

    # Add markers (vertical dashed lines) at important points
    m1=re.match('^\W+([\w\s]+)\W+$',line)
    if m1:
        yline=numpy.linspace(0,extra_lives[len(extra_lives)-1],100)
        xline=yline*0+i
#        plt.plot(xline,yline,'k--')

    if re.match('\s*[0-9]+',line):
        i=i+1
        score_index=numpy.append(score_index,[i])
        score=int(line)
        lives=starting_lives+sum(numpy.where(extra_lives < score,1,0))
        if len(sys.argv) > 2:
            score_per_life=float(score)
        else:
            score_per_life=float(score)/float(lives)
        scores=numpy.append(scores,[score_per_life])
        if MaxSoFar(scores,i-1):
            highscores=numpy.append(highscores,[scores[i-1]])
            highscores_index=numpy.append(highscores_index,[i])

# Compute the mean progression
i=0
averages_index=numpy.array([])
score_averages=numpy.array([])
for score in scores:
    i=i+1
    if i%average_interval==int(average_interval/2) and i<len(scores)-int(average_interval/2):
        averages_index=numpy.append(averages_index,[i])
        score_average=numpy.median(scores[i-int(average_interval/2):i+int(average_interval/2)])
        score_averages=numpy.append(score_averages,[score_average])

# Plot the results
fig=plt.figure()
imwriter=anim.ImageMagickFileWriter(fps=10)
imwriter.setup(fig,sys.argv[1]+'.gif',dpi=100)
ax=plt.axes(xlim=(0,max(score_index)),ylim=(0,1.1*max(scores)))
point, = ax.plot([],[],'ko')
point2, = ax.plot([],[],'b*',markersize=12)
point3, = ax.plot([],[],'b*',markersize=15,fillstyle='none')
point4, = ax.plot([],[],'c*',markersize=18,fillstyle='none')
point5, = ax.plot([],[],'c*',markersize=21,fillstyle='none')
text=ax.text(max(score_index)*0.15,max(scores)*0.9,'',fontsize=12)
xlabel=ax.set_xlabel('Game #')
ylabel=ax.set_ylabel('Score')
title=ax.set_title(sys.argv[1]+' Scores')

#figanim=anim.FuncAnimation(fig,animate,fargs=(score_index,scores,highscores_index,highscores,imwriter),init_func=init,
#                                          frames=len(scores),interval=7,blit=True,
#                                          repeat=True)

#plt.show()
interval=2
k=0
for j in range(len(scores)):
    
    animate(j,score_index,scores,highscores_index,highscores)
    if j % interval==0:
        imwriter.grab_frame()
        k+=1
imwriter.finish()

scorefile.close()

animate_cmd="magick convert "+sys.argv[1]+".gif ( -clone "+format(k-1)+" -set delay 1000 ) "+sys.argv[1]+"_2.gif"
print(animate_cmd)
returned_value=os.system(animate_cmd)
