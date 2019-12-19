import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy
import time
import re
import sys


# Number of game plays to average over in mean progression
average_interval=20

scorefile=open("C:/Users/sp4ce/OneDrive/Documents/"+sys.argv[1]+"/"+sys.argv[1]+".txt","r")
if scorefile.mode == 'r':
    contents = scorefile.readlines()

# Parse the file header giving information about the scoring
m1=re.match('Starting Lives: ([0-9]+)',contents[0])
if m1:
    lives=m1.groups()
    starting_lives=int(lives[0])
m2=re.match('Extra Lives: ',contents[1])
if m2:
    extras=re.findall('[0-9]+',contents[1])
    extra_lives=numpy.array([])
    for i in range(0,len(extras)):
        extra_lives=numpy.append(extra_lives,int(extras[i]))

# Step through the file, storing scores and markers
i=0
score_index=numpy.array([])
scores=numpy.array([])
for line in contents:
    line=line.strip()

    # Add markers (vertical dashed lines) at important points
    m1=re.match('^\W+([\w\s]+)\W+$',line)
    if m1:
        yline=numpy.linspace(0,extra_lives[len(extra_lives)-1],100)
        xline=yline*0+i
        print(line,i)
#        plt.plot(xline,yline,'k--')

    if re.match('\s*[0-9]+',line):
        i=i+1
        score_index=numpy.append(score_index,[i])
        score=int(line)
        lives=starting_lives+sum(numpy.where(extra_lives < score,1,0))
        if len(sys.argv) > 2 and sys.argv[2]=='score':
            score_per_life=float(score)
        else:
            score_per_life=float(score)/float(lives)
        scores=numpy.append(scores,[score_per_life])

# Compute the mean progression
i=0
averages_index=numpy.array([])
score_averages=numpy.array([])
for score in scores:
    i=i+1
    if i%average_interval==int(average_interval/2) and i<=len(scores)-int(average_interval/2):
        averages_index=numpy.append(averages_index,[i])
        score_average=numpy.mean(scores[i-int(average_interval/2):i+int(average_interval/2)])
        score_averages=numpy.append(score_averages,[score_average])

print(i)
# Plot the results

plt.plot(averages_index,score_averages,'r')
plt.plot(score_index,scores,'ko')
if sys.argv[2] == 'score':
    plt.ylabel('Score')
else:
    plt.ylabel('Score per Life')
plt.xlabel('Play #')
plt.ylim(bottom=0)
plt.show()

scorefile.close()
