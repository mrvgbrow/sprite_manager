import matplotlib.pyplot as plt
import numpy as np
import time
import random
import sys

f=open('Scripts.txt','w')

if len(sys.argv) > 2:
    scores_per_sim=int(sys.argv[1])
    n_highscores=int(sys.argv[2])
else:
    scores_per_sim=1000
    n_highscores=-1    

n_sims=1000
reference_sim=-1

n_highscores_all=np.array([])
for nsim in range(n_sims):
    highscore_games=np.array([])
    highscore=-1
    for ngame in range(scores_per_sim):
        score=0
#        compval=random.randint(0,1)
        compval=1
        score=np.random.normal(0,1,1)
#        newscore=score
#        while newscore>1.0:
#            print(newscore)
#            newscore=np.random.normal(0,1,1)
#            score+=newscore
#        for j in range(10):
#            score=score+random.randint(1,10)
        while random.randint(0,1)==compval:
            score=score+1
        if nsim==reference_sim:
            f.write(str(score)+'\n')
        if score > highscore:
            highscore=score
            highscore_games=np.append(highscore_games,[ngame])
    n_highscores_all=np.append(n_highscores_all,[len(highscore_games)])
print(len(n_highscores_all[np.where(n_highscores_all>=n_highscores)]))
print(np.mean(n_highscores_all))
p=0
for ngame in range(scores_per_sim):
    p+=1/float(ngame+1)
print(p)
f.close()