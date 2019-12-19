import matplotlib.pyplot as plt
import numpy as np
import time

#test
mu=0.0
sigma=1
scores_per_sim=100000
n_sims=1000

s=np.random.normal(mu,sigma,scores_per_sim*n_sims)

sim=0
index=0
highscore=-10000
highscore_games=np.array([])
highscore_index=0
n_highscores=np.array([])
for rnum in s:
    if index > scores_per_sim-1:
        sim+=1
        index=0
        highscore=-10000
        highscore_index=0
    if rnum > highscore:
#        print(str(rnum)+' '+str(index))
        highscore=rnum
        if len(highscore_games)<highscore_index+1:
            highscore_games=np.append(highscore_games,index)
            n_highscores=np.append(n_highscores,1)
        else:
            highscore_games[highscore_index]+=index
            n_highscores[highscore_index]+=1
        highscore_index+=1      
    index+=1

i=0
for game in highscore_games:
    print(str(float(game)/float(n_highscores[i]))+' '+str(game)+' '+str(n_highscores[i]))
    i+=1
