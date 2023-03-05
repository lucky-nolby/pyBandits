# -*- coding: utf-8 -*-
'''A utility class for evaluating the performance of a policy in multi-armed bandit problems.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.10 $"

import random

import numpy as np
#from translate.misc.progressbar import ProgressBar

class Evaluation:
  
    def __init__(self, env, pol, nbRepetitions, horizon, tsav=[]):
        if len(tsav)>0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)
        self.env = env
        self.pol = pol
        self.nbRepetitions = nbRepetitions
        self.horizon = horizon
        self.nbArms = env.nbArms
        self.nbPulls = np.zeros((self.nbRepetitions, self.nbArms))
        self.cumReward = np.zeros((self.nbRepetitions, len(tsav)))
                 
        # progress = ProgressBar()
        for k in range(nbRepetitions): # progress(range(nbRepetitions)):
            if nbRepetitions < 10 or k % (nbRepetitions/10)==0:
                print(k)
            result = env.play(pol, horizon)
            self.nbPulls[k,:] = result.getNbPulls()
            # print(f'result:{result.rewards.shape}')
            # print(f'type(result):{result.rewards[0]}')
            # print(f'type:{np.cumsum(result.rewards)[0]}')
            self.cumReward[k,:] = np.cumsum(result.rewards)[tsav]
        # progress.finish()
     
    def meanReward(self):
        return sum(self.cumReward[:,-1])/len(self.cumReward[:,-1])

    def meanNbDraws(self):
        return np.mean(self.nbPulls ,0) 

    def meanRegret(self):
        print(f'max.expectation:{max([arm.expectation for arm in self.env.arms])}')
        return (1+self.tsav)*max([arm.expectation for arm in self.env.arms]) - np.mean(self.cumReward, 0)

    # def meanRegret(self):
    #     # print(f'cumreward:{self.cumReward}')
    #     return (1+self.tsav)*10-np.mean(self.cumReward,0)
