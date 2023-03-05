# -*- coding: utf-8 -*-
'''
Environement for a Multi-armed bandit problem 
with arms given in the 'arms' list 
'''

__author__ = "Olivier Cappé,Aurélien Garivier"
__version__ = "$Revision: 1.5 $"

from Result import *
from environment.Environment import Environment

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""
    
    def __init__(self, arms):
        self.arms = arms
        self.nbArms = len(arms)
        # supposed to have a property nbArms

    def play(self, policy, horizon):
        policy.startGame()
        result = Result(self.nbArms, horizon)
        for t in range(horizon):
            choice = policy.choice()
            reward = self.arms[choice].draw()
            # reward=0
            print(f'reward:{reward}')
            policy.getReward(choice, reward)
            result.store(t, choice, reward)
        return result


class MAB_new(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, arms):
        self.arms = arms
        # self.nbArms = len(arms)
        # self.nbArms=arms.shape()
        self.nbArms=len(arms)
        # supposed to have a property nbArms

    def play(self, policy, horizon):
        policy.startGame()
        result = Result(self.nbArms, horizon)
        for t in range(horizon):
            choice = policy.choice()
            reward = self.arms[choice].draw(t)
            # print(f'reward:{reward}')
            # reward=0
            policy.getReward(choice, reward)
            result.store(t, choice, reward)
        return result
