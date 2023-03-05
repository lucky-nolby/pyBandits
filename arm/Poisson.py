# -*- coding: utf-8 -*-
'''Poisson distributed arm.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.6 $"

from scipy.stats import poisson
from math import isinf,exp

from arm.Arm import Arm

class Poisson(Arm):
	"""Poisson distributed arm, possibly truncated."""
	def __init__(self, p, trunc = float('inf')):
		self.p = p
		self.trunc = trunc
		if isinf(trunc):
			self.expectation = p
		else:
			q = exp(-p)
			sq = q
			self.expectation = 0
			for k in range(1, self.trunc):
				q = q * p / k
				self.expectation += k * q
				sq += q
			self.expectation += self.trunc * (1-sq)
				
	def draw(self):
		return min(poisson.rvs(self.p), self.trunc)
