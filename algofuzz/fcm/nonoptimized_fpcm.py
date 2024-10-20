"""
This module contains the implementation of the Fuzzy-Possibilistic C-Means Clustering algorithm proposed by Pal, Pal and Bezdek in 1997.
"""

from algofuzz.fcm.nonoptimized_gfpcm import NonoptimizedGFPCM

class NonoptimizedFPCM(NonoptimizedGFPCM):
    w_prob: int = 1
