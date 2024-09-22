"""
This module contains the implementation of the F(C+1)M algorithm, which is an extension of the Fuzzy C-Means algorithm with an extra noise cluster proposed by R. Dave in 1993.
"""

from pydantic import Field
from algofuzz.fcm.fcm import FCM

class FCPlus1M(FCM):
    """
    Partitions a numeric dataset using the F(C+1)M algorithm.
    """
    eta: float = Field(default=2.5)
    """
    The penalty factor for the noise cluster. The default value is 2.5.
    """

    def calculate_initial_sum(self):
        corrected_m = -2 / (self.m - 1)
        return self.eta ** corrected_m
