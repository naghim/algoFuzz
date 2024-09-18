from pydantic import Field
from algofuzz.fcm.fcm import FCM

class FCPlus1M(FCM):
    eta: float = Field(default=2.5)

    def calculate_initial_sum(self):
        corrected_m = -2 / (self.m - 1)
        return self.eta ** corrected_m
