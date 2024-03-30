"""
Provide Monte Carlo estimation helpers.

Dawson Ren
January 6th, 2022
"""
import math


def hoeffding_bound(e: float, alpha: float) -> int:
    # provides the number of samples needed in MC simulation to give error
    # +/- e*h (h is supremum of outputs of the estimation) with probability 1 - alpha.
    return int((math.log(2 / alpha)) / (2 * e**2)) + 1
