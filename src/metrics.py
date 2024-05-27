"""
Provide functions for calculating metrics.
"""
import math

import numpy as np

def social_welfare_relative(alpha):
    """
    Calculate the social welfare of the allocation.

    Assume allocs and demands columns equal to the number of agents.
    The rows are separate runs (batched).
    """
    if alpha == 1:
        def nsw_relative(allocs: np.ndarray, demands: np.ndarray):
            total_demand = demands.sum(axis=1)
            utilities = np.minimum(allocs / demands, np.ones_like(allocs))
            proportions = demands / total_demand[:, np.newaxis]
            return np.prod(np.power(utilities, proportions), axis=1)
        return nsw_relative
    elif alpha == np.inf:
        def ks_relative(allocs, demands):
            utilities = np.minimum(allocs / demands, np.ones_like(allocs))
            return np.min(utilities, axis=1)
        return ks_relative
    else:
        def alpha_relative(allocs, demands):
            total_demand = demands.sum(axis=1)
            utilities = np.minimum(allocs / demands, np.ones_like(allocs))
            demand_times_utility = demands * np.power(utilities, 1 - alpha)
            weighted_sum = np.sum(demand_times_utility, axis=1)
            return np.power((1 / total_demand * weighted_sum), 1 / (1 - alpha))
        return alpha_relative

def max_distance_to_hindsight(allocs: np.ndarray, demands: np.ndarray, supply):
    """
    Calculate the maximum distance to hindsight.
    """
    total_demand = demands.sum(axis=1)
    hindsight_allocs = np.minimum(supply * demands / total_demand[:, np.newaxis], demands)
    return np.max(np.abs(allocs - hindsight_allocs), axis=1)

def max_envy(allocs: np.ndarray, demands: np.ndarray):
    """
    Calculate the maximum envy.
    """
    fill_rates = np.minimum(allocs / demands, np.ones_like(allocs))
    return np.max(fill_rates, axis=1) - np.min(fill_rates, axis=1)