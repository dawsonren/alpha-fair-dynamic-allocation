from typing import List, Tuple, Optional
from itertools import permutations
from functools import lru_cache
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from src.dists import Distribution

@dataclass
class State:
    """
    The value u = (c, d), where
    - c is the supply available at time t
    - d is the demand at time t
    """
    c: np.float64
    d: Optional[np.float64]

    def next_state(self, x, d_next):
        """
        Update the state to the next state for ex-post.
        """
        return State(self.c - x, d_next)
    
    def next_state_ex_ante(self, x):
        """
        Update the state to the next state for ex-ante.
        """
        return State(self.c - x, self.d)
    
    def update_demand_ex_ante(self, d_next):
        """
        Update the state to the next state for ex-ante.
        """
        return State(self.c, d_next)
    
@dataclass
class ExtraState:
    """
    The value a = (p, beta_min, verbosity, calc), where
    - p is the probability you're on the current sample path
    - beta_min is the minimum fill rate you've seen so far
    - verbosity is the level of output
    - calc is whether to calculate Manshadi's objectives

    These aren't states for our DP/MDP, but are useful for passing around
    because they stay the same throughout the recursion - that is, if calc
    is true at time 0, then it's true for all future times.

    For example, verbosity is useful for controlling the level of output
    on a specific evaluation - when we're looking for optimal policies,
    we don't want to print everything, but when we're evaluating a policy,
    we do.
    """
    p: np.float64
    beta_min: np.float64
    verbosity: int
    calc: bool

    def next_state(self, p_next, fill_rate):
        """
        Update the state to the next state.
        """
        return ExtraState(self.p * p_next, min(fill_rate, self.beta_min), self.verbosity, self.calc)
    
    def silence(self):
        """
        Make is so that verbosity is set to 0, and calc is set to False.
        """
        return ExtraState(self.p, self.beta_min, 0, False)

def normalize_demands(demand_distributions):
    """
    Normalize total expected demand to 1.
    """
    total_demand = sum([dist.mean() for dist in demand_distributions])
    return [dist.scale(1 / total_demand) for dist in demand_distributions]

class AllocationSolver:
    """
    This class represents the max-min fill rate allocation problem.

    At time i, the we arrive at the ith node.
    """

    def __init__(
        self,
        demand_distributions: List[Distribution],
        initial_supply,
        allocation_method="exact",
        tfr=None,
        verbosity=0,
        alloc_step=0.1,
        normalize_demand=False
    ):
        # total number of nodes
        self.N = len(demand_distributions)

        # prepare demand distributions
        self.demand_distributions = (
            demand_distributions
            if not normalize_demand
            else normalize_demands(demand_distributions)
        )
        self.demand_means = [dist.mean() for dist in self.demand_distributions]
        
        # the amount of preliminary supply, s_0
        self.initial_supply = initial_supply

        # discretizations, the increment of optimal allocation
        self.alloc_step = alloc_step

        # verbosity = 0, no output, = 1, log all except final allocation, >= 2, log all
        # this is the overall verbosity, we control each evaluation's verbosity with ExtraState
        self.verbosity = verbosity

        # set up the allocation and sequencing functions
        if tfr is None and allocation_method == "tfr":
            raise ValueError("Must specify target fill rate.")

        self.allocation_method_to_function = {
            "exact": self.optimal_allocation,
            "ppa": self.ppa_allocation,
            "tfr": self.tfr_allocation(tfr=tfr),
        }
        self.allocation_method = allocation_method
        self.allocation_function = self.allocation_method_to_function[allocation_method]

        # keeps track of expected fill rate at each node
        # at the end, take the minimum to get their ex-ante objective
        self.manshadi_ex_ante = np.zeros(self.N)
        # keeps track of expected fill rate across sample paths
        # use the parameter p to update, this contains the probability of being on a given sample path
        self.manshadi_ex_post = 0

    def change_initial_supply(self, initial_supply):
        self.initial_supply = initial_supply

    def max_supply_needed(self):
        return sum([dist.max() for dist in self.demand_distributions])

    def find_min_fill_rate_and_waste(self, t, state: State, x, extra: ExtraState):
        def realized_demand_find_min_fill_rate_and_waste(d_next, p_next):
            fill_rate = min(x / state.d, 1)
            Z, w = self.evaluate_allocation_policy(
                t + 1, state.next_state(x, d_next), extra.next_state(p_next, fill_rate)
            )
            return min(Z, fill_rate), w

        return self.demand_distributions[t].expect_with_prob(
            lambda d_next, p_next: np.array(
                realized_demand_find_min_fill_rate_and_waste(d_next, p_next)
            )
        )
    
    def find_min_fill_rate_and_waste_ex_ante(self, t, state: State, x, extra: ExtraState):
        Z, w = self.evaluate_allocation_policy_ex_ante(t + 1, state.next_state_ex_ante(x), extra)
        fill_rate = x / state.d
        return min(fill_rate, Z), w

    def evaluate_allocation_policy_ex_ante(self, t, state: State, extra: ExtraState):
        def realized_demand_find_min_fill_rate_and_waste(d_i, p_i):
            if t == self.N:
                Z, w = min(1, state.c / d_i), max(state.c - d_i, 0)
                if self.verbosity >= 2 and extra.verbosity == 1:
                    print(
                        f"At time {t} with d_t={d_i} and c_t={state.c}, allocate what's left with Z={round(Z, 2)} and waste={round(w, 2)}."
                    )
                
                if extra.calc:
                    # recall t is one-indexed
                    self.manshadi_ex_ante[t - 1] += extra.p * p_i * Z
                    self.manshadi_ex_post += extra.p * p_i * min(Z, extra.beta_min)

                return Z, w
            
            x = self.allocation_function(t, state.update_demand_ex_ante(d_i), extra, ex_ante=True)
            fill_rate = x / d_i
            Z, w = self.evaluate_allocation_policy_ex_ante(
                t + 1, state.next_state_ex_ante(x), extra.next_state(p_i, fill_rate)
            )

            if self.verbosity >= 1 and extra.verbosity == 1:
                print(
                    f"At time {t} with d_t={d_i} and c_t={state.c}, we allocate x_t={x}."
                )

            if extra.calc:
                self.manshadi_ex_ante[t - 1] += extra.p * p_i * fill_rate
            
            return min(Z, fill_rate), w
            
        return self.demand_distributions[t - 1].expect_with_prob(
            lambda d_next, p_next: np.array(
                realized_demand_find_min_fill_rate_and_waste(d_next, p_next)
            )
        )

    def evaluate_allocation_policy(self, t, state: State, extra: ExtraState):
        """
        Inputs:
        - t - the current time
        - i - the current node
        - c_t - the supply available to us at time t
        - S_t - the set of nodes we have yet to visit
        - d_t - the demand at time t
        - p - the probability of being on the current sample path (from time 1 to t)
        """
        if t == self.N:
            # if we're at the last node
            Z, w = min(1, state.c / state.d), max(state.c - state.d, 0)

            if extra.calc:
                self.manshadi_ex_ante[t - 1] += extra.p * Z
                self.manshadi_ex_post += extra.p * min(Z, extra.beta_min)

            if self.verbosity >= 2 and extra.verbosity == 1:
                print(
                    f"At time {t} with d_t={state.d} and c_t={state.c}, allocate what's left with Z={round(Z, 2)} and waste={round(w, 2)}."
                )
        else:
            x = self.allocation_function(t, state, extra, ex_ante=False)
            Z, w = self.find_min_fill_rate_and_waste(t, state, x, extra)
            fill_rate = x / state.d

            if extra.calc:
                self.manshadi_ex_ante[t - 1] += extra.p * fill_rate

            if self.verbosity >= 1 and extra.verbosity == 1:
                print(
                    f"At time {t} with d_t={state.d} and c_t={state.c}, we allocate x_t={x}."
                )

        return Z, w

    # @lru_cache(maxsize=4096)
    def optimal_allocation(self, t, state: State, extra: ExtraState, ex_ante=False):
        """
        The brute-force optimal allocation for nodes i to the end.

        Given:
        - state - the current state (c_t, i, d_t, S_t)

        Returns:
        - x_t, the optimal allocation at time t
        """
        # discretize to beta level
        x_values = np.arange(0, state.c + self.alloc_step / 2, self.alloc_step)
        best_z = 0
        best_x = 0

        evaluator = self.find_min_fill_rate_and_waste_ex_ante if ex_ante else self.find_min_fill_rate_and_waste

        # search through all possible allocations
        for x in x_values:
            # don't print for all x
            new_z, _ = evaluator(
                t, state, x, extra.silence()
            )
            if new_z >= best_z:
                best_z = new_z
                best_x = x

        return best_x

    def ppa_allocation(self, t, state: State, extra: ExtraState, ex_ante=False):
        expected_future_demand = sum(
            [self.demand_distributions[i].mean() for i in range(t, self.N)]
        )
        return np.min([state.d, state.c * (state.d / (state.d + expected_future_demand))], axis=0)

    def tfr_allocation(self, tfr):
        def alloc(t, state: State, extra: ExtraState, ex_ante=False):
            return min(tfr * state.d, state.c)

        return alloc
    
    def solve(self, ex_ante=False) -> np.ndarray[np.float64, np.float64]:
        """
        Solves the max-min fill rate problem.

        Returns the max-min fill rate and the waste as a tuple (Z, w)
        """
        # reset manshadi objectives
        self.manshadi_ex_ante = np.zeros(self.N)
        self.manshadi_ex_post = 0

        if not ex_ante:
            return self.demand_distributions[0].expect_with_prob(
                lambda d, p: np.array(
                    self.evaluate_allocation_policy(1, State(self.initial_supply, d), ExtraState(p, 1, 1, True))
                )
            )
        else:
            # solve
            return self.evaluate_allocation_policy_ex_ante(1, State(self.initial_supply, None), ExtraState(1, 1, 1, True))
    
    def solve_all(self):
        """
        Solve for 
        - optimal ex-ante objective
        - optimal ex-post objective
        - waste under optimal ex-ante policy
        - waste under optimal ex-post policy
        """
        if self.verbosity >= 1:
            print("Solving ex-ante...")
        Z_a, w_a = self.solve(ex_ante=True)
        if self.verbosity >= 1:
            print("\nSolving ex-post...")
        Z_p, w_p = self.solve(ex_ante=False)

        return Z_a, Z_p, w_a, w_p

    def monte_carlo_estimate_fill_rate_at_nodes(self, n: int, ex_ante=False):
        # draw n sample paths from each distribution and calculate
        # the expected fill rate at each node
        fill_rate_at_node = np.zeros(self.N)

        for _ in range(n):
            c_t = self.initial_supply
            t = 1
            p = 1

            while t < self.N:
                # draw demand
                d_t, p_t = self.demand_distributions[t - 1].sample_with_prob(1)
                d_t = d_t[0]
                p_t = p_t[0]
                p *= p_t
                x_t = self.allocation_function(t, State(c_t, d_t), ExtraState(p, 1, 0, False), ex_ante=ex_ante)

                # index by the node index, not index on the path
                fill_rate_at_node[t - 1] += min(x_t / d_t, 1)

                # update state
                c_t -= x_t
                t += 1

            # update the last node
            d_t, _ = self.demand_distributions[t - 1].sample_with_prob(1)
            d_t = d_t[0]
            fill_rate_at_node[t - 1] += min(c_t / d_t, 1)

        return fill_rate_at_node / n

    def plot_demand_distributions(self):
        fig, ax = plt.subplots()
        for i, dist in enumerate(self.demand_distributions):
            ax.plot(dist.xk, dist.pk, label=f"Node {i}")
        ax.legend()
        plt.xlabel("Demand")
        plt.ylabel("Probability")
        plt.title("Demand Distributions")
        # set range of x-axis
        plt.xlim(0, max([dist.xk[-1] for dist in self.demand_distributions]))
        # set range of y-axis
        plt.ylim(0, 1)
        plt.show()

    def __str__(self) -> str:
        s = []
        for i, dist in enumerate(self.demand_distributions):
            s.append(f"Node {i}: {str(dist)}")
        return "\n".join(s)
