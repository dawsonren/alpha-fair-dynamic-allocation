from typing import List, Optional
import math
from dataclasses import dataclass, field

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
    The value a = (p, verbosity, allocations, demands), where
    - p is the probability you're on the current sample path
    - verbosity is the level of output
    - allocations is the list of allocations so far
    - demands is the list of demands so far
    """
    p: np.float64
    verbosity: int
    allocations: List[np.float64] = field(default_factory=list)
    demands: List[np.float64] = field(default_factory=list)

    def next_state(self, p_next, alloc, demand):
        """
        Update the state to the next state.
        """
        return ExtraState(self.p * p_next, self.verbosity, self.allocations + [alloc], self.demands + [demand])
    
    def silence(self):
        """
        Make is so that verbosity is set to 0, and calc is set to False.
        """
        return ExtraState(self.p, 0, self.allocations, self.demands)

def normalize_demands(demand_distributions):
    """
    Normalize total expected demand to 1.
    """
    total_demand = sum([dist.mean() for dist in demand_distributions])
    return [dist.scale(1 / total_demand) for dist in demand_distributions]

def social_welfare_relative(alpha):
    """
    Calculate the social welfare of the allocation.
    """
    if alpha == 1:
        def nsw_relative(allocs, demands):
            total_demand = sum(demands)
            utilities = [min(alloc / demand, 1) if demand > 0 else 1 for alloc, demand in zip(allocs, demands)]
            proportions = [demand / total_demand for demand in demands]
            return math.prod([u ** p for u, p in zip(utilities, proportions)])
        return nsw_relative
    elif alpha == np.inf:
        def ks_relative(allocs, demands):
            utilities = [min(alloc / demand, 1) if demand > 0 else 1 for alloc, demand in zip(allocs, demands)]
            return min(utilities)
        return ks_relative
    else:
        def alpha_relative(allocs, demands):
            total_demand = sum(demands)
            utilities = [min(alloc / demand, 1) if demand > 0 else 1 for alloc, demand in zip(allocs, demands)]
            return ((1 / total_demand) * sum([demand * u ** (1 - alpha) for u, demand in zip(utilities, demands)])) ** (1 / (1 - alpha))
        return alpha_relative

def social_welfare_absolute(alpha):
    """
    Calculate the social welfare of the allocation.
    """
    if alpha == 1:
        def nsw_absolute(allocs, demands):
            utilities = [min(alloc / demand, 1) if demand > 0 else 1 for alloc, demand in zip(allocs, demands)]
            n = len(allocs)
            return math.prod([u ** (1 / n) for u in utilities])
        return nsw_absolute
    elif alpha == np.inf:
        def ks_absolute(allocs, demands):
            utilities = [min(alloc / demand, 1) if demand > 0 else 1 for alloc, demand in zip(allocs, demands)]
            return min(utilities)
        return ks_absolute
    else:
        def alpha_absolute(allocs, demands):
            n = len(allocs)
            utilities = [min(alloc / demand, 1) if demand > 0 else 1 for alloc, demand in zip(allocs, demands)]
            return sum([(1 / n) * u ** (1 - alpha) for u in utilities]) ** (1 / (1 - alpha))
        return alpha_absolute

def empty_extra_state():
    return ExtraState(1, 0, [], [])

class AllocationSolver:
    """
    This class represents the alpha-fair dynamic allocation problem.

    At time i, the we arrive at the ith node.
    """

    def __init__(
        self,
        demand_distributions: List[Distribution],
        initial_supply,
        alpha=np.inf,
        allocation_method="exact",
        equity="relative",
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

        # the inequity aversion parameter alpha
        self.alpha = alpha

        # the equity criterion
        if equity not in ["absolute", "relative"]:
            raise ValueError("Equity must be either absolute or relative.")
        
        # the social welfare function
        self.social_welfare = social_welfare_absolute(alpha) if equity == "absolute" else social_welfare_relative(alpha)

        # verbosity = 0, no output, = 1, log all except final allocation, >= 2, log all
        # this is the overall verbosity, we control each evaluation's verbosity with ExtraState
        self.verbosity = verbosity

        # set up the allocation and sequencing functions
        self.allocation_method_to_function = {
            "exact": self.optimal_allocation,
            "ppa": self.ppa_allocation
        }
        self.allocation_method = allocation_method
        self.allocation_function = self.allocation_method_to_function[allocation_method]

    def change_initial_supply(self, initial_supply):
        self.initial_supply = initial_supply

    def change_allocation_method(self, allocation_method):
        self.allocation_method = allocation_method
        self.allocation_function = self.allocation_method_to_function[allocation_method]

    def change_alpha(self, alpha, equity="relative"):
        # the social welfare function
        self.social_welfare = social_welfare_absolute(alpha) if equity == "absolute" else social_welfare_relative(alpha)

    def max_supply_needed(self):
        return sum([dist.max() for dist in self.demand_distributions])

    def find_welfare_and_waste(self, t, state: State, x, extra: ExtraState):
        return self.demand_distributions[t].expect_with_prob(
            lambda d_next, p_next: np.array(
                self.evaluate_allocation_policy(
                    t + 1, state.next_state(x, d_next), extra.next_state(p_next, x, state.d)
                )
            )
        )
    
    def find_welfare_and_waste_ex_ante(self, t, state: State, x, extra: ExtraState):
        Z, w = self.evaluate_allocation_policy_ex_ante(t + 1, state.next_state_ex_ante(x), extra)
        return Z, w

    def evaluate_allocation_policy_ex_ante(self, t, state: State, extra: ExtraState):
        def realized_demand_find_welfare_and_waste(d_i, p_i):
            if t == self.N:
                # if we're at the last node
                alloc = min(state.c, d_i)
                Z, w = self.social_welfare(extra.allocations + [alloc], extra.demands + [d_i]), max(state.c - d_i, 0)
                if self.verbosity >= 2 and extra.verbosity == 1:
                    print(
                        f"At time {t} with d_t={d_i} and c_t={state.c}, allocate what's left with Z={round(Z, 4)} and waste={round(w, 4)}."
                    )

                return Z, w
            
            x = self.allocation_function(t, state.update_demand_ex_ante(d_i), extra, ex_ante=True)
            Z, w = self.evaluate_allocation_policy_ex_ante(
                t + 1, state.next_state_ex_ante(x), extra.next_state(p_i, x, d_i)
            )

            if self.verbosity >= 1 and extra.verbosity == 1:
                print(
                    f"At time {t} with d_t={d_i} and c_t={state.c}, we allocate x_t={x}."
                )
            
            return Z, w
            
        return self.demand_distributions[t - 1].expect_with_prob(
            lambda d_next, p_next: np.array(
                realized_demand_find_welfare_and_waste(d_next, p_next)
            )
        )

    def evaluate_allocation_policy(self, t, state: State, extra: ExtraState):
        """
        Inputs:
        - t - the current time
        - State - the current state (c_t, d_t)
        - ExtraState - the extra state (p, beta_min, verbosity, calc)
        """
        if t == self.N:
            # if we're at the last node
            alloc = min(state.c, state.d)
            Z, w = self.social_welfare(extra.allocations + [alloc], extra.demands + [state.d]), max(state.c - state.d, 0)

            if self.verbosity >= 2 and extra.verbosity == 1:
                print(
                    f"At time {t} with d_t={state.d} and c_t={state.c}, allocate what's left with Z={round(Z, 4)} and waste={round(w, 4)}."
                )
        else:
            x = self.allocation_function(t, state, extra, ex_ante=False)
            Z, w = self.find_welfare_and_waste(t, state, x, extra)

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
        - state - the current state (c_t, d_t)

        Returns:
        - x_t, the optimal allocation at time t
        """
        # the range of x values should be the minimum of d and c
        # subtract machine epsilon to avoid giving all capacity away
        # we only would do this when we're at the last node, which
        # is already handled by another function
        x_values = np.arange(self.alloc_step,
                             min(state.d + 3 * self.alloc_step / 2, state.c - np.finfo(np.float32).eps),
                             self.alloc_step)
        best_z = 0
        best_x = 0

        evaluator = self.find_welfare_and_waste_ex_ante if ex_ante else self.find_welfare_and_waste

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
    
    def solve(self, ex_ante=False):
        """
        Solves the alpha-fair dynamic allocation problem.

        Returns the welfare and the waste as a tuple (Z, w)
        """
        if not ex_ante:
            return self.demand_distributions[0].expect_with_prob(
                lambda d, p: np.array(
                    self.evaluate_allocation_policy(1, State(self.initial_supply, d), ExtraState(p, 1, [], []))
                )
            )
        else:
            # solve
            return self.evaluate_allocation_policy_ex_ante(1, State(self.initial_supply, None), ExtraState(1, self.verbosity, [], []))
    
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
                x_t = self.allocation_function(t, State(c_t, d_t), ExtraState(p, 0, [], []), ex_ante=ex_ante)

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

    def provide_Z_vs_x_plot_data(self, t, state: State, ex_ante=False):
        """
        Returns the Z value for each possible allocation at some state.
        """
        x_values = np.arange(self.alloc_step, state.c + self.alloc_step, self.alloc_step)
        Z_values = np.zeros(x_values.size)

        evaluator = self.find_welfare_and_waste_ex_ante if ex_ante else self.find_welfare_and_waste

        for i, x in enumerate(x_values):
            Z_values[i], _ = evaluator(
                t, state, x, empty_extra_state()
            )

        return x_values, Z_values

    def provide_x_vs_d_plot_data(self, t, state: State, step, ex_ante=False):
        """
        Provide the policy mapping from the demand to the allocation.
        """
        d_values = np.arange(step, state.d, step)
        x_values = np.zeros(d_values.size)

        for i, d in enumerate(d_values):
            alt_state = State(state.c, d)
            x_values[i] = self.allocation_function(t + 1, alt_state, empty_extra_state(), ex_ante)

        return d_values, x_values

    def __str__(self):
        s = []
        for i, dist in enumerate(self.demand_distributions):
            s.append(f"Node {i}: {str(dist)}")
        return "\n".join(s)
