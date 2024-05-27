from typing import List, Optional
import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dists import Distribution
from src.monte_carlo import hoeffding_bound
from src.metrics import social_welfare_relative, max_distance_to_hindsight, max_envy
from src.optimize import golden_section_search

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
    
@dataclass
class ExtraState:
    """
    The value a = (p, verbosity, allocations, demands), where
    - p is the probability you're on the current sample path
    - verbosity is the level of output
    - allocations is the list of allocations so far
    - demands is the list of demands so far
    - min_fill is the minimum fill rate so far
    - max_fill is the maximum fill rate so far
    """
    p: np.float64
    verbosity: int
    allocations: List[np.float64] = field(default_factory=list)
    demands: List[np.float64] = field(default_factory=list)
    min_fill: np.float64 = 1
    max_fill: np.float64 = 0

    def next_state(self, p_next, alloc, demand):
        """
        Update the state to the next state.
        """
        return ExtraState(self.p * p_next, self.verbosity, self.allocations + [alloc], self.demands + [demand], min(self.min_fill, alloc / demand), max(self.max_fill, alloc / demand))
    
    def silence(self):
        """
        Make is so that verbosity is set to 0, and calc is set to False.
        """
        return ExtraState(self.p, 0, self.allocations, self.demands, self.min_fill, self.max_fill)

def normalize_demands(demand_distributions):
    """
    Normalize total expected demand to 1.
    """
    total_demand = sum([dist.mean() for dist in demand_distributions])
    return [dist.scale(1 / total_demand) for dist in demand_distributions]

def empty_extra_state():
    return ExtraState(1, 0, [], [])

def ppa_allocation_batched(demands, supplies, expected_future_demand):
    return np.min([demands, supplies * (demands / (demands + expected_future_demand))], axis=0)

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
        verbosity=0,
        alloc_step=0.1,
        normalize_demand=False
    ):
        # total number of nodes
        self.N = len(demand_distributions)

        # make sure all demand support points positive
        for dist in demand_distributions:
            assert dist.min() >= 0

        # make sure there are demand distributions at all
        assert self.N > 0

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

        # schedule for lambdas
        self.lambda_schedule = [0 for _ in range(self.N)]

        # the social welfare function, takes in 2d arrays
        self.social_welfare = lambda allocs, demands: social_welfare_relative(alpha)(allocs[np.newaxis, :], demands[np.newaxis, :])

        # verbosity = 0, no output, = 1, log all except final allocation, >= 2, log all
        # this is the overall verbosity, we control each evaluation's verbosity with ExtraState
        self.verbosity = verbosity

        # set up the allocation and sequencing functions
        self.allocation_method_to_function = {
            "exact": self.optimal_allocation,
            "ppa": self.ppa_allocation,
            "lambda": self.lambda_allocation,
            "lambda_time": self.lambda_time_allocation,
            "greedy": self.greedy,
            "max_demand": self.max_demand,
            "lien": self.lien_allocation,
            # "hope_guardrail_12": self.hope_guardrail(1 / 2),
            # "hope_guardrail_13": self.hope_guardrail(1 / 3),
            "saffe": self.saffe_allocation
        }
        self.allocation_method = allocation_method
        self.allocation_function = self.allocation_method_to_function[allocation_method]

    def change_initial_supply(self, initial_supply):
        self.initial_supply = initial_supply

    def change_allocation_method(self, allocation_method):
        self.allocation_method = allocation_method
        self.allocation_function = self.allocation_method_to_function[allocation_method]
        self.lambda_schedule = [0 for _ in range(self.N)]

    def change_alpha(self, alpha):
        # the social welfare function
        self.social_welfare = self.social_welfare = lambda allocs, demands: social_welfare_relative(alpha)(allocs[np.newaxis, :], demands[np.newaxis, :])
        self.alpha = alpha

    def set_lambda_schedule(self, lambda_schedule):
        assert len(lambda_schedule) == self.N
        self.lambda_schedule = lambda_schedule

    def optimize_lambda(self, tol=1e-3, monte_carlo=False, batch=True, batch_size=1000, debug=False):
        """
        Optimize the lambda schedule.
        """
        self.change_allocation_method("lambda_time")

        def objective(l):
            self.set_lambda_schedule([l for _ in range(self.N)])
            # choose n to satisfy e = 0.01 with probability 0.01
            n = hoeffding_bound(tol, 0.01)
            if not monte_carlo:
                return self.solve()[0]
            if batch:
                return self.monte_carlo_performance_metrics_batched(n, batch_size=batch_size)["social_welfare"]
            return self.monte_carlo_performance_metrics(n)["social_welfare"]

        result = golden_section_search(objective, -2, 2, tol=tol, debug=debug)
        self.set_lambda_schedule([result for _ in range(self.N)])
        return result

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
            Z, w = self.social_welfare(np.array(extra.allocations + [alloc]), np.array(extra.demands + [state.d]))[0], max(state.c - state.d, 0)

            if self.verbosity >= 2 and extra.verbosity == 1:
                print(
                    f"At time {t} with d_t={state.d} and c_t={state.c}, allocate what's left with Z={round(Z, 4)} and waste={round(w, 4)}."
                )
        else:
            x = self.allocation_function(t, state, extra)
            Z, w = self.find_welfare_and_waste(t, state, x, extra)

            if self.verbosity >= 1 and extra.verbosity == 1:
                print(
                    f"At time {t} with d_t={state.d} and c_t={state.c}, we allocate x_t={x}."
                )

        return Z, w

    # @lru_cache(maxsize=4096)
    def optimal_allocation(self, t, state: State, extra: ExtraState):
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

        evaluator = self.find_welfare_and_waste

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

    def ppa_allocation(self, t: int, state: State, extra: ExtraState):
        expected_future_demand = sum(
            [self.demand_distributions[i].mean() for i in range(t, self.N)]
        )
        return np.min([state.d, state.c * (state.d / (state.d + expected_future_demand))], axis=0)
    
    def lien_allocation(self, t: int, state: State, extra: ExtraState):
        if t == self.N - 1:
            return np.min([state.d, state.c])

        this_median = self.demand_distributions[t].median()
        next_median = self.demand_distributions[t + 1].median()
        this_mean = self.demand_distributions[t].mean()
        next_mean = self.demand_distributions[t + 1].mean()
        next_stddev = self.demand_distributions[t + 1].stddev_val
        expected_future_demand = sum(
            [self.demand_distributions[i].mean() for i in range(t, self.N)]
        )
        delta = (this_median - next_median) / ((1 / 2) * (this_median + next_median))
        budget_portion = state.c * (this_mean + next_mean) / expected_future_demand
        heuristic_threshold = budget_portion * (state.d / (state.d + next_median + delta * np.sqrt(next_stddev)))
        return min(heuristic_threshold, extra.min_fill * state.d, state.c)

    def greedy(self, t: int, state: State, extra: ExtraState):
        # fulfill all demand, even if it exhausts us
        return np.min([state.d, state.c])

    def hope_guardrail(self, Lt=1 / 2):
        cthresh = 1
        cup = 1
        clow = 1

        remaining = self.N
        size_future = 1 + np.sum([dist.mean() for dist in self.demand_distributions[1:]])
        threshold_lower = (self.initial_supply / size_future) * (1 - (np.sqrt(cthresh * remaining * np.log(remaining)) / size_future) - (cthresh * np.log(remaining) / size_future))
        threshold_upper = threshold_lower + (cthresh / (remaining ** Lt))

        if threshold_lower < 0:
            raise ValueError("Threshold lower is negative.")

        def hope_guardrail_allocation(t: int, state: State, extra: ExtraState):
            # assume sizes are all equal to 1, since this concerns aggregate demand at each location
            remaining = self.N - t
            confidence_bound = np.sqrt(np.mean(self.demand_means) * remaining)

            if state.c < threshold_lower:
                return state.c
            if state.c >= threshold_lower * (np.sum(self.demand_means[t + 1:]) + clow * confidence_bound) + threshold_upper:
                return threshold_upper
            return threshold_lower
            
        return hope_guardrail_allocation
    
    def max_demand(self, t: int, state: State, extra: ExtraState):
        # like PPA, but for worst-case demand
        sum_max_demands = sum(
            [self.demand_distributions[i].max() for i in range(t, self.N)]
        )
        return np.min([state.d, state.c * (state.d / (state.d + sum_max_demands))], axis=0)

    def lambda_allocation(self, t: int, state: State, extra: ExtraState):
        expected_future_demand = sum(
            [self.demand_distributions[i].mean() + self.lambda_schedule[i] * self.demand_distributions[i].stddev_val for i in range(t, self.N)]
        )
        return np.min([state.d, state.c * (state.d / (state.d + expected_future_demand))], axis=0)

    def lambda_time_allocation(self, t: int, state: State, extra: ExtraState):
        # lambda is the first element of the lambda schedule
        l = self.lambda_schedule[0]
        expected_future_demand = sum(
            [self.demand_distributions[i].mean() + l * np.sqrt(self.N - t) * self.demand_distributions[i].stddev_val for i in range(t, self.N)]
        )
        return np.min([state.d, state.c * (state.d / (state.d + expected_future_demand))], axis=0)
    
    def saffe_allocation(self, t: int, state: State, extra: ExtraState):
        def give_to_agent(num_agents, supply, demands, past_allocations):
            # equivalent to Theorem 3 in Hassandezeh et al. 2023
            # with weights all equal to 1
            indices = np.argsort(np.array(demands) + np.array(past_allocations))
            j = 0
            allocations = np.zeros(num_agents)

            while j < num_agents and supply >= 0:
                threshold = sum([max(demands[indices[j]] + past_allocations[indices[j]] - past_allocations[indices[k]], 0) for k in range(j, num_agents)])
                if supply <= threshold:
                    mu = supply / (num_agents - j - 1) + past_allocations[indices[j]]
                    for k in range(j, num_agents):
                        allocations[indices[k]] = max(mu - past_allocations[indices[j]], 0)
                    break
                else:
                    allocations[indices[j]] = demands[indices[j]]
                    j += 1
        
            return allocations

        Y = [0 for _ in range(t - 1)] + [state.d] + self.demand_means[t:]
        allocations = give_to_agent(self.N, state.c, Y, extra.allocations + [0 for _ in range(self.N - (t - 1))])
        return allocations[t] * state.d / Y[t]
    
    def solve(self):
        """
        Solves the alpha-fair dynamic allocation problem.

        Returns the welfare and the waste as a tuple (Z, w)
        """
        return self.demand_distributions[0].expect_with_prob(
            lambda d, p: np.array(
                self.evaluate_allocation_policy(1, State(self.initial_supply, d), empty_extra_state())
            )
        )

    def monte_carlo_performance_metrics(self, n=1000):
        # draw n sample paths from each distribution and calculate
        # the performance metrics

        # the performance metrics are (in expectation):
        # - social welfare
        # - waste
        # - minimum fill rate
        # - maximum distance to offline allocation
        # - max_envy
        # - fill rate at each node

        # return a Pandas DataFrame with the performance metrics
        # with the mean and standard deviation of the mean, which
        # we assume to be normally distributed. 

        sum_social_welfare = 0
        sum_waste = 0
        sum_min_fill = 0
        sum_max_distance = 0
        sum_max_envy = 0

        for _ in range(n):
            state = State(self.initial_supply, self.demand_distributions[0].sample(1)[0])
            extra = empty_extra_state()

            # run the simulation
            for t in range(1, self.N):
                x = self.allocation_function(t, state, extra)
                extra = extra.next_state(1, x, state.d)
                state = state.next_state(x, self.demand_distributions[t].sample(1)[0])

            # final allocation for the last node
            x = min(state.c, state.d)
            extra = extra.next_state(1, x, state.d)

            # calculate the performance metrics
            allocations = np.array(extra.allocations)[np.newaxis, :]
            demands = np.array(extra.demands)[np.newaxis, :]

            social_welfare = self.social_welfare(np.array(extra.allocations), np.array(extra.demands))
            waste = max(state.c - state.d, 0)
            min_fill = extra.min_fill
            max_distance = max_distance_to_hindsight(allocations, demands, self.initial_supply)
            max_envy_value = max_envy(allocations, demands)

            sum_social_welfare += social_welfare
            sum_waste += waste
            sum_min_fill += min_fill
            sum_max_distance += max_distance
            sum_max_envy += max_envy_value

        return dict(
            social_welfare=sum_social_welfare / n,
            waste=sum_waste / n,
            min_fill=sum_min_fill / n,
            max_dist=sum_max_distance / n,
            max_envy=sum_max_envy / n
        )
    
    def monte_carlo_performance_metrics_batched(self, n: int, batch_size=1000):
        if self.allocation_method not in ["ppa", "lambda", "lambda_time"]:
            return self.monte_carlo_performance_metrics(n)

        sum_social_welfare = 0
        sum_waste = 0
        sum_min_fill = 0
        sum_max_distance = 0
        sum_max_envy = 0

        iters = n // batch_size + 1
        true_n = iters * batch_size

        for _ in range(iters):
            demands = self.demand_distributions[0].sample(batch_size)
            supplies = np.ones(batch_size) * self.initial_supply
            allocations = np.zeros((batch_size, self.N))
            all_demands = np.zeros((batch_size, self.N))
            min_fills = np.ones(batch_size)
            max_fills = np.zeros(batch_size)

            # run the simulation
            for t in range(1, self.N):
                # expected future demand
                expected_future_demand = np.sum(self.demand_means[t:]) + np.dot(self.lambda_schedule[t:], [dist.stddev_val for dist in self.demand_distributions[t:]])
                x = ppa_allocation_batched(demands, supplies, expected_future_demand)
                allocations[:, t - 1] = x
                all_demands[:, t - 1] = demands
                min_fills = np.minimum(min_fills, x / demands)
                max_fills = np.maximum(max_fills, x / demands)

                supplies = np.maximum(supplies - x, 0)
                demands = self.demand_distributions[t].sample(batch_size)

            # final allocation for the last node
            t = self.N - 1
            x = np.minimum(supplies, demands)
            allocations[:, t] = x
            all_demands[:, t] = demands
            min_fills = np.minimum(min_fills, x / demands)
            max_fills = np.maximum(max_fills, x / demands)

            # calculate the performance metrics
            social_welfare = social_welfare_relative(self.alpha)(allocations, all_demands).sum()
            waste = np.maximum(supplies - demands, 0).sum()
            min_fill = min_fills.sum()
            max_distance = max_distance_to_hindsight(allocations, all_demands, self.initial_supply).sum()
            max_envy_value = max_envy(allocations, all_demands).sum()

            sum_social_welfare += social_welfare
            sum_waste += waste
            sum_min_fill += min_fill
            sum_max_distance += max_distance
            sum_max_envy += max_envy_value

        return dict(
            social_welfare=sum_social_welfare / true_n,
            waste=sum_waste / true_n,
            min_fill=sum_min_fill / true_n,
            max_dist=sum_max_distance / true_n,
            max_envy=sum_max_envy / true_n
        )

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

    def provide_Z_vs_x_plot_data(self, t, state: State):
        """
        Returns the Z value for each possible allocation at some state.
        """
        x_values = np.arange(self.alloc_step, state.c + self.alloc_step, self.alloc_step)
        Z_values = np.zeros(x_values.size)

        evaluator = self.find_welfare_and_waste

        for i, x in enumerate(x_values):
            Z_values[i], _ = evaluator(
                t, state, x, empty_extra_state()
            )

        return x_values, Z_values

    def provide_x_vs_d_plot_data(self, t, state: State, step):
        """
        Provide the policy mapping from the demand to the allocation.
        """
        d_values = np.arange(step, state.d, step)
        x_values = np.zeros(d_values.size)

        for i, d in enumerate(d_values):
            alt_state = State(state.c, d)
            x_values[i] = self.allocation_function(t + 1, alt_state, empty_extra_state())

        return d_values, x_values
    
    def provide_Z_versus_lambda_plot_data(self, step):
        """
        Provide the Z value for each lambda value.
        """
        lambdas = np.arange(-2, 2, step)
        Z_values = np.zeros_like(lambdas)

        for i, l in enumerate(lambdas):
            # set the same across all nodes
            self.set_lambda_schedule([l for _ in range(self.N)])
            Z_values[i], _ = self.solve()

        return lambdas, Z_values
    
    def provide_Z_versus_lambda_s_plot_data(self, lambda_step, supply_step):
        """
        Provide the Z value for each lambda value and supply value.
        """
        lambda_values = np.arange(-2, 2, lambda_step)
        s_values = np.arange(supply_step, self.max_supply_needed(), supply_step)
        Z_values = np.zeros((lambda_values.size, s_values.size))

        for i, l in enumerate(lambda_values):
            for j, s in enumerate(s_values):
                self.set_lambda_schedule([l for _ in range(self.N)])
                self.change_initial_supply(s)
                Z_values[i, j], _ = self.solve()

        return lambda_values, s_values, Z_values
    
    def provide_Z_versus_three_node_lambda_plot_data(self, step):
        """
        Provide the Z value for each lambda value.
        """
        # for three-node problems, set lambda for nodes after the first
        # since the variability at the first node doesn't matter
        # as we get its realized demand
        assert self.N == 3

        lambdas = np.arange(-2, 2, step)
        Z_values = np.zeros((lambdas.size, lambdas.size))

        for i, l in enumerate(lambdas):
            for j, m in enumerate(lambdas):
                # set the same across all nodes
                self.set_lambda_schedule([0, l, m])
                Z_values[i, j], _ = self.solve()

        return lambdas, Z_values
    
    def copy(self):
        return AllocationSolver(
            self.demand_distributions,
            self.initial_supply,
            self.alpha,
            self.allocation_method,
            self.verbosity,
            self.alloc_step,
            False
        )

    def __str__(self):
        s = []
        for i, dist in enumerate(self.demand_distributions):
            s.append(f"Node {i}: {str(dist)}")
        return "\n".join(s)
