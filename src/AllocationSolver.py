from typing import List, Optional
import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from src.dists import Distribution
from src.monte_carlo import hoeffding_bound
from src.metrics import social_welfare_relative, max_distance_to_hindsight, max_envy, calculate_fill_rates, confidence_interval
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

def greedy_allocation_batched(demands, supplies):
    return np.min([demands, supplies], axis=0)

def guarantee_allocation_batched(demands, supplies, sum_guarantees):
    return np.min([demands, supplies * (demands / (demands + sum_guarantees))], axis=0)

def hope_online_batched(demands, supplies, demand_means):
    # def hope_online(budget, size, mean, stdev):
    allocation = np.zeros(len(demand_means))
    budget_remaining = supplies
    
    size_future = 1 + np.sum(demand_means[1:])
    
    for i in range(len(allocation)):
        size_future = 1 + np.sum(demand_means[i + 1:])
        allocation[i] = min(budget_remaining, budget_remaining / size_future)
        budget_remaining -= allocation[i]
    
    return allocation

def lambda_bound(demand_distributions: List[Distribution], lambda_time=False):
    """
    Only consider lambdas which keep the projected demand at each node
    positive. For lambda_time, make sure that we'd keep all allocations
    positive for all possible demand sequences.

    That is, make sure that for all nodes i, we have
    E[d_i] + \sqrt{N - i} \lambda \sigma_i > 0
    (omit the square root for lambda)
    """
    N = len(demand_distributions)
    bound = -2
    for i in range(N - 1):
        expected_future_demand = sum([dist.mean() for dist in demand_distributions[i + 1:]])
        sum_expected_variances = sum([
            dist.stddev_val * np.sqrt(N - j) if lambda_time else dist.stddev_val 
            for j, dist in enumerate(demand_distributions[i + 1:])
        ])
        bound = max(bound, -expected_future_demand / sum_expected_variances)

    return bound

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
            "guarantee": self.guarantee,
            "hope_online": self.hope_online
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

    def get_lambda_range(self):
        return (lambda_bound(self.demand_distributions), 2)

    def optimize_lambda(self, tol=1e-3, monte_carlo=False, batch=True, debug=False):
        """
        Optimize the lambda schedule.
        """
        self.change_allocation_method("lambda_time")
        lower_lambda, upper_lambda = self.get_lambda_range()

        def objective(l):
            self.set_lambda_schedule([l for _ in range(self.N)])
            # choose n to satisfy e = 0.01 with probability 0.01
            # use Bonferroni correction to get the probability of all being within e
            phi = (1 + np.sqrt(5)) / 2
            number_of_gss_iterations = int(np.log(tol / (upper_lambda - lower_lambda)) / -np.log(phi)) + 1
            n = hoeffding_bound(tol, 0.01 / (2 * number_of_gss_iterations))
            if not monte_carlo:
                return self.solve()[0]
            if batch:
                return self.monte_carlo_performance_metrics_batched(n)["social_welfare"]
            return self.monte_carlo_performance_metrics(n)["social_welfare"]

        result = golden_section_search(objective, lower_lambda, upper_lambda, tol=tol, debug=debug)
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

    def greedy(self, t: int, state: State, extra: ExtraState):
        # fulfill all demand, even if it exhausts us
        return np.min([state.d, state.c])

    def hope_online(self, t, state: State, extra: ExtraState):
        return 0
    
    def guarantee(self, t: int, state: State, extra: ExtraState):
        # like PPA, but for worst-case demand
        sum_guarantees = sum(
            [self.demand_distributions[i].max() for i in range(t, self.N)]
        )
        return np.min([state.d, state.c * (state.d / (state.d + sum_guarantees))], axis=0)

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

        social_welfares = np.zeros(n)
        wastes = np.zeros(n)
        min_fills = np.zeros(n)
        max_distances = np.zeros(n)
        max_envies = np.zeros(n)
        fill_rates = np.zeros((n, self.N))

        for i in range(n):
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
            fill_rate = calculate_fill_rates(allocations, demands)

            social_welfares[i] = social_welfare[0]
            wastes[i] = waste
            min_fills[i] = min_fill
            max_distances[i] = max_distance[0]
            max_envies[i] = max_envy_value[0]
            fill_rates[i, :] = fill_rate[0]

        return dict(
            social_welfare=confidence_interval(social_welfares),
            waste=confidence_interval(wastes),
            min_fill=confidence_interval(min_fills),
            max_dist=confidence_interval(max_distances),
            max_envy=confidence_interval(max_envies),
            fill_rates=[confidence_interval(fill_rate, confidence=0.95) for fill_rate in fill_rates.T]
        )
    
    def monte_carlo_performance_metrics_batched(self, n: int):
        if self.allocation_method not in ["ppa", "lambda", "lambda_time", "greedy", "guarantee"]:
            return self.monte_carlo_performance_metrics(n)

        batch_size = min(2000, n)

        iters = n // batch_size + 1
        true_n = iters * batch_size

        social_welfares = np.zeros(true_n)
        wastes = np.zeros(true_n)
        min_fills = np.zeros(true_n)
        max_distances = np.zeros(true_n)
        max_envies = np.zeros(true_n)
        fill_rates = np.zeros((true_n, self.N))

        for iter in range(iters):
            demands = self.demand_distributions[0].sample(batch_size)
            supplies = np.ones(batch_size) * self.initial_supply
            allocations = np.zeros((batch_size, self.N))
            all_demands = np.zeros((batch_size, self.N))
            batch_min_fills = np.ones(batch_size)
            batch_max_fills = np.zeros(batch_size)

            # run the simulation
            for t in range(1, self.N):
                if self.allocation_method in ["lambda", "lambda_time", "ppa"]:
                    # expected future demand
                    expected_future_demand = np.sum(self.demand_means[t:]) + np.dot(self.lambda_schedule[t:], [dist.stddev_val for dist in self.demand_distributions[t:]])
                    x = ppa_allocation_batched(demands, supplies, expected_future_demand)
                elif self.allocation_method == "greedy":
                    x = greedy_allocation_batched(demands, supplies)
                elif self.allocation_method == "guarantee":
                    sum_guarantees = sum([dist.max() for dist in self.demand_distributions[t:]])
                    x = guarantee_allocation_batched(demands, supplies, sum_guarantees)
                allocations[:, t - 1] = x
                all_demands[:, t - 1] = demands
                batch_min_fills = np.minimum(batch_min_fills, x / demands)
                batch_max_fills = np.maximum(batch_max_fills, x / demands)

                supplies = np.maximum(supplies - x, 0)
                demands = self.demand_distributions[t].sample(batch_size)

            # final allocation for the last node
            t = self.N - 1
            x = np.minimum(supplies, demands)
            allocations[:, t] = x
            all_demands[:, t] = demands
            batch_min_fills = np.minimum(batch_min_fills, x / demands)
            batch_max_fills = np.maximum(batch_max_fills, x / demands)

            # calculate the performance metrics
            social_welfares[iter * batch_size:(iter + 1) * batch_size] = social_welfare_relative(self.alpha)(allocations, all_demands)
            wastes[iter * batch_size:(iter + 1) * batch_size] = np.maximum(supplies - demands, 0)
            min_fills[iter * batch_size:(iter + 1) * batch_size] = batch_min_fills
            max_distances[iter * batch_size:(iter + 1) * batch_size] = max_distance_to_hindsight(allocations, all_demands, self.initial_supply)
            max_envies[iter * batch_size:(iter + 1) * batch_size] = max_envy(allocations, all_demands)
            fill_rates[iter * batch_size:(iter + 1) * batch_size, :] = calculate_fill_rates(allocations, all_demands)

        return dict(
            social_welfare=confidence_interval(social_welfares),
            waste=confidence_interval(wastes),
            min_fill=confidence_interval(min_fills),
            max_dist=confidence_interval(max_distances),
            max_envy=confidence_interval(max_envies),
            fill_rates=[confidence_interval(fill_rate, confidence=0.95) for fill_rate in fill_rates.T]
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
