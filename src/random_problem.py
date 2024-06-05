"""
Generate random AllocationSolver instances for testing and theorem proving
"""
import random

from src.AllocationSolver import AllocationSolver
from src.dists import Distribution, SymmetricDiscreteDistribution, NormalDistribution, ParetoDistribution, UniformDistribution

def generate_general_distribution(dist_points=2):
    """
    N: int, number of nodes
    dist_type: int, type of distribution to use, if number, then generates distribution with that many points
    """

    def gen_dist():
        # we add the (10 ** -round_to) to avoid 0
        xk = [random.random() for _ in range(dist_points)]
        pk = [random.random() for _ in range(dist_points)]
        # normalize pk to 1
        pk = [p / sum(pk) for p in pk]
        return Distribution(xk=xk, pk=pk)
    
    return gen_dist

def generate_symmetric_discrete_distribution(mean=None, p=None, d=None, relative=False):
    """
    mean: float, mean of the distribution
    """
    def gen_dist():
        p_val = random.random() if p is None else p
        # relative controls when d is a proportion of m (same CV), or d is absolute (same stddev)
        delta = random.random() if d is None else d
        if relative:
            m = random.random() if mean is None else mean
            delta = delta * m
        else:
            m = random.random() + delta if mean is None else mean
            if m <= delta: raise ValueError("Mean must be greater than delta")

        return SymmetricDiscreteDistribution(mean=m, stddev=delta, p=p_val)
    
    return gen_dist

def generate_normal_distribution(mean=(1, 2), cv=0.2, n=10):
    """
    mean: tuple, range of means to generate
    stddev: float, standard deviation of the distribution
    n: int, number of points to generate
    """
    def gen_dist():
        a, b = mean
        m = random.random() * (b - a) + a
        sd = cv * m
        return NormalDistribution(mean=m, stddev=sd, n=n)
    
    return gen_dist

def generate_pareto_distribution(mean=(1, 2), n=10):
    """
    mean: tuple, range of means to generate
    n: int, number of points to generate
    """
    def gen_dist():
        a, b = mean
        m = random.random() * (b - a) + a
        return ParetoDistribution(mean=m, n=n)
    
    return gen_dist

def generate_uniform_distribution(mean=(1, 2), n=10):
    """
    mean: tuple, range of means to generate
    n: int, number of points to generate
    """
    def gen_dist():
        a, b = mean
        a, b = list(sorted([random.random() * (b - a) + a, random.random() * (b - a) + a]))
        return UniformDistribution(a, b, n=n)
    
    return gen_dist

# generate random problem
def generate_random_problem(N, distribution_generator=None, allocation_method="ppa", alloc_step=0.1, supply_scarcity=1):
    """
    N: int, number of nodes
    dist_type: int, type of distribution to use, if number, then generates distribution with that many points
    allocation_method: str, allocation method to use
    sequence_method: str, sequence method to use
    """
    dist_gen = distribution_generator if distribution_generator is not None else generate_general_distribution(2)
    demand_distributions = []
    for _ in range(N):
        demand_distributions.append(dist_gen())

    mean_demand = sum([d.mean() for d in demand_distributions])

    solver = AllocationSolver(demand_distributions=demand_distributions,
                              initial_supply=mean_demand * supply_scarcity,
                              allocation_method=allocation_method,
                              alloc_step=alloc_step)
    return solver