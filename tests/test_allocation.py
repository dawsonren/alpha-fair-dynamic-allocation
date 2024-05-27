"""
Basic Tests for Allocation
"""
import unittest
import random
import math
from scipy import integrate
import numpy as np

from src.AllocationSolver import AllocationSolver, State, ExtraState
from src.dists import Distribution, UniformDistribution, SymmetricDiscreteDistribution
from src.monte_carlo import hoeffding_bound
from src.random_problem import generate_random_problem, generate_general_distribution

from tests.random_property_test import random_property_test


class TestAllocationSolver(unittest.TestCase):
    def test_simple(self):
        node = Distribution(xk=(1, 2), pk=(0.5, 0.5))
        solver = AllocationSolver([node], 1, alloc_step=0.1)
        self.assertEqual(solver.solve()[0], 0.75)
        self.assertEqual(solver.solve()[1], 0)

    def test_simple2(self):
        node_1 = Distribution(xk=(120,), pk=(1,))
        node_2 = Distribution(xk=(40,), pk=(1,))
        solver = AllocationSolver([node_1, node_2], 40, alloc_step=10)
        self.assertEqual(solver.solve()[0], 1 / 4)
        self.assertEqual(solver.solve()[1], 0)

    def test_lien(self):
        # Test from Lien, et al. (2014) Figure 2
        node_1 = Distribution(xk=(80, 120), pk=(0.5, 0.5))
        node_2 = Distribution(xk=(40, 60), pk=(0.5, 0.5))
        # setting verbosity = 1 lets you see that the allocations
        # are indeed 75 if d_1 = 80, and 80 if d_1 = 120
        solver = AllocationSolver([node_1, node_2], 130, alloc_step=1, verbosity=1)
        # waste is 4.5
        self.assertEqual(solver.solve()[1], 4.5)

    def test_deterministic_oversupply(self):
        # when s > sum of demands
        # random.seed(42)

        # function that generates random instance
        def generate_random_deterministic_oversupply(allocation_method):
            def generate():
                nodes = []
                total_demand = 0
                for _ in range(3):
                    demand = random.random()
                    total_demand += demand
                    node = Distribution(xk=(demand,), pk=(1,))
                    nodes.append(node)

                # supply is strictly greater than demand
                solver = AllocationSolver(
                    nodes, total_demand + 1, allocation_method=allocation_method
                )
                return solver

            return generate

        # property that we want to test
        def test_random_deterministic_oversupply(instance: AllocationSolver):
            Z, _ = instance.solve()
            self.assertAlmostEqual(Z, 1)
            

        random_property_test(
            generate_random_deterministic_oversupply("ppa"),
            test_random_deterministic_oversupply,
            instances=10,
        )

        random_property_test(
            generate_random_deterministic_oversupply("exact"),
            test_random_deterministic_oversupply,
            instances=1,
        )

    def test_handwritten_1(self):
        # solved by hand
        node_1 = Distribution(xk=(3, 5), pk=(0.5, 0.5))
        node_2 = Distribution(xk=(2, 6), pk=(0.5, 0.5))

        solver = AllocationSolver([node_1, node_2], 3)
        self.assertAlmostEqual(solver.solve()[0], 12 / 35, places=2)

    def test_ppa_policy(self):
        node_1 = UniformDistribution(5, 7, n=3)
        node_2 = UniformDistribution(2, 6, n=3)
        node_3 = UniformDistribution(1, 3, n=3)
        solver = AllocationSolver([node_1, node_2, node_3], initial_supply=8)

        self.assertEqual(solver.ppa_allocation(1, State(8, 5), ExtraState(1, 1, 0, 0)), 8 * (5 / 11))
        self.assertEqual(solver.ppa_allocation(1, State(8, 6), ExtraState(1, 1, 0, 0)), 4)
        self.assertEqual(solver.ppa_allocation(2, State(4, 3), ExtraState(1, 1, 0, 0)), 2.4)

    def test_uniform_specific(self):
        # compare to raw DP, encountered in testing
        c = 1
        d1 = 0.3
        d2 = 0.6
        mu = 1
        node_1 = UniformDistribution(mu - d1, mu + d1, n=50)
        node_2 = UniformDistribution(mu - d2, mu + d2, n=50)
        solver = AllocationSolver(
            [node_1, node_2], initial_supply=c, allocation_method="ppa"
        )

        def Z12(d2, d1):
            x = min(d1 * c / (d1 + mu), d1)
            f = min(x / d1, (c - x) / d2, 1)
            return f

        integral_12, _ = integrate.dblquad(Z12, mu - d1, mu + d1, mu - d2, mu + d2)

        self.assertAlmostEqual(
            solver.solve()[0],
            (1 / (4 * d1 * d2)) * integral_12,
            places=4,
        )

    def test_uniform_random_generation(self):
        def generate_uniform_random_generation_instance():
            # compare to raw DP, encountered in testing
            d1 = random.random()
            d2 = random.random()
            mu = 1
            # uniform between 0 and maximum
            c = random.random() * (2 * mu + d1 + d2)
            node_1 = UniformDistribution(mu - d1, mu + d1, n=50)
            node_2 = UniformDistribution(mu - d2, mu + d2, n=50)
            solver = AllocationSolver(
                [node_1, node_2], initial_supply=c, allocation_method="ppa"
            )

            def Z12(d2, d1):
                x = min(d1 * c / (d1 + mu), d1)
                f = min(x / d1, (c - x) / d2, 1)
                return f

            integral_12, _ = integrate.dblquad(Z12, mu - d1, mu + d1, mu - d2, mu + d2)

            return (
                solver,
                (1 / (4 * d1 * d2)) * integral_12
            )

        def test_uniform_random_generation_instance(inputs):
            solver, integral_12 = inputs
            self.assertAlmostEqual(
                solver.solve()[0], integral_12, places=4
            )

        random_property_test(
            generate_uniform_random_generation_instance,
            test_uniform_random_generation_instance,
            instances=10,
        )

    def test_monte_carlo(self):
        random.seed(42)
        # set numpy seed
        np.random.seed(42)

        def test_monte_carlo_instance(prob):
            n = hoeffding_bound(0.01, 0.01)
            Z, w = prob.solve()
            metrics = prob.monte_carlo_performance_metrics(n)

            self.assertAlmostEqual(metrics["social_welfare"], Z, places=2)
            self.assertAlmostEqual(metrics["waste"], w, places=2)

        random_property_test(
            lambda: generate_random_problem(4, generate_general_distribution(4), "ppa"),
            test_monte_carlo_instance,
            instances=10
        )
        

if __name__ == "__main__":
    unittest.main()
