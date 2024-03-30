"""
Provide utility functions.

This file provides the following distributions:
- Distribution: a discrete distribution with some pmf.
- UniformDistribution: a discrete approximation of a continuous uniform distribution.
- NormalDistribution: a discrete approximation of a continuous normal distribution.
- SymmetricDiscreteDistribution: a discrete symmetric distribution.
- GammaDistribution: a discrete approximation of a continuous gamma distribution.

"""
import math
from statistics import fmean

import numpy as np
import scipy as sp

class Distribution:
    def __init__(self, xk, pk):
        """
        Defines a discrete distribution with some pmf.
        """
        self.xk = xk
        self.pk = pk

        # sort by ascending xk
        self.xk, self.pk = zip(*sorted(zip(self.xk, self.pk)))

        # just precompute these
        self.mean_val = fmean(xk, pk)
        # Var[X] = E[X^2] - E[X]^2
        self.stddev_val = math.sqrt(self.expect(lambda x: x**2) - self.mean() ** 2)
        self.min_val = min(xk)
        self.max_val = max(xk)

    def expect(self, f):
        # simple expectation over a pmf
        return sum([p * f(x) for x, p in zip(self.xk, self.pk)])
    
    def expect_with_prob(self, f):
        return sum([p * f(x, p) for x, p in zip(self.xk, self.pk)])

    def mean(self):
        return self.mean_val

    def stddev(self):
        return self.stddev_val

    def cv(self):
        return self.stddev_val / self.mean_val

    def min(self):
        # the minimum value of the support
        return self.min_val

    def max(self):
        # the maximum value of the support
        return self.max_val

    def sample(self, n):
        # draw a random sample of size n
        return np.random.choice(self.xk, n, p=self.pk)
    
    def sample_with_prob(self, n):
        # draw a random sample of size n
        idx = np.random.choice(range(len(self.xk)), n, p=self.pk)
        return np.array(self.xk)[idx], np.array(self.pk)[idx]

    def scale(self, a):
        # scale support by a
        return Distribution(xk=tuple([a * x for x in self.xk]), pk=self.pk)

    def __str__(self) -> str:
        return "{ " + ", ".join([f"{x}: {p}" for x, p in zip(self.xk, self.pk)]) + " }"

### Discrete Approximations of Continuous Distributions
# https://users.ssc.wisc.edu/~jkennan/research/DiscreteApprox.pdf
# Result: max error over support scales with 1 / n, where
# n is the number of points used. Uses equiprobable buckets.
class ContinuousApproximation(Distribution):
    def __init__(self, pdf, inv_cdf, n):
        xk = [inv_cdf((2 * i + 1) / (2 * n)) for i in range(n)]
        pk = [pdf(xk[i]) for i in range(n)]
        pk = [p / sum(pk) for p in pk]
        super().__init__(xk, pk)

class UniformDistribution(ContinuousApproximation):
    def __init__(self, a, b, n=10):
        """
        Approximate continuous uniform distribution.
        """
        super().__init__(lambda x: 1 / (b - a), lambda p: a + p * (b - a), n)

class NormalDistribution(ContinuousApproximation):
    def __init__(self, mean, stddev, n=10):
        """
        Approximate continuous normal distribution.
        """

        if stddev == 0:
            raise ValueError("stddev must be positive")

        super().__init__(
            lambda x: sp.stats.norm.pdf(x, loc=mean, scale=stddev),
            lambda p: sp.stats.norm.ppf(p, loc=mean, scale=stddev),
            n,
        )

class SymmetricDiscreteDistribution(Distribution):
    def __init__(self, mean, delta, p):
        """
        Consider the distribution parameterized by $\mu, \sigma, p$.

        PMF:
        - $p(\mu-\delta)=\frac{1-p}{2}$
        - p(\mu)=p$
        - p(\mu+\delta)=\frac{1-p}{2}$.

        It is symmetric, centered at $\mu$.
        """

        xk = (mean - delta, mean, mean + delta)
        pk = ((1 - p) / 2, p, (1 - p) / 2)
        super().__init__(xk, pk)

class GammaDistribution(ContinuousApproximation):
    def __init__(self, shape, scale, n=10):
        """
        Approximate with continuous gamma distribution.
        """
        # use quantiles to get equiprobable buckets for the support

        if shape == 0 or scale == 0:
            raise ValueError("stddev must be positive")
        
        super().__init__(
            lambda x: sp.stats.gamma.pdf(x, a=shape, scale=scale),
            lambda p: sp.stats.gamma.ppf(p, a=shape, scale=scale),
            n,
        )
