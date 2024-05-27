"""
Provide golden section search for derivative-free optimization
of a single-variable unimodal function.
"""
import numpy as np

def golden_section_search(f, a, b, tol=1e-5, debug=False):
    """
    Find the maximum of a unimodal function f on the interval [a, b].
    """
    iterations = 1

    # golden ratio
    phi = (1 + 5 ** 0.5) / 2

    # initial points
    c = b - (b - a) / phi
    d = a + (b - a) / phi

    while abs(c - d) > tol:
        fc, fd = f(c), f(d)

        if debug:
            print(f"Iteration {iterations}")
            print(f"Initial points: {c}, {d}")
            print(f"Initial values: {fc}, {fd}")
        
        if fc > fd:
            b = d
        else:
            a = c

        c = b - (b - a) / phi
        d = a + (b - a) / phi

        iterations += 1

    return (a + b) / 2
