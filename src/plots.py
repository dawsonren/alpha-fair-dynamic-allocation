"""
Utility functions for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from src.AllocationSolver import AllocationSolver, State, ExtraState

def plot_Z_versus_x(prob: AllocationSolver, t, state: State, ex_ante=False):
    x_values, Z_values = prob.provide_Z_vs_x_plot_data(t, state, ex_ante=ex_ante)

    plt.plot(x_values, Z_values)
    plt.vlines(state.c, 0, 1, colors="r", label="Supply (c)", linestyles="dotted")
    plt.vlines(state.d, 0, 1, colors="b", label="Demand (d)", linestyles="dotted")
    plt.vlines(prob.ppa_allocation(t, state, ExtraState(1, 1, 0, False), ex_ante=ex_ante), 0, 1, colors="g", label=f"PPA", linestyles="dotted")

    plt.xlabel("x")
    plt.ylabel("Z")
    plt.ylim(0, 1)
    plt.title(f"{'Ex-ante' if ex_ante else 'Ex-post'} Z vs. x at Node {state.i}, Time {t}, Supply {state.c}, Demand {state.d}")
    plt.legend()

def plot_x_versus_d(prob: AllocationSolver, t, state: State, step, ex_ante=False):
    d_values, x_values = prob.provide_x_vs_d_plot_data(t, state, step, ex_ante=ex_ante)

    plt.plot(d_values, x_values)

    # draw identity line
    plt.plot(d_values, d_values, linestyle="--", color="black")

    plt.xlabel("d")
    plt.ylabel("x")
    # make the plot square
    plt.xlim(0, d_values.max())
    plt.ylim(0, d_values.max())
    plt.title(f"{'Ex-ante' if ex_ante else 'Ex-post'} x vs. d at Node {t}, Time {t}, Supply {state.c}, Demand {state.d}")

def plot_against_alpha(data_generator, x_lab="x", y_lab="y", title=r"Plot against $\alpha$", identity=False):
    """
    data_generator should be a function that takes alpha
    and returns a tuple of (x_values, y_values) to plot.
    """
    alphas = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, np.inf])
    for alpha in alphas:
        x_values, y_values = data_generator(alpha)
        plt.plot(x_values, y_values, label=rf"$\alpha$={alpha}")

    if identity:
        # draw identity line
        plt.plot(x_values, x_values, linestyle="--", color="black")

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.legend()