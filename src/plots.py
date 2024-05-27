"""
Utility functions for plotting.
"""
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns

from src.AllocationSolver import AllocationSolver, State, ExtraState
from src.monte_carlo import hoeffding_bound

map_metric_to_latex = {
    "social_welfare": r"$\mathbb{E}[U_{\alpha}]$",
    "waste": r"$\Delta_{PE}$",
    "min_fill": r"$\mathbb{E}[\min_i \beta_i]$",
    "max_dist": r"$\Delta_{Prop}$",
    "max_envy": r"$\Delta_{EF}$"
}

map_algo_to_name = {
    "exact": "Exact",
    "ppa": "PPA",
    "lambda_time": "PPA-F",
    "greedy": "Greedy",
    "max_demand": "MaxDemand",
    "lien": "MaxMin",
    "hope_guardrail_12": r"GuardedHope $L_t=\frac{1}{2}$",
    "hope_guardrail_13": r"GuardedHope $L_t=\frac{1}{3}$",
    "saffe": "SAFFE"
}

# mpl colors
map_algo_to_color = {
    "exact": "tab:purple",
    "ppa": "tab:orange",
    "lambda_time": "tab:blue",
    "greedy": "tab:green",
    "max_demand": "tab:red",
    "lien": "tab:purple",
    "hope_guardrail_12": "tab:brown",
    "hope_guardrail_13": "tab:pink",
    "saffe": "tab:gray"
}

def plot_x_versus_d(prob: AllocationSolver, t, state: State, step):
    d_values, x_values = prob.provide_x_vs_d_plot_data(t, state, step)

    plt.plot(d_values, x_values)

    # draw identity line
    plt.plot(d_values, d_values, linestyle="--", color="black")

    plt.xlabel("d")
    plt.ylabel("x")
    # make the plot square
    plt.xlim(0, d_values.max())
    plt.ylim(0, d_values.max())
    plt.title(f"Ex-post x vs. d at Node {t}, Time {t}, Supply {state.c}, Demand {state.d}")

def plot_Z_versus_lambda(prob: AllocationSolver, step, compare_opt=False):
    lambda_vals, Z_values = prob.provide_Z_versus_lambda_plot_data(step)

    plt.plot(lambda_vals, Z_values)

    plt.xlabel(r"$\lambda$")
    plt.ylabel("Expected Social Welfare")
    plt.ylim(0, 1)
    plt.title(rf"Expected Social Welfare vs. $\lambda$ for $\alpha$={prob.alpha}")

    # draw the lambda that maximizes Z, out of the values given
    lambda_max = lambda_vals[np.argmax(Z_values)]
    plt.vlines(lambda_max, 0, 1, colors="r", label=rf"$\lambda$={round(lambda_max, 3)}", linestyles="dotted")

    if compare_opt:
        # draw the Z value with exact allocation
        new_prob = prob.copy()
        new_prob.change_allocation_method("exact")
        Z_star, _ = new_prob.solve()
        plt.hlines(Z_star, lambda_vals.min(), lambda_vals.max(), colors="b", label="Optimal Social Welfare", linestyles="dotted")

        print(f"Optimality gap for best lambda: {Z_star - Z_values[np.argmax(Z_values)]}")

    plt.legend()

def plot_Z_versus_lambda_s(prob: AllocationSolver, lambda_step, supply_step):
    lambda_values, s_values, Z_values = prob.provide_Z_versus_lambda_s_plot_data(lambda_step, supply_step)

    X, Y = np.meshgrid(lambda_values, s_values, indexing="ij")
    plt.contourf(X, Y, Z_values, levels=20, cmap="viridis")
    plt.colorbar()

    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$s$")
    plt.title(rf"Ex-post Z vs. $\lambda$, $s$ for $\alpha$={prob.alpha}")

def plot_best_lambda_versus_s(prob: AllocationSolver, lambda_step, supply_step, compare_opt=False):
    """
    Plot the best lambda value for each supply value.
    """
    lambda_values, s_values, Z_values = prob.provide_Z_versus_lambda_s_plot_data(lambda_step, supply_step)

    best_lambdas = np.argmax(Z_values, axis=0)
    plt.plot(s_values, lambda_values[best_lambdas])

    plt.xlabel(r"$s$")
    plt.ylabel(r"$\lambda$")
    plt.title(rf"Best $\lambda$ vs. $s$ for $\alpha$={prob.alpha}")


def plot_Z_versus_three_node_lambda(prob: AllocationSolver, step, compare_opt=False):
    lambda_vals, Z_values = prob.provide_Z_versus_three_node_lambda_plot_data(step)

    X, Y = np.meshgrid(lambda_vals, lambda_vals)
    plt.contourf(X, Y, Z_values, levels=20, cmap="viridis")
    plt.colorbar()
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel(r"$\lambda_2$")
    plt.title(rf"Ex-post Z vs. $\lambda_1, \lambda_2$ for $\alpha$={prob.alpha}")

    if compare_opt:
        new_prob = prob.copy()
        new_prob.change_allocation_method("exact")
        Z_star, _ = new_prob.solve()

        print(f"Optimality gap for best lambda: {Z_star - Z_values.max()}")

def plot_versus_alpha(data_generator, x_lab="x", y_lab="y", title="Plot versus alpha"):
    """
    data_generator should be a function that takes alpha
    and returns a tuple of (x_values, y_values) to plot.
    """
    alphas = np.array([0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf])
    results = np.array([data_generator(alpha) for alpha in alphas])
    plt.plot(alphas, results, label="PPA-AF")

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)

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

def plot_against_algos(data_generator, algos, x_lab="x", y_lab="y", title="Plot against different allocation methods"):
    """
    Plot the Z values for the different allocation methods.
    """
    for algo in algos:
        x_values, y_values = data_generator(algo)
        plt.plot(x_values, y_values, label=map_algo_to_name[algo])

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.legend()

def plot_alpha_versus_lambda(prob_generator: Callable[[], AllocationSolver], replications=100, monte_carlo=False, tol=0.01):
    alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lambdas = np.zeros((replications, alphas.size))

    for i in range(replications):
        prob = prob_generator()
        for j, alpha in enumerate(alphas):
            prob.change_alpha(alpha)
            lambdas[i, j] = prob.optimize_lambda(tol=tol, monte_carlo=monte_carlo)

    sns.set_context("talk")
    sns.set_style("whitegrid")

    # plot each replication as a line, weak opacity
    for i in range(replications):
        plt.plot(alphas, lambdas[i], alpha=0.1, color="blue")

    # plot the average
    plt.plot(alphas, np.mean(lambdas, axis=0), color="red", label=r"Average $\lambda$")

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"Optimal $\lambda$")
    plt.title(r"Optimal $\lambda$ versus $\alpha$")
    plt.legend()
