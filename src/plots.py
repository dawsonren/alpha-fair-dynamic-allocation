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
from tqdm import tqdm

from src.AllocationSolver import AllocationSolver, State

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
    "lambda_time": "PPA-AF",
    "lambda": r"PPA-AF (Constant $\lambda$)",
    "greedy": "GREEDY",
    "guarantee": "GUARANTEE",
    "hope_online": "hope_online"
}

# mpl colors
map_algo_to_color = {
    "exact": "tab:purple",
    "ppa": "tab:orange",
    "lambda_time": "tab:blue",
    "lambda": "tab:cyan",
    "greedy": "tab:green",
    "guarantee": "tab:red",
    "hope_online": "tab:gray"
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

###
### Plots for the final report
###

def generate_data(prob: AllocationSolver, algos, n=1000, debug=False, tol=0.01):
    """
    Uses supply equal to the expected demand.

    Returns (x-axis, y-axis) for each of the four graphs (see synthetic_summary_plots).
    """
    num_agents = prob.N

    # get metrics vs alpha for num_agents
    if debug:
        print("Starting metrics vs. alpha...")

    alpha_values = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])
    social_welfare_vs_alpha = np.zeros((len(algos), alpha_values.size, 2))
    waste_vs_alpha = np.zeros((len(algos), alpha_values.size, 2))

    for i, algo in enumerate(algos):
        if debug: print(f"Starting {algo}...")
        prob.change_allocation_method(algo)
        for j, alpha in tqdm(enumerate(alpha_values)):
            if debug: print(f"Alpha: {alpha}")
            prob.change_alpha(alpha)
            if algo == "lambda_time": prob.optimize_lambda(tol=tol, monte_carlo=True, batch=True)
            results = prob.monte_carlo_performance_metrics_batched(n)
            social_welfare_vs_alpha[i, j, :] = results["social_welfare"]
            waste_vs_alpha[i, j, :] = results["waste"]

    # the last two plots are for alpha = infinity
    prob.change_alpha(np.inf)

    # get metrics for arrival order with 100 agents
    fill_rates = np.zeros((len(algos), num_agents, 2))

    if debug:
        print("Starting arrival order vs. fill rates...")

    for i, algo in enumerate(algos):
        if debug: print(f"Starting {algo}...")
        if algo == "lambda_time": prob.optimize_lambda(tol=tol, monte_carlo=True, batch=True)
        prob.change_allocation_method(algo)
        results = prob.monte_carlo_performance_metrics_batched(n)
        fill_rates[i, :, :] = results["fill_rates"]

    agent_position_vector = np.arange(1, num_agents + 1)

    # get fill rate vs uncertainty of each agent
    uncertainty_vector = np.array([dist.stddev_val for dist in prob.demand_distributions])
    fill_rate_versus_uncertainty = np.zeros((len(algos), uncertainty_vector.size))

    if debug:
        print("Starting fill rate vs. uncertainty...")    

    # rearrange the agents by uncertainty
    agent_order = np.argsort(uncertainty_vector)
    # rearrange the 1st axis of fill_rates
    fill_rate_versus_uncertainty = (fill_rates[:, agent_order, 0] + fill_rates[:, agent_order, 1]) / 2

    return (alpha_values, social_welfare_vs_alpha), (alpha_values, waste_vs_alpha), (agent_position_vector, fill_rates), (uncertainty_vector, fill_rate_versus_uncertainty)

def save_data(data, filename):
    # where the data has the form
    # (alpha_values, social_welfare_vs_alpha), (alpha_values, waste_vs_alpha), (agent_position_vector, fill_rates), (uncertainty_vector, fill_rate_versus_uncertainty)
    # with appropriate keywords

    # save to ../../data/
    np.savez(f"../../data/{filename}.npz",
        alpha_values=data[0][0],
        social_welfare_vs_alpha=data[0][1],
        waste_vs_alpha=data[1][1],
        agent_position_vector=data[2][0],
        fill_rates=data[2][1],
        uncertainty_vector=data[3][0],
        fill_rate_versus_uncertainty=data[3][1]
    )

def load_data(filename):
    data = np.load(f"../../data/{filename}.npz")
    return (
        (data["alpha_values"], data["social_welfare_vs_alpha"]),
        (data["alpha_values"], data["waste_vs_alpha"]),
        (data["agent_position_vector"], data["fill_rates"]),
        (data["uncertainty_vector"], data["fill_rate_versus_uncertainty"])
    )

def summary_plots(data, algos):
    """
    Top left: social welfare vs. alpha, with greedy as the baseline (normalized to 1)
    Top right: waste vs. alpha
    Bottom left: fill rate vs. arrival order
    Bottom right: fill rate vs. uncertainty

    NOTE: In general, the numpy arrays are of shape (num_algos, num_x_axis_values, 2)
    where the last dimension is (upper bound, lower bound). The confidence interval
    is symmetric, so the center line is the mean of the two bounds.
    """
    # seaborn style for report
    sns.set_context("paper")
    sns.set_style("whitegrid")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # unpack the data
    alpha_values, social_welfare_vs_alpha = data[0]
    _, waste_vs_alpha = data[1]
    agent_position_vector, fill_rates = data[2]
    uncertainty_vector, fill_rate_versus_uncertainty = data[3]

    # plot social welfare vs alpha
    for i, algo in enumerate(algos):
        average = (social_welfare_vs_alpha[i, :, 0] + social_welfare_vs_alpha[i, :, 1]) / 2
        sns.lineplot(x=alpha_values, y=average, label=map_algo_to_name[algo], color=map_algo_to_color[algo], ax=axs[0, 0])
        axs[0, 0].fill_between(alpha_values, social_welfare_vs_alpha[i, :, 1], social_welfare_vs_alpha[i, :, 0], alpha=0.1, color=map_algo_to_color[algo])

    axs[0, 0].set_xlabel(r"Alpha ($\alpha$)")
    axs[0, 0].set_ylabel("Expected Social Welfare")
    axs[0, 0].set_title(rf"Expected Social Welfare vs. Alpha ($\alpha$) for {agent_position_vector.max()} agents")

    # plot waste vs alpha
    for i, algo in enumerate(algos):
        # omit from plot because it's too high
        if algo == "guarantee": continue
        average = (waste_vs_alpha[i, :, 0] + waste_vs_alpha[i, :, 1]) / 2
        sns.lineplot(x=alpha_values, y=average, label=map_algo_to_name[algo], color=map_algo_to_color[algo], ax=axs[0, 1])
        axs[0, 1].fill_between(alpha_values, waste_vs_alpha[i, :, 1], waste_vs_alpha[i, :, 0], alpha=0.1, color=map_algo_to_color[algo])

    axs[0, 1].set_xlabel(r"Alpha ($\alpha$)")
    axs[0, 1].set_ylabel("Expected Waste")
    axs[0, 1].set_title(rf"Expected Waste vs. Alpha ($\alpha$) for {agent_position_vector.max()} agents")

    # plot fill rates for arrival order
    for i, algo in enumerate(algos):
        average = (fill_rates[i, :, 0] + fill_rates[i, :, 1]) / 2
        sns.lineplot(x=np.arange(1, agent_position_vector.size + 1), y=average, label=map_algo_to_name[algo], color=map_algo_to_color[algo], ax=axs[1, 0])
        axs[1, 0].fill_between(np.arange(1, agent_position_vector.size + 1), fill_rates[i, :, 1], fill_rates[i, :, 0], alpha=0.1, color=map_algo_to_color[algo])

    axs[1, 0].set_xlabel("Arrival Order")
    axs[1, 0].set_ylabel(r"Fill Rate ($\beta_i$)")
    axs[1, 0].set_title(r"Fill Rate vs. Arrival Order for $\alpha=\infty$")
    axs[1, 0].legend()

    # scatterplot of fill rate vs uncertainty
    for i, algo in enumerate(algos):
        sns.scatterplot(x=uncertainty_vector, y=fill_rate_versus_uncertainty[i], label=map_algo_to_name[algo], color=map_algo_to_color[algo], ax=axs[1, 1])
    
    axs[1, 1].set_xlabel(r"Standard Deviation of Demand ($\sigma_i$)")
    axs[1, 1].set_ylabel(r"Fill Rate ($\beta_i$)")
    axs[1, 1].set_title(r"Fill Rate vs. Uncertainty for $\alpha=\infty$")
    axs[1, 1].legend()
    
    plt.tight_layout()


def generate_scarcity_data(prob: AllocationSolver, algos, n=1000, debug=False, tol=0.01):
    """
    Get supply scarcity vs social welfare for alpha = [0, 1, 2, infty].
    """
    alpha_values = np.array([0, 1, 2, np.inf])
    scarcity_values = np.arange(0.1, 1.5, 0.1)
    social_welfare_vs_alpha_and_scarcity = np.zeros((len(algos), alpha_values.size, scarcity_values.size, 2))

    for i, algo in enumerate(algos):
        if debug: print(f"Starting {algo}...")
        prob.change_allocation_method(algo)
        for j, alpha in tqdm(enumerate(alpha_values)):
            if debug: print(f"Alpha: {alpha}")
            prob.change_alpha(alpha)
            for k, scarcity in enumerate(scarcity_values):
                prob.change_initial_supply(scarcity * sum(prob.demand_means))
                if algo == "lambda_time": prob.optimize_lambda(tol=tol, monte_carlo=True, batch=True)
                results = prob.monte_carlo_performance_metrics_batched(n)
                social_welfare_vs_alpha_and_scarcity[i, j, k, :] = results["social_welfare"]

    return (alpha_values, scarcity_values, social_welfare_vs_alpha_and_scarcity)

def save_scarcity_data(data, filename):
    # where the data has the form
    # (alpha_values, scarcity_values, social_welfare_vs_alpha_and_scarcity)

    # save to ../../data/
    np.savez(f"../../data/{filename}.npz",
        alpha_values=data[0],
        scarcity_values=data[1],
        social_welfare_vs_alpha_and_scarcity=data[2]
    )

def load_scarcity_data(filename):
    data = np.load(f"../../data/{filename}.npz")
    return (
        data["alpha_values"],
        data["scarcity_values"],
        data["social_welfare_vs_alpha_and_scarcity"]
    )

def scarcity_plots(data, algos):
    """
    Top left: alpha = 0
    Top right: alpha = 1
    Bottom left: alpha = 2
    Bottom right: alpha = infinity

    NOTE: In general, the numpy arrays are of shape (num_algos, num_x_axis_values, num_y_axis_values, 2)
    where the last dimension is (upper bound, lower bound). The confidence interval
    is symmetric, so the center line is the mean of the two bounds.
    """
    # seaborn style for report
    sns.set_context("paper")
    sns.set_style("whitegrid")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # unpack the data
    alpha_values, scarcity_values, social_welfare_vs_alpha_and_scarcity = data

    # plot social welfare vs alpha
    for i, algo in enumerate(algos):
        for j, alpha in enumerate(alpha_values):
            average = (social_welfare_vs_alpha_and_scarcity[i, j, :, 0] + social_welfare_vs_alpha_and_scarcity[i, j, :, 1]) / 2
            sns.lineplot(x=scarcity_values, y=average, label=map_algo_to_name[algo], color=map_algo_to_color[algo], ax=axs[j // 2, j % 2])
            axs[j // 2, j % 2].fill_between(scarcity_values, social_welfare_vs_alpha_and_scarcity[i, j, :, 1], social_welfare_vs_alpha_and_scarcity[i, j, :, 0], alpha=0.1, color=map_algo_to_color[algo])
            axs[j // 2, j % 2].set_xlabel("Supply Scarcity")
            axs[j // 2, j % 2].set_ylabel("Expected Social Welfare")
            axs[j // 2, j % 2].set_title(rf"Expected Social Welfare vs. Initial Supply Scarcity for $\alpha={alpha}$")

            # set ylim to 0, 1
            axs[j // 2, j % 2].set_ylim(0, 1.05)

    plt.tight_layout()