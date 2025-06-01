from collections import defaultdict

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator


def load_reference(filename: str) -> tuple[float, float]:
    # TODO: load reference values for the mean and variance.
    # ====================================================================
    with open(filename, "r") as f:
        lines = f.readlines()
    mean = float(lines[0].split("=")[-1])
    var = float(lines[1].split("=")[-1])
    # ====================================================================
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    n_samples: int,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    rule="random",
    seed=42,
) -> npt.NDArray:
    # TODO: simulate the oscillator with the given parameters and return
    # generated solutions.
    # ====================================================================
    np.random.seed(seed)
    if rule == "random":
        samples = omega_distr.sample(n_samples, seed=seed)
    elif rule == "halton":
        samples = omega_distr.sample(n_samples, rule="halton", seed=seed)
    else:
        raise ValueError("Unknown sampling rule")
    
    sample_solutions = np.zeros((n_samples, len(t_grid)))
    
    for i in range(n_samples):
        omega_i = samples[i]
        oscillator = Oscillator(omega=omega_i, **model_kwargs)
        y = oscillator.discretize(
            method="odeint",
            y0=init_cond["y0"],
            y1=init_cond["y1"],
            t_grid=t_grid,
        )
        sample_solutions[i, :] = y
    # ====================================================================
    return sample_solutions


def compute_relative_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO: compute the relative errors of the mean and variance
    # estimates.
    # ====================================================================
    y_T = samples[:, -1]
    mean_estimate = np.mean(y_T)
    var_estimate = np.var(y_T, ddof=1)

    mean_rel_error = abs(1 - mean_estimate / mean_ref)
    var_rel_error = abs(1 - var_estimate / var_ref)
    # ====================================================================
    return mean_rel_error, var_rel_error


if __name__ == "__main__":
    # TODO: define the parameters of the simulations.
    # ====================================================================
    model_kwargs = {
        "c": 0.5,
        "k": 2.0,
        "f": 0.5,
    }
    init_cond = {
        "y0": 0.5,
        "y1": 0.0,
    }
    t_grid = np.arange(0, 10 + 0.01, 0.01)  # time grid from 0 to 10 with dt = 0.01

    mean_ref, var_ref = load_reference("./data/oscillator_ref.txt")

    omega_distr = cp.Uniform(0.95, 1.05)
    # ====================================================================

    # TODO: run the simulations.
    # ====================================================================
    # TODO: compute the statistics.
    # ====================================================================
    sample_sizes = [10, 100, 1000, 10000]
    methods = ["random", "halton"]

    rel_errors = defaultdict(list)

    for method in methods:
        for n in sample_sizes:
            samples = simulate(t_grid, omega_distr, n, model_kwargs, init_cond, rule=method)
            mean_rel_err, var_rel_err = compute_relative_errors(samples, mean_ref, var_ref)
            rel_errors[f"{method}_mean"].append(mean_rel_err)
            rel_errors[f"{method}_var"].append(var_rel_err)
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    plt.figure(figsize=(10, 5))
    for method in methods:
        plt.plot(sample_sizes, rel_errors[f"{method}_mean"], label=f"{method} - mean rel error", marker='o')
        plt.plot(sample_sizes, rel_errors[f"{method}_var"], label=f"{method} - var rel error", marker='x')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Samples (log)")
    plt.ylabel("Relative Error (log)")
    plt.legend()
    plt.title("Relative Error vs Number of Samples")
    plt.grid(True)
    plt.show()
    # ====================================================================

    # TODO: plot sampled trajectories.
    # ====================================================================
    n_plot_samples = 10
    sample_solutions = simulate(t_grid, omega_distr, n_plot_samples, model_kwargs, init_cond, rule="random")
    
    plt.figure(figsize=(10, 5))
    for i in range(n_plot_samples):
        plt.plot(t_grid, sample_solutions[i, :])
    plt.xlabel("Time t")
    plt.ylabel("Displacement y(t)")
    plt.title("Sampled Oscillator Trajectories (Random Sampling)")
    plt.grid(True)
    plt.show()
    # ====================================================================
