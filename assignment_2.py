from functools import partial
from typing import Callable

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import monte_carlo
import math

Function = Callable[[float], float]


def f(x: float) -> float:
    # --------------------------------------------------------------------
    # define the target function f(x)=sin(x)
    return math.sin(x)
    # --------------------------------------------------------------------


def analytical_integral(a: float, b: float) -> float:
    # --------------------------------------------------------------------
    # exact ∫_a^b sin(x) dx = [-cos(x)]_a^b = cos(a) - cos(b)
    return math.cos(a) - math.cos(b)
    # --------------------------------------------------------------------


def transform(samples: npt.NDArray, a: float, b: float) -> npt.NDArray:
    # --------------------------------------------------------------------
    # (left unchanged)
    samples = np.zeros_like(samples)
    for i in range(len(samples)):
        samples[i] = a + (b - 1) * i
    return samples
    # --------------------------------------------------------------------


def integral_sin2(x: float) -> float:
    return 0.5 * x - 0.25 * math.sin(2 * x)

def integrate_mc(
    f: Function,
    a: float,
    b: float,
    n_samples: int,
    with_transform: bool = False,
    seed: int = 42,
) -> tuple[float, float]:
    # --------------------------------------------------------------------
    # Compute ∫_a^b f(x) dx via Monte Carlo and return (estimate, RMSE).
    #
    # - If with_transform=True: 
    #     draw U ∼ Unif(0,1), set X = a + (b-a)*U, then
    #     Ĩ = (b - a) * mean( f(X) ), RMSE = (b-a)*sqrt(Var[ f(X) ]/N).
    #
    # - If with_transform=False:
    #     draw X ∼ Unif(a,b) directly, then
    #     Ĩ = (b - a) * mean( f(X) ), RMSE = (b-a)*sqrt(Var[ f(X) ]/N).
    # --------------------------------------------------------------------
    mc_estimate = 0.0
    rmse = 0.0


    if with_transform:
        # ----------------------------------------------------------------
        # 1) Draw U ~ Unif(0,1), then X = a + (b-a)*U
        # ----------------------------------------------------------------
        dist0_1 = cp.Uniform(0.0, 1.0)
        U = dist0_1.sample(size=n_samples, seed=seed)
        X = a + (b - a) * U
        fX = np.array([f(xi) for xi in X])
        mc_estimate = (b - a) * np.mean(fX)

        # ----------------------------------------------------------------
        # 2) Compute Var[ sin(X) ] on [a,b]:
        #    E[ sin(X) ]   = (cos(a) - cos(b)) / (b - a)
        #    E[ sin^2(X) ] = [ (x/2 - sin(2x)/4 ) ]_a^b / (b - a)
        #    Var = E[sin^2] - (E[sin])^2
        # ----------------------------------------------------------------
        E_sin_ab = (math.cos(a) - math.cos(b)) / (b - a)

        E_sin2_ab = (integral_sin2(b) - integral_sin2(a)) / (b - a)
        var_sin_ab = E_sin2_ab - E_sin_ab**2

        rmse = abs(b - a) * math.sqrt(var_sin_ab / n_samples)

    else:
        # ----------------------------------------------------------------
        # Draw X ~ Unif(a,b) directly, then Ĩ = (b-a)*mean(f(X))
        # ----------------------------------------------------------------
        dist_a_b = cp.Uniform(a, b)
        X = dist_a_b.sample(size=n_samples, seed=seed)
        fX = np.array([f(xi) for xi in X])
        mc_estimate = (b - a) * np.mean(fX)

        # ----------------------------------------------------------------
        # Compute Var[ sin(X) ] on [a,b] exactly (same as above)
        # ----------------------------------------------------------------
        E_sin_ab = (math.cos(a) - math.cos(b)) / (b - a)


        E_sin2_ab = (integral_sin2(b) - integral_sin2(a)) / (b - a)
        var_sin_ab = E_sin2_ab - E_sin_ab**2

        rmse = abs(b - a) * math.sqrt(var_sin_ab / n_samples)

    return mc_estimate, rmse
    # ====================================================================


if __name__ == "__main__":
    # ------------------------
    # Choose parameters:
    # ------------------------
    N_list = [10, 100, 1000, 10000]

    # ── For Assignment 2.1: integrate on [0,1] without transform ──
    a = 2.0
    b = 4.0
    with_transform = True
    seed = 65

    # ── For Assignment 2.2: integrate on [2,4] with transform ──
    # a = 2.0
    # b = 4.0
    # with_transform = True
    # seed = 65

    # ------------------------
    # Compute MC estimates and errors:
    # ------------------------
    results = {}
    for n in N_list:
        mc_estimate, rmse = integrate_mc(f, a, b, n, with_transform, seed)
        exact_error = abs(analytical_integral(a, b) - mc_estimate)
        results[n] = {
            "mc_estimate": mc_estimate,
            "rmse": rmse,
            "exact_error": exact_error
        }
        print(f"N={n:5d} → Ĩ={mc_estimate:.8f}, RMSE≈{rmse:.8f}, |error|={exact_error:.8f}")

    # ------------------------
    # Plot results (log–log):
    # ------------------------
    exact_error_list = [results[n]["exact_error"] for n in N_list]
    rmse_list        = [results[n]["rmse"]         for n in N_list]

    plt.figure(figsize=(6, 4))
    plt.loglog(
        N_list,
        exact_error_list,
        "o-",
        label="Exact $|F - \\hat I_N|$"
    )
    plt.loglog(
        N_list,
        rmse_list,
        "s--",
        label="Theoretical RMSE"
    )
    plt.xlabel("Number of samples $N$")
    plt.ylabel("Error")
    plt.title("Monte Carlo: Exact Error vs. RMSE (log–log)")
    plt.grid(which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()
