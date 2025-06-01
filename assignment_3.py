import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from utils.sampling import control_variates, importance_sampling, monte_carlo

# f(x) = exp(x).
def f(x: float) -> float:
     # TODO: define the target function.
    # ====================================================================
    return np.exp(x)
    # ====================================================================


def analytical_integral() -> float:
    # TODO: compute the analytical integral of f on [0, 1].
    # ====================================================================
    return np.exp(1) - 1
    # ====================================================================

def run_monte_carlo(Ns: list[int], seed: int = 2025):
    # TODO: run the Monte Carlo method and return the absolute error
    # of the estimation.
    # ====================================================================
    np.random.seed(seed)
    true_val = analytical_integral()
    errors = []
    for N in Ns:
        x = np.random.uniform(0, 1, size=N)
        est = np.mean(f(x))
        errors.append(abs(est - true_val))
    return errors
    # ====================================================================


def run_control_variates(Ns: list[int], seed: int = 2025):
    # TODO: run the control variate method for and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    np.random.seed(seed)
    true_val = analytical_integral()
    phi1 = lambda x: x
    phi2 = lambda x: 1 + x
    phi3 = lambda x: 1 + x + x**2 / 2
    phi_fns = [phi1, phi2, phi3]

    all_errors = []
    for phi in phi_fns:
        errors = []
        for N in Ns:
            x = np.random.uniform(0, 1, size=N)
            fx = f(x)
            phix = phi(x)
            c = np.cov(fx, phix)[0, 1] / np.var(phix)
            expected_phi = np.mean([phi(xi) for xi in np.linspace(0, 1, 1000)])
            adjusted = fx - c * (phix - expected_phi)
            estimate = np.mean(adjusted)
            errors.append(abs(estimate - true_val))
        all_errors.append(errors)
    return tuple(all_errors)
    # ====================================================================


def run_importance_sampling(Ns: list[int], seed: int = 2025):
    # TODO: run the importance sampling method and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    np.random.seed(seed)
    true_val = analytical_integral()
    betas = [(5, 1), (0.5, 0.5)]
    errors_all = []
    for alpha, beta in betas:
        errors = []
        dist = cp.Beta(alpha, beta)
        for N in Ns:
            x = dist.sample(N, rule="random")
            w = f(x) / dist.pdf(x)
            est = np.mean(w)
            errors.append(abs(est - true_val))
        errors_all.append(errors)
    return tuple(errors_all)
    # ====================================================================


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    Ns = [10, 100, 1000, 10000]

    # TODO: run all the methods.
    # ====================================================================
    mc_errors = run_monte_carlo(Ns)
    cv_errors1, cv_errors2, cv_errors3 = run_control_variates(Ns)
    is_errors1, is_errors2 = run_importance_sampling(Ns)

    # table of abosolute value
    df = pd.DataFrame({
        "Sample Size": Ns,
        "Standard MC": mc_errors,
        "CV: phi1 = x": cv_errors1,
        "CV: phi2 = 1 + x": cv_errors2,
        "CV: phi3 = 1 + x + x^2/2": cv_errors3,
        "IS: Beta(5,1)": is_errors1,
        "IS: Beta(0.5,0.5)": is_errors2
    })

    print("Absolute Errors for Each Strategy:\n")
    print(df.to_markdown(index=False))

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    plt.figure(figsize=(10, 6))
    plt.loglog(Ns, mc_errors, label="Standard MC", marker='o')
    plt.loglog(Ns, cv_errors1, label="CV: phi1", marker='o')
    plt.loglog(Ns, cv_errors2, label="CV: phi2", marker='o')
    plt.loglog(Ns, cv_errors3, label="CV: phi3", marker='o')
    plt.loglog(Ns, is_errors1, label="IS: Beta(5,1)", marker='o')
    plt.loglog(Ns, is_errors2, label="IS: Beta(0.5,0.5)", marker='o')
    plt.xlabel("Number of Samples (log scale)")
    plt.ylabel("Absolute Error (log scale)")
    plt.title("Monte Carlo Integration Error Comparison")
    plt.grid(True, which="both", linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()
