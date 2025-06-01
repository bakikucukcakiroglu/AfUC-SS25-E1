import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def sample_normal(
    n_samples: int, mu_target: npt.NDArray, V_target: npt.NDArray, seed: int = 42
) -> npt.NDArray:
    # TODO: generate samples from multivariate normal distribution.
    # ====================================================================
    np.random.seed(seed)
    # returns shape (n_samples, d)
    raw = np.random.multivariate_normal(mu_target, V_target, size=n_samples)
    # transpose → shape becomes (d, n_samples)
    samples = raw.T
    # ====================================================================

    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # TODO: estimate mean and covariance of the samples.
    # ====================================================================
    d, N = samples.shape

    #Compute sample mean
    mean = np.zeros(d)
    for j in  range(d):
        for k in range(N):
            mean[j] += samples[j, k] / N

    # Compute sample covariance (unbiased: divide by N-1)
    covariance = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            acc = 0.0
            for k in range(N):
                acc += (samples[i, k] - mean[i]) * (samples[j, k] - mean[j])
            covariance[i, j] = acc / (N - 1)

    # ====================================================================

    return mean, covariance


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    
    mu = np.array([-0.4, 1.1])               # true mean (d = 2)
    V = np.array([[2.0,  0.4],
                  [0.4,  1.0]])             # true covariance (2 × 2)

    d = mu.shape[0]  # = 2

    # The four sample‐sizes to test
    N_list = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

    # ====================================================================
    # TODO: estimate mean, covariance, and compute the required errors.
    # ====================================================================
    all_results: dict[int, dict[str, np.ndarray]] = {}
    for n in N_list:
        samp = sample_normal(n, mu, V, seed=69)
        
        all_results[n] = {"samples": samp}

    for n in N_list:
        samples = all_results[n]["samples"]           
        m_hat, V_hat = compute_moments(samples)
        all_results[n]["mean"] = m_hat
        all_results[n]["covariance"] = V_hat

    for n in N_list:
        d = mu.shape[0]  # =2
        m_hat = all_results[n]["mean"]
        V_hat = all_results[n]["covariance"]

        err_mean = np.zeros(d)
        for j in range(d):
            err_mean[j] = abs(m_hat[j] - mu[j])
        all_results[n]["abs_error_mean"] = err_mean

        err_cov = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                err_cov[i, j] = abs(V_hat[i, j] - V[i, j])
        all_results[n]["abs_error_covariance"] = err_cov

     # ====================================================================
    # TODO: plot the results on the log-log scale.
    # ====================================================================
    print("\nMean (each coordinate):")
    for n in N_list:
        print(f"  N = {n:5d} →", all_results[n]["mean"])
        
    print("\nCovariance (each coordinate):")
    for n in N_list:
        print(f"  N = {n:5d} →", all_results[n]["covariance"])
    
    print("\nAbsolute errors of the mean (each coordinate):")
    for n in N_list:
        print(f"  N = {n:5d} →", all_results[n]["abs_error_mean"])

    print("\nAbsolute errors of the covariance matrix (2×2):")
    for n in N_list:
        print(f"  N = {n:5d} →\n{all_results[n]['abs_error_covariance']}\n")

    emp_err_mu1   = [all_results[n]["abs_error_mean"][0]   for n in N_list]
    emp_err_cov11 = [all_results[n]["abs_error_covariance"][0, 0] for n in N_list]
    emp_err_cov12 = [all_results[n]["abs_error_covariance"][0, 1] for n in N_list]

    theoretical_rmse_mu1 = [np.sqrt(V[0, 0] / n) for n in N_list]  


    plt.figure(figsize=(6, 4))
    plt.loglog(
        N_list,
        emp_err_mu1,   
        "o-", 
        label=r"$|\hat\mu_1 - (-0.4)|$"
    )
    plt.loglog(
        N_list,
        emp_err_cov11, 
        "s--", 
        label=r"$|\hat V_{11} - 2|$"
    )
    plt.loglog(
        N_list,
        emp_err_cov12, 
        "x-.", 
        label=r"$|\hat V_{12} - 0.4|$"
    )
    plt.loglog(
        N_list,
        theoretical_rmse_mu1,  
        "k--", 
        label=r"Theoretical $\sqrt{2/N}$"
    )
    plt.xlabel("Number of samples $N$")
    plt.ylabel("Absolute error")
    plt.title("Monte Carlo Absolute Errors vs. $N$ (log–log)")
    plt.grid(which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


