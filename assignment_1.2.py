import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def sample_normal(
    n_samples: int, mu_target: npt.NDArray, V_target: npt.NDArray, seed: int = 42
) -> npt.NDArray:
    """
    Generate `n_samples` draws from a multivariate normal N(mu_target, V_target),
    and return a (d × n_samples) array, where d = len(mu_target).
    """
    np.random.seed(seed)
    # np.random.multivariate_normal(...) returns shape (n_samples, d)
    raw = np.random.multivariate_normal(mu_target, V_target, size=n_samples)
    # transpose → shape becomes (d, n_samples)
    samples = raw.T
    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Given `samples` of shape (d, N), compute:
      - `mean`: 1D array of length d, where mean[j] = (1/N) * sum_{k=1}^N samples[j, k]
      - `covariance`: (d × d) matrix, where
          covariance[i,j] = (1/(N-1)) * sum_{k=1}^N (samples[i,k] - mean[i]) * (samples[j,k] - mean[j])
    Returns (mean, covariance).
    """
    d, N = samples.shape

    # 1) Compute sample mean, dimension‐wise
    mean = np.zeros(d)
    for j in range(d):
        for k in range(N):
            mean[j] += samples[j, k] / N

    # 2) Compute sample covariance (unbiased: divide by N-1)
    covariance = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            acc = 0.0
            for k in range(N):
                acc += (samples[i, k] - mean[i]) * (samples[j, k] - mean[j])
            covariance[i, j] = acc / (N - 1)

    return mean, covariance


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 1) Define the true parameters (bivariate normal).
    # -----------------------------------------------------------------------
    mu = np.array([-0.4, 1.1])               # true mean (d = 2)
    V = np.array([[2.0,  0.4],
                  [0.4,  1.0]])             # true covariance (2 × 2)

    d = mu.shape[0]  # = 2

    # The four sample‐sizes to test
    N_list = [10, 100, 1000, 10000]

    # -----------------------------------------------------------------------
    # 2) Generate Monte Carlo draws and store them in a dictionary.
    # -----------------------------------------------------------------------
    all_results: dict[int, dict[str, np.ndarray]] = {}
    for n in N_list:
        samp = sample_normal(n, mu, V, seed=65)
        # Store each batch in a sub‐dict
        all_results[n] = {"samples": samp}

    # -----------------------------------------------------------------------
    # 3) For each N, compute sample‐mean and sample‐covariance.
    # -----------------------------------------------------------------------
    for n in N_list:
        samples = all_results[n]["samples"]            # shape: (2, n)
        m_hat, V_hat = compute_moments(samples)
        all_results[n]["mean"] = m_hat
        all_results[n]["covariance"] = V_hat

    # -----------------------------------------------------------------------
    # 4) Compute absolute‐error arrays: one error for each coordinate of the mean,
    #    and a full 2×2 matrix of errors for the covariance.
    #    We will later extract just the first‐coordinate mean‐error,
    #    the (1,1) covariance‐error, and the (1,2) covariance‐error.
    # -----------------------------------------------------------------------
    for n in N_list:
        d = mu.shape[0]  # =2
        m_hat = all_results[n]["mean"]
        V_hat = all_results[n]["covariance"]

        # (a) mean‐absolute‐error (length = d)
        err_mean = np.zeros(d)
        for j in range(d):
            err_mean[j] = abs(m_hat[j] - mu[j])
        all_results[n]["abs_error_mean"] = err_mean

        # (b) covariance‐absolute‐error (d × d)
        err_cov = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                err_cov[i, j] = abs(V_hat[i, j] - V[i, j])
        all_results[n]["abs_error_covariance"] = err_cov

    # -----------------------------------------------------------------------
    # 5) Print out the absolute errors 
    # -----------------------------------------------------------------------
    print("\nAbsolute errors of the mean (each coordinate):")
    for n in N_list:
        print(f"  N = {n:5d} →", all_results[n]["abs_error_mean"])

    print("\nAbsolute errors of the covariance matrix (2×2):")
    for n in N_list:
        print(f"  N = {n:5d} →\n{all_results[n]['abs_error_covariance']}\n")

    # -----------------------------------------------------------------------
    # 6) Build lists of errors for:
    #      - the first coordinate of the mean: |m̂[0] − (−0.4)|
    #      - the (1,1) entry of covariance: |V̂[0,0] − 2.0|
    #      - the (1,2) entry of covariance: |V̂[0,1] − 0.4|
    # -----------------------------------------------------------------------
    emp_err_mu1   = [all_results[n]["abs_error_mean"][0]   for n in N_list]
    emp_err_cov11 = [all_results[n]["abs_error_covariance"][0, 0] for n in N_list]
    emp_err_cov12 = [all_results[n]["abs_error_covariance"][0, 1] for n in N_list]

    # -----------------------------------------------------------------------
    # 7) Theoretical RMSE for the mean‐estimator (first coordinate).
    #    We know Var(X1) = V[0,0] = 2. => σ_{X1} = sqrt(2).
    #    So RMSE(μ̂1) = sqrt(2 / N).
    # -----------------------------------------------------------------------
    theoretical_rmse_mu1 = [np.sqrt(V[0, 0] / n) for n in N_list]  # sqrt(2/N)

    # -----------------------------------------------------------------------
    # 8) Plot absolute errors vs. N on a log–log scale
    # -----------------------------------------------------------------------
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
    plt.xlabel("Number of samples $N$")
    plt.ylabel("Absolute error")
    plt.title("Monte Carlo Absolute Errors vs. $N$ (log–log)")
    plt.grid(which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6, 4))
    plt.loglog(
        N_list,
        theoretical_rmse_mu1,  
        "k--", 
        label=r"Theoretical $\sqrt{2/N}$"
    )
    plt.xlabel("Number of samples $N$")
    plt.ylabel(r"Error in $\mu_1$")
    plt.title(r"Mean Estimator: Theoretical RMSE (log–log)")
    plt.grid(which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

