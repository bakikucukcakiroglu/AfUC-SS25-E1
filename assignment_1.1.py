import numpy as np
import numpy.typing as npt

def load_grades(filename: str) -> npt.NDArray:
    # TODO: read grades from the file.
    # ====================================================================
    grades = np.loadtxt(filename)
    # ====================================================================
    return grades

def python_compute(array: npt.NDArray) -> tuple[float, float]:
    # TODO: compute the mean and the variance using standard Python.
    # ====================================================================
    mean, var = 0.0, 0.0
    n = len(array)
    
    # sample mean = x̄ = sum_i(Xi)/N
    for x in array:
        mean += x
    mean /= n
    
    # sample variance = S^2 = (1/(n-1)) * sum_i((Xi - x̄)^2)
    for x in array:
        var += (x - mean) ** 2
    var /= (n - 1)
    # ====================================================================
    return mean, var

def numpy_compute(array: npt.NDArray, ddof: int = 0) -> tuple[float, float]:
    # TODO: compute the mean and the variance using numpy.
    # ====================================================================
    mean, var = 0.0, 0.0
    
    # np.mean computes the mean
    mean = np.mean(array)
    
    # np.var computes the variance; ddof=1 for sample variance
    var = np.var(array, ddof=ddof)
    # ====================================================================
    return mean, var

if __name__ == "__main__":
    # TODO: load the grades from the file, compute the mean and the
    # variance using both implementations and report the results.
    # ====================================================================
    grades = load_grades('./data/G.txt')
    
    mean_py, var_py = python_compute(grades)
    print(f"Python Computation -> Mean: {mean_py}, Variance: {var_py}")
    
    mean_np, var_np = numpy_compute(grades)
    print(f"NumPy Computation  -> Mean: {mean_np}, Variance: {var_np}")

    # ====================================================================
