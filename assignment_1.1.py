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
    mean, var = 0, 0
    
    n = len(array)
    
    ## sample mean = x̄ = sumi(Xi/N)
    for x in array :
        mean += x/n
    
    
    # sample variance = S^2 = n/n-1 * E[(X-E[X])^2] = (n/(n-1))* sumi(((Xi - x̄)^2)/n) 
    # (1/(n-1))* sumi(((Xi - x̄)^2))= sumi((Xi-x̄)^2/(n-1)) 
    for x in array: 
        var += ((x - mean)**2)/(n-1)
    
    # ====================================================================
    return mean, var


def numpy_compute(array: npt.NDArray, ddof: int = 0) -> tuple[float, float]:
    # TODO: compute the mean and the variance using numpy.
    # ====================================================================
    mean, var = 0, 1
    
    mean = np.mean(array)
    
    var = np.var(array, ddof=1)
    
    # ====================================================================
    return mean, var


if __name__ == "__main__":
    # TODO: load the grades from the file, compute the mean and the
    # variance using both implementations and report the results.
    # ====================================================================
    
    grades = load_grades('./data/G.txt')
    print(grades)
    
    print(python_compute(grades))
    
    print(numpy_compute(grades, len(grades)-1))
 
    pass
    # ====================================================================
