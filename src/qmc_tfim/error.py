# error.py

import numpy as np
from typing import Callable, List, Union, Tuple
from scipy import stats

class Measurement:
    def __init__(self, value: float, error: float):
        self.value = value
        self.error = error

    def __repr__(self):
        return f"{self.value} ± {self.error}"

def mean_and_stderr(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> Measurement:
    y = f(x)
    
    mu = np.mean(y)
    sigma = stats.sem(y)  # Standard Error of the Mean
    
    return Measurement(mu, sigma)

def mean_and_stderr_identity(x: np.ndarray) -> Measurement:
    return mean_and_stderr(lambda x: x, x)

def jackknife(f: Callable[..., float], *x: np.ndarray) -> Measurement:
    sum_x = [np.sum(xi) for xi in x]
    N = len(x[0])
    
    f_J = np.zeros(N)
    for i in range(N):
        x_J = [(sum_xi - xi[i]) / (N - 1) for sum_xi, xi in zip(sum_x, x)]
        f_J[i] = f(*x_J)
    
    mu = np.mean(f_J)
    sigma = np.sqrt(N - 1) * np.std(f_J, ddof=1)
    
    return Measurement(mu, sigma)

def jackknife_identity(x: np.ndarray) -> Measurement:
    return jackknife(lambda x: x, x)
