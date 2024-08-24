# measurements.py

import numpy as np
from numpy.fft import fft, ifft
from typing import List, Tuple

from updates import issiteoperator, isdiagonal

def sample(qmc_state):
    operator_list = qmc_state.operator_list
    M = len(operator_list) // 2
    spin_prop = qmc_state.left_config.copy()

    for op in operator_list[:M]:  # propagate half the list only (to the middle)
        if issiteoperator(op) and not isdiagonal(op):
            spin_prop[op[1]] ^= 1  # spinflip
    return spin_prop

def simulation_cell(qmc_state):
    operator_list = qmc_state.operator_list
    cell = np.zeros((len(qmc_state.left_config), len(operator_list)), dtype=bool)
    spin_prop = qmc_state.left_config.copy()

    for n, op in enumerate(operator_list):
        if issiteoperator(op) and not isdiagonal(op):
            spin_prop[op[1]] ^= 1  # spinflip
        cell[:, n] = spin_prop
    return cell

def magnetization(spin_prop):
    return np.mean(2 * spin_prop - 1)

def num_single_site_diag(operator_list):
    return np.mean([issiteoperator(x) and isdiagonal(x) for x in operator_list])

def num_single_site_offdiag(operator_list):
    return np.mean([issiteoperator(x) and not isdiagonal(x) for x in operator_list])

def num_single_site(operator_list):
    return np.mean([issiteoperator(x) for x in operator_list])

def autocorrelation(m: np.ndarray) -> np.ndarray:
    N = len(m)
    m_prime = m - np.mean(m)
    m_prime = np.concatenate([m_prime, np.zeros(N)])
    mw = fft(m_prime)
    s = np.abs(mw)**2

    chi = np.real(ifft(s)[:N])
    chi /= (2 * N)  # normalize FFT
    chi /= np.arange(N, 0, -1) - 1
    return chi

def correlation_time(m: np.ndarray) -> float:
    ac = autocorrelation(m)
    ac_0 = ac[0]

    corr_time = 0.0
    for M in range(len(ac)):
        corr_time += (ac[M] / ac_0)
        if M >= 10 * corr_time:
            break

    return corr_time
