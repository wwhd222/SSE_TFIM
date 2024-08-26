import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

from qmc_tfim import TFIM, BinaryThermalState, mc_step_beta
from qmc_tfim.lattice import lattice_bond_spins
from qmc_tfim.measurements import calculate_energy
from qmc_tfim.error import mean_and_stderr, Measurement

class Measurement:
    def __init__(self, value: float, error: float):
        self.value = value
        self.error = error

def custom_mean_and_stderr(data: np.ndarray) -> Measurement:
    """
    Calculate the mean and standard error of the given data.
    
    :param data: NumPy array of data points
    :return: Measurement object with mean as value and standard error as error
    """
    if len(data) < 2:
        print("Warning: Not enough data points for error estimation.")
        return Measurement(np.mean(data), np.nan)
    
    mean = np.mean(data)
    
    if np.all(data == data[0]):
        print("Warning: All data points are identical.")
        return Measurement(mean, 0.0)
    
    # Calculate sample standard deviation
    std = np.std(data, ddof=1)
    
    # Calculate standard error
    stderr = std / np.sqrt(len(data))
    
    return Measurement(mean, stderr)

def calculate_energy_stats(ns: np.ndarray, beta: float, J: float, h: float, Ns: int, Nb: int) -> Measurement:
    """
    Calculate energy statistics from the number of operators.
    
    :param ns: Array of number of operators from Monte Carlo steps
    :param beta: Inverse temperature
    :param J: Interaction strength
    :param h: Transverse field strength
    :param Ns: Number of spins
    :param Nb: Number of bonds
    :return: Measurement object with mean energy and its standard error
    """
    energies = -ns / beta
    energies_per_site = energies / Ns
    return custom_mean_and_stderr(energies_per_site)

def adjust_cutoff(qmc_state, num_operators, step):
    num_operators = int(num_operators)
    new_m = num_operators + num_operators // 3
    old_length = len(qmc_state.operator_list)
    new_length = 2 * new_m

    if new_length <= old_length:
        return False

    length_diff = new_length - old_length
    qmc_state.operator_list.extend([(0, 0)] * length_diff)
    
    new_array_size = int(new_length * 2)
    qmc_state.linked_list = np.zeros(new_array_size, dtype=int)
    qmc_state.leg_types = np.zeros(new_array_size, dtype=bool)
    qmc_state.associates = [(0, 0, 0) for _ in range(new_array_size)]

    return True

def run_simulation(L, J, h, beta, M, num_measurements, num_warmup, skip):
    bond_spin, Ns, Nb = lattice_bond_spins(L, True)
    H = TFIM(bond_spin, 1, Ns, Nb, h, J)
    
    operator_list = [(0, 0) for _ in range(2*M)]
    qmc_state = BinaryThermalState(H, 2*M, operator_list)
    
    for _ in tqdm(range(num_warmup), desc="Warm up", leave=False):
        num_ops = mc_step_beta(lambda *args: None, qmc_state, H, beta)
        adjust_cutoff(qmc_state, num_ops, _)
    
    ns = np.zeros(num_measurements)
    
    for i in tqdm(range(num_measurements), desc="MCMC", leave=False):
        ns[i] = mc_step_beta(lambda *args: None, qmc_state, H, beta)
        adjust_cutoff(qmc_state, ns[i], num_warmup + i)
        for _ in range(skip):
            mc_step_beta(lambda *args: None, qmc_state, H, beta)

    energy = calculate_energy_stats(ns, beta, J, h, H.nspins(), H.nbonds())
    return energy, len(qmc_state.operator_list) // 2

def process_gamma(args):
    L, gamma, beta, M, num_measurements, num_warmup, skip = args
    J = 1 / np.sqrt(gamma)
    h = np.sqrt(gamma)
    
    energy, final_m = run_simulation(L, J, h, beta, M, num_measurements, num_warmup, skip)
    return gamma, energy, final_m

def parameter_sweep_parallel(L, gamma_values, beta, M, num_measurements, num_warmup, skip):
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    
    args = [(L, gamma, beta, M, num_measurements, num_warmup, skip) for gamma in gamma_values]
    
    results = list(tqdm(pool.imap(process_gamma, args), total=len(gamma_values), desc="Processing γ values"))
    
    pool.close()
    pool.join()
    
    return sorted(results, key=lambda x: x[0])

def save_results(filename, results):
    with open(filename, 'w') as f:
        f.write("gamma\tenergy\tenergy_err\tfinal_m\n")
        for gamma, energy, final_m in results:
            f.write(f"{gamma}\t{energy.value}\t{energy.error}\t{final_m}\n")

def main():
    L = 12
    gamma_values = np.linspace(0.1, 4, 4)
    beta = 1000
    M = 100
    num_measurements = 1000
    num_warmup = 500
    skip = 10

    results = parameter_sweep_parallel(L, gamma_values, beta, M, num_measurements, num_warmup, skip)
    
    save_results(f'energy_results_beta_{beta}_L_{L}.txt', results)

    gammas = [res[0] for res in results]
    energies = [res[1].value for res in results]
    energy_errs = [res[1].error for res in results]
    final_m_values = [res[2] for res in results]

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(gammas, energies, yerr=energy_errs, fmt='o-')
    plt.xlabel(r'$\gamma = \Omega/J$', fontsize=14)
    plt.ylabel('Energy per site', fontsize=14)
    plt.title(f'Energy vs $\Omega/J$ (β = {beta}, L = {L})', fontsize=16)
    
    plt.subplot(1, 2, 2)
    plt.plot(gammas, final_m_values, 'o-')
    plt.xlabel(r'$\gamma = \Omega/J$', fontsize=14)
    plt.ylabel('Final M Value', fontsize=14)
    plt.title('Final M Value vs $\Omega/J$', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'energy_results_beta_{beta}_L_{L}.png', dpi=300)
    print(f"Results plot saved as energy_results_beta_{beta}_L_{L}.png")

if __name__ == "__main__":
    main()
