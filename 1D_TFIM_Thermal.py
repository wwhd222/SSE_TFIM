import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable
from scipy import stats
import multiprocessing as mp

from qmc_tfim import TFIM, BinaryThermalState, mc_step_beta
from qmc_tfim.lattice import lattice_bond_spins
from qmc_tfim.measurements import calculate_energy
from qmc_tfim.error import mean_and_stderr, Measurement

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
    
    for step in tqdm(range(num_warmup), desc="Warm up", leave=False):
        num_ops = mc_step_beta(lambda *args: None, qmc_state, H, beta)
        adjust_cutoff(qmc_state, num_ops, step)
    
    ns = np.zeros(num_measurements)
    
    for i in tqdm(range(num_measurements), desc="MCMC", leave=False):
        ns[i] = int(mc_step_beta(lambda *args: None, qmc_state, H, beta))
        
        adjust_cutoff(qmc_state, ns[i], num_warmup + i)
        
        for _ in range(skip):
            mc_step_beta(lambda *args: None, qmc_state, H, beta)
    
    energy = mean_and_stderr(lambda x: calculate_energy(x, beta, J, h, Ns, Nb), ns)
    
    return energy, len(qmc_state.operator_list) // 2

def save_results(filename, results):
    with open(filename, 'w') as f:
        f.write("gamma\tenergy\tenergy_err\tfinal_m\n")
        for res in results:
            gamma, energy, final_m = res
            f.write(f"{gamma}\t{energy.value}\t{energy.error}\t{final_m}\n")
            
def process_gamma(args):
    L, gamma, beta, M, num_measurements, num_warmup, skip = args
    J = -1 / np.sqrt(gamma)
    h = np.sqrt(gamma)
    
    energy, final_m = run_simulation(L, J, h, beta, M, num_measurements, num_warmup, skip)
    return gamma, energy, final_m

def parameter_sweep_parallel(L: int, gamma_values: np.ndarray, beta: float, M: int, num_measurements: int, num_warmup: int, skip: int):
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    
    args = [(L, gamma, beta, M, num_measurements, num_warmup, skip) for gamma in gamma_values]
    
    results = list(tqdm(pool.imap(process_gamma, args), total=len(gamma_values), desc="Processing γ values"))
    
    pool.close()
    pool.join()
    
    return sorted(results, key=lambda x: x[0])

def main():
    L = 50
    gamma_values = np.linspace(0.1, 4, 40)
    beta = 1000
    M = 20
    num_measurements = 1000
    num_warmup = 1000
    skip = 0

    results = parameter_sweep_parallel(L, gamma_values, beta, M, num_measurements, num_warmup, skip)
    
    save_results(f'results_beta_{beta}_L_{L}.txt', results)

    gammas = [res[0] for res in results]
    energies = [res[1].value for res in results]
    energy_errs = [res[1].error for res in results]
    final_m_values = [res[2] for res in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.errorbar(gammas, energies, yerr=energy_errs, fmt='o-')
    ax1.set_xlabel(r'$\gamma = \Omega/J$', fontsize=18)
    ax1.set_ylabel('Energy per site', fontsize=18)
    ax1.set_title(f'Energy vs $\Omega$/J (β = {beta}, L = {L})', fontsize=18)
    
    ax2.plot(gammas, final_m_values, 'o-')
    ax2.set_xlabel(r'$\gamma = \Omega/J$', fontsize=18)
    ax2.set_ylabel('Final M Value', fontsize=18)
    ax2.set_title('Final M Value vs $\Omega$/J', fontsize=18)
    
    plt.tight_layout()
    plt.savefig(f'results_beta_{beta}_L_{L}.png', dpi=600)
    print(f"Results plot saved as results_beta_{beta}_L_{L}.png")

if __name__ == "__main__":
    main()
