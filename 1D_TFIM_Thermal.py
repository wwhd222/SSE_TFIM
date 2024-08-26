import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable
from scipy import stats

from qmc_tfim import TFIM, BinaryThermalState, mc_step_beta, magnetization
from qmc_tfim.lattice import lattice_bond_spins
from qmc_tfim.error import mean_and_stderr, Measurement

def adjust_cutoff(qmc_state, num_operators, step):
    new_m = num_operators + num_operators // 3
    if new_m <= len(qmc_state.operator_list) // 2:
        return False

    old_length = len(qmc_state.operator_list)
    new_length = 2 * new_m
    qmc_state.operator_list.extend([(0, 0)] * (new_length - old_length))
    
    # Adjust the size of other related arrays
    qmc_state.linked_list = np.zeros(new_length * 2, dtype=int)
    qmc_state.leg_types = np.zeros(new_length * 2, dtype=bool)
    qmc_state.associates = [(0, 0, 0) for _ in range(new_length * 2)]

    print(f" Step: {step}  New cutoff M: {new_m}")
    return True

def run_simulation(L, J, h, beta, M, num_measurements, num_warmup, skip):
    bond_spin, Ns, Nb = lattice_bond_spins(L, True)  # True for periodic boundary conditions
    H = TFIM(bond_spin, 1, Ns, Nb, h, J)  # 1 for 1D
    
    operator_list = [(0, 0) for _ in range(2*M)]
    qmc_state = BinaryThermalState(H, 2*M, operator_list)
    
    # Warm-up
    for step in tqdm(range(num_warmup), desc="Warm up", leave=False):
        num_ops = mc_step_beta(lambda *args: None, qmc_state, H, beta)
        adjust_cutoff(qmc_state, num_ops, step)
    
    measurements = np.zeros((num_measurements, Ns), dtype=int)
    mags = np.zeros(num_measurements)
    ns = np.zeros(num_measurements)
    
    for i in tqdm(range(num_measurements), desc="MCMC", leave=False):
        ns[i] = mc_step_beta(lambda cd, qs, H: 
                             (mags.__setitem__(i, magnetization(qs.left_config)),
                              measurements.__setitem__((i, slice(None)), qs.left_config)),
                             qmc_state, H, beta)
        
        adjust_cutoff(qmc_state, ns[i], num_warmup + i)
        
        for _ in range(skip):
            mc_step_beta(lambda *args: None, qmc_state, H, beta)
    
    mag = mean_and_stderr(lambda x: x, mags)
    abs_mag = mean_and_stderr(np.abs, mags)
    mag_sqr = mean_and_stderr(np.square, mags)
    
    energy = mean_and_stderr(lambda x: -x/beta, ns)
    energy.value += H.J * Nb + H.h * Ns
    energy.value /= Ns
    energy.error /= Ns
    
    return mag, abs_mag, mag_sqr, energy, measurements, len(qmc_state.operator_list) // 2

def parameter_sweep(L: int, gamma_values: np.ndarray, beta: float, M: int, num_measurements: int, num_warmup: int, skip: int):
    results = []
    
    for gamma in tqdm(gamma_values, desc="Processing γ values"):
        J = -1 / np.sqrt(gamma)  # Ising Coupling
        h = np.sqrt(gamma)  # Transverse field
        
        print(f"Running simulation for gamma = {gamma:.4f}, J = {J:.4f}, h = {h:.4f}")
        
        mag, abs_mag, mag_sqr, energy, measurements, final_m = run_simulation(L, J, h, beta, M, num_measurements, num_warmup, skip)
        results.append((gamma, energy, mag, abs_mag, mag_sqr, measurements, final_m))
    
    return results

def save_results(filename, results):
    with open(filename, 'w') as f:
        f.write("gamma\tenergy\tenergy_err\tmag\tmag_err\tabs_mag\tabs_mag_err\tmag_sqr\tmag_sqr_err\tfinal_m\n")
        for res in results:
            gamma, energy, mag, abs_mag, mag_sqr, _, final_m = res
            f.write(f"{gamma}\t{energy.value}\t{energy.error}\t{mag.value}\t{mag.error}\t"
                    f"{abs_mag.value}\t{abs_mag.error}\t{mag_sqr.value}\t{mag_sqr.error}\t{final_m}\n")

def main():
    L = 50  # Size of the lattice, here 1D
    gamma_values = np.linspace(0.1, 3, 20)
    beta = 100  # reverse temperature
    M = 20  # initial cut-off M
    num_measurements = 5000  # steps of measuring
    num_warmup = 500  # steps of heating
    skip = 0  # steps to skip between measurements

    results = parameter_sweep(L, gamma_values, beta, M, num_measurements, num_warmup, skip)
    
    save_results(f'results_beta_{beta}_L_{L}.txt', results)

    # Extract data for plotting
    gammas = [res[0] for res in results]
    energies = [res[1].value for res in results]
    energy_errs = [res[1].error for res in results]
    abs_mags = [res[3].value for res in results]
    abs_mag_errs = [res[3].error for res in results]
    final_m_values = [res[6] for res in results]

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.errorbar(gammas, energies, yerr=energy_errs, fmt='o-')
    ax1.set_xlabel(r'$\gamma = \Omega/J$', fontsize=18)
    ax1.set_ylabel('Energy per site', fontsize=18)
    ax1.set_title(f'Energy vs $\Omega$/J (β = {beta}, L = {L})', fontsize=18)
    
    ax2.errorbar(gammas, abs_mags, yerr=abs_mag_errs, fmt='o-')
    ax2.set_xlabel(r'$\gamma = \Omega/J$', fontsize=18)
    ax2.set_ylabel('|Magnetization|', fontsize=18)
    ax2.set_title(f'|Magnetization| vs $\Omega$/J (β = {beta}, L = {L})', fontsize=18)
    
    ax3.plot(gammas, final_m_values, 'o-')
    ax3.set_xlabel(r'$\gamma = \Omega/J$', fontsize=18)
    ax3.set_ylabel('Final M Value', fontsize=18)
    ax3.set_title('Final M Value vs $\Omega$/J', fontsize=18)
    
    plt.tight_layout()
    plt.savefig(f'results_beta_{beta}_L_{L}.png', dpi=600)
    print(f"Results plot saved as results_beta_{beta}_L_{L}.png")

if __name__ == "__main__":
    main()
