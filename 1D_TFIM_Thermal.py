import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy import stats
from qmc_tfim import TFIM, BinaryThermalState, mc_step_beta, magnetization
from qmc_tfim.lattice import lattice_bond_spins
from qmc_tfim.error import mean_and_stderr, Measurement

def run_simulation(L, J, h, beta, M, num_measurements, num_warmup):
    #print(f"Starting simulation with L={L}, J={J}, h={h}, beta={beta}, M={M}")
    
    bond_spin, Ns, Nb = lattice_bond_spins(L, True)  # True for periodic boundary conditions
    #print(f"Lattice created with Ns={Ns}, Nb={Nb}")
    
    H = TFIM(bond_spin, 1, Ns, Nb, h, J)  # 1 for 1D
    #print(f"Hamiltonian created")
    
    operator_list = [(0, 0) for _ in range(2*M)]
    #print(f"Initial operator list created with length {len(operator_list)}")
    
    qmc_state = BinaryThermalState(H, 2*M, operator_list)
    #print(f"QMC state initialized")
    
    #print("Starting warm-up")
    for i in range(num_warmup):
        #print(f"Warm-up step {i+1}/{num_warmup}")
        mc_step_beta(lambda *args: None, qmc_state, H, beta)
    
    #print("Starting measurements")
    energies = []
    mags = []
    for i in range(num_measurements):
        #print(f"Measurement {i+1}/{num_measurements}")
        num_ops = mc_step_beta(lambda cd, qs, H: mags.append(magnetization(qs.left_config)), qmc_state, H, beta)
        energies.append(-num_ops / (beta * Ns) + H.J * Nb / Ns + H.h)
    
    if energies:  # ensure energies is not empty
        energy_measurement = mean_and_stderr(lambda x: x, np.array(energies))
        mag_measurement = mean_and_stderr(np.abs, np.array(mags))
        
        avg_energy, std_energy = energy_measurement.value, energy_measurement.error
        avg_mag, std_mag = mag_measurement.value, mag_measurement.error
    else:
        #print("Warning: No energy measurements were made.")
        avg_energy, std_energy = 0, 0
        avg_mag, std_mag = 0, 0
    
    #print(f"Simulation completed. Average energy: {avg_energy}, Average magnetization: {avg_mag}")
    
    return avg_energy, std_energy, avg_mag, std_mag

def parameter_sweep(L: int, gamma_values: np.ndarray, beta: float, M: int, num_measurements: int, num_warmup: int):
    results = []
    
    for gamma in gamma_values:
        J = 1 / np.sqrt(gamma)  # Ising Coupling
        h = np.sqrt(gamma)  # Transverse field
        
        print(f"Running simulation for gamma = {gamma:.4f}, J = {J:.4f}, h = {h:.4f}")
        
        energy, energy_err, mag, mag_err = run_simulation(L, J, h, beta, M, num_measurements, num_warmup)
        results.append((gamma, energy, energy_err, mag, mag_err))
    
    return results

def main():
    L = 50  # Size of the lattice, here 1D
    gamma_values = np.linspace(0.1, 3, 20)
    beta = 10000  # reverse temperature
    M = 10000  # cut-off M
    num_measurements = 10000  # steps of measuring
    num_warmup = 1000  # steps of heating

    results = parameter_sweep(L,gamma_values, beta, M, num_measurements, num_warmup)

    gammas, energies, energy_errs, mags, mag_errs = zip(*results)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(gammas, energies, yerr=energy_errs, fmt='o-')
    plt.xlabel(r'$\gamma = \Omega/J$', fontsize=18)
    plt.ylabel('Energy per site', fontsize=18)
    plt.title(f'Energy vs h/J (β = {beta}, L = {L})', fontsize=18)
    
    plt.subplot(1, 2, 2)
    plt.errorbar(gammas, mags, yerr=mag_errs, fmt='o-')
    plt.xlabel(r'$\gamma = \Omega/J$', fontsize=18)
    plt.ylabel('|Magnetization|', fontsize=18)
    plt.title(f'|Magnetization| vs h/J (β = {beta}, L = {L})', fontsize=18)
    
    plt.tight_layout()
    plt.savefig(f'results_beta_{beta}_L_{L}.png',dpi=600)
    print(f"Results plot saved as results_beta_{beta}_L_{L}.png")

if __name__ == "__main__":
    main()
