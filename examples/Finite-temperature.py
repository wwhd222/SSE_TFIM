import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
import os

def run_simulation(dims, field, interaction, beta, M, measurements, skip):
    cmd = [
        "python", "run_simulation.py", "mixedstate",
        "--dims", *map(str, dims),
        "--field", str(field),
        "--interaction", str(interaction),
        "--beta", str(beta),
        "-M", str(M),
        "--measurements", str(measurements),
        "--skip", str(skip)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def extract_data(output):
    energy_match = re.search(r"⟨E⟩\s*=\s*([-\d.]+)\s*\+/-\s*([-\d.]+)", output)
    mag_match = re.search(r"⟨\|M\|⟩\s*=\s*([-\d.]+)\s*\+/-\s*([-\d.]+)", output)
    
    energy = float(energy_match.group(1)) if energy_match else None
    energy_err = float(energy_match.group(2)) if energy_match else None
    mag = float(mag_match.group(1)) if mag_match else None
    mag_err = float(mag_match.group(2)) if mag_match else None
    
    return energy, energy_err, mag, mag_err

def parameter_sweep(dims, J, h_range, beta, M, measurements, skip):
    results = []
    for h in h_range:
        print(f"Running simulation for h/J = {h/J:.2f}")
        output = run_simulation(dims, h, J, beta, M, measurements, skip)
        energy, energy_err, mag, mag_err = extract_data(output)
        results.append((h/J, energy, energy_err, mag, mag_err))
    return results

def plot_results(results, beta):
    h_J_ratios, energies, energy_errs, mags, mag_errs = zip(*results)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(h_J_ratios, energies, yerr=energy_errs, fmt='o-')
    plt.xlabel('h/J')
    plt.ylabel('Energy')
    plt.title(f'Energy vs h/J (β = {beta})')
    
    plt.subplot(1, 2, 2)
    plt.errorbar(h_J_ratios, mags, yerr=mag_errs, fmt='o-')
    plt.xlabel('h/J')
    plt.ylabel('|Magnetization|')
    plt.title(f'|Magnetization| vs h/J (β = {beta})')
    
    plt.tight_layout()
    plt.savefig(f'results_beta_{beta}.png')
    print(f"Results plot saved as results_beta_{beta}.png")

def main():
    # Simulation parameters
    dims = [10, 10]  # 10x10 lattice
    J = 1.0  # Fixed interaction strength
    h_range = np.linspace(0.1, 3.0, 20)  # Range of h values
    beta = 5.0  # Inverse temperature
    M = 2000  # Operator list size
    measurements = 10000  # Number of measurements
    skip = 10  # Steps to skip between measurements

    results = parameter_sweep(dims, J, h_range, beta, M, measurements, skip)
    
    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open(f'results/sweep_results_beta_{beta}.txt', 'w') as f:
        f.write("h/J\tEnergy\tEnergy_err\tMagnetization\tMagnetization_err\n")
        for result in results:
            f.write("\t".join(map(str, result)) + "\n")
    
    plot_results(results, beta)

if __name__ == "__main__":
    main()
