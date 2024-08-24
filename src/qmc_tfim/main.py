# main.py

import argparse
import numpy as np
import os
import json
from tqdm import tqdm
from pathlib import Path

from qmc_tfim.lattice import lattice_bond_spins
from qmc_tfim.hamiltonian import TFIM
from qmc_tfim.qmc_state import BinaryGroundState, BinaryThermalState
from qmc_tfim.updates import mc_step, mc_step_beta
from qmc_tfim.measurements import sample, magnetization, num_single_site_diag, correlation_time
from qmc_tfim.error import mean_and_stderr, jackknife

def init_mc_cli(args):
    PBC = args.periodic
    h = args.field
    J = args.interaction

    Dim = len(args.dims)
    nX = args.dims

    BC_name = "PBC" if PBC else "OBC"

    if Dim == 1:
        nX = nX[0]
        bond_spin, Ns, Nb = lattice_bond_spins(nX, PBC)
    elif Dim == 2:
        bond_spin, Ns, Nb = lattice_bond_spins(nX, PBC)
        nX = nX[0]
    else:
        raise ValueError("Unsupported number of dimensions")

    # MC parameters
    M = args.M  # length of the operator_list is 2M
    MCS = args.measurements  # the number of samples to record
    EQ_MCS = MCS // 10
    skip = args.skip  # number of MC steps to perform between each msmt

    d = {
        "Dim": Dim,
        "nX": nX,
        "BC": BC_name,
        "J": J,
        "h": h,
        "skip": skip,
        "M": M
    }
    mc_opts = (M, MCS, EQ_MCS, skip)

    H = TFIM(bond_spin, Dim, Ns, Nb, h, J)
    if hasattr(args, 'beta'):
        qmc_state = BinaryThermalState(H, 2*M)
    else:
        qmc_state = BinaryGroundState(H, M)

    return H, qmc_state, json.dumps(d, sort_keys=True), mc_opts

def make_info_file(info_file, samples_file, mc_opts, op_list_length, observables, corr_time):
    M, MCS, EQ_MCS, skip = mc_opts
    mag, abs_mag, mag_sqr, energy = observables

    with open(info_file, "w") as f:
        print(f"⟨M⟩   = {mag.value:16f} +/- {mag.error:16f}", file=f)
        print(f"⟨|M|⟩ = {abs_mag.value:16f} +/- {abs_mag.error:16f}", file=f)
        print(f"⟨M^2⟩ = {mag_sqr.value:16f} +/- {mag_sqr.error:16f}", file=f)
        print(f"⟨E⟩   = {energy.value:16f} +/- {energy.error:16f}\n", file=f)

        print(f"Correlation time: {corr_time}\n", file=f)

        print(f"Initial Operator list length: {2*M}", file=f)
        print(f"Final Operator list length: {op_list_length}", file=f)
        print(f"Number of MC measurements: {MCS}", file=f)
        print(f"Number of equilibration steps: {EQ_MCS}", file=f)
        print(f"Number of skips between measurements: {skip}\n", file=f)

        print(f"Samples outputted to file: {samples_file}", file=f)

def save_data(path, mc_opts, qmc_state, measurements, observables, corr_time):
    info_file = path + "_info.txt"
    samples_file = path + "_samples.txt"
    qmc_state_file = path + "_state.json"

    np.savetxt(samples_file, measurements, fmt='%d', delimiter=' ')

    with open(qmc_state_file, 'w') as f:
        json.dump(qmc_state.__dict__, f)

    M = len(qmc_state.operator_list)

    make_info_file(info_file, samples_file, mc_opts, M, observables, corr_time)

def mixedstate(args):
    H, qmc_state, sname, mc_opts = init_mc_cli(args)
    beta = args.beta

    M, MCS, EQ_MCS, skip = mc_opts

    path = os.path.join("data", "sims", "mixedstate", f"{sname}_beta{beta:.4f}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    measurements = np.zeros((MCS, H.nspins()), dtype=int)
    mags = np.zeros(MCS)
    ns = np.zeros(MCS)

    max_ns = max(tqdm([mc_step_beta(qmc_state, H, beta, eq=True) for _ in range(EQ_MCS)], desc="Warm up"))

    qmc_state.resize_op_list(int(1.5 * max_ns + 0.5))

    for i in tqdm(range(MCS), desc="MCMC"):
        def measurement(cluster_data, qmc_state, H):
            spin_prop = qmc_state.left_config
            measurements[i, :] = spin_prop
            mags[i] = magnetization(spin_prop)
        
        ns[i] = mc_step_beta(measurement, qmc_state, H, beta)

        for _ in range(skip):
            mc_step_beta(lambda *args: None, qmc_state, H, beta)

    mag = mean_and_stderr(mags)
    abs_mag = mean_and_stderr(np.abs(mags))
    mag_sqr = mean_and_stderr(np.square(mags))

    energy = mean_and_stderr(lambda x: -x/beta, ns)
    energy.value += H.J * H.nbonds() + H.h * H.nspins()
    energy.value /= H.nspins()
    energy.error /= H.nspins()

    observables = (mag, abs_mag, mag_sqr, energy)

    corr_time = correlation_time(np.square(mags))

    save_data(path, mc_opts, qmc_state, measurements, observables, corr_time)

def groundstate(args):
    H, qmc_state, sname, mc_opts = init_mc_cli(args)

    M, MCS, EQ_MCS, skip = mc_opts

    path = os.path.join("data", "sims", "groundstate", sname)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    measurements = np.zeros((MCS, H.nspins()), dtype=int)
    mags = np.zeros(MCS)
    ns = np.zeros(MCS)

    for _ in tqdm(range(EQ_MCS), desc="Warm up"):
        mc_step(lambda *args: None, qmc_state, H)

    for i in tqdm(range(MCS), desc="MCMC"):
        def measurement(cluster_data, qmc_state, H):
            spin_prop = sample(qmc_state)
            measurements[i, :] = spin_prop
            ns[i] = num_single_site_diag(qmc_state.operator_list)
            mags[i] = magnetization(spin_prop)
        
        mc_step(measurement, qmc_state, H)

        for _ in range(skip):
            mc_step(lambda *args: None, qmc_state, H)

    mag = mean_and_stderr(mags)
    abs_mag = mean_and_stderr(np.abs(mags))
    mag_sqr = mean_and_stderr(np.square(mags))

    def energy_func(n):
        if H.h != 0:
            return (-H.h * ((1.0 / n) - 1)) + H.J * (H.nbonds() / H.nspins())
        else:
            return H.J * (H.nbonds() / H.nspins())
    
    energy = jackknife(energy_func, ns)

    observables = (mag, abs_mag, mag_sqr, energy)

    corr_time = correlation_time(np.square(mags))

    save_data(path, mc_opts, qmc_state, measurements, observables, corr_time)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # Groundstate parser
    parser_gs = subparsers.add_parser('groundstate', help='Use Projector SSE to simulate the ground state')
    parser_gs.add_argument('dims', type=int, nargs='+', help='The dimensions of the square lattice')
    parser_gs.add_argument('--periodic', '-p', action='store_true', help='Periodic BCs')
    parser_gs.add_argument('--field', type=float, default=1.0, help='Strength of the transverse field')
    parser_gs.add_argument('--interaction', '-J', type=float, default=1.0, help='Strength of the interaction')
    parser_gs.add_argument('-M', type=int, default=1000, help='Half-size of the operator list')
    parser_gs.add_argument('--measurements', '-n', type=int, default=100000, help='Number of samples to record')
    parser_gs.add_argument('--skip', '-s', type=int, default=0, help='Number of MC steps to perform between each measurement')
    parser_gs.set_defaults(func=groundstate)

    # Mixedstate parser
    parser_ms = subparsers.add_parser('mixedstate', help='Use vanilla SSE to simulate the system at non-zero temperature')
    parser_ms.add_argument('dims', type=int, nargs='+', help='The dimensions of the square lattice')
    parser_ms.add_argument('--periodic', '-p', action='store_true', help='Periodic BCs')
    parser_ms.add_argument('--field', type=float, default=1.0, help='Strength of the transverse field')
    parser_ms.add_argument('--interaction', '-J', type=float, default=1.0, help='Strength of the interaction')
    parser_ms.add_argument('-M', type=int, default=1000, help='Half-size of the operator list')
    parser_ms.add_argument('--measurements', '-n', type=int, default=100000, help='Number of samples to record')
    parser_ms.add_argument('--skip', '-s', type=int, default=0, help='Number of MC steps to perform between each measurement')
    parser_ms.add_argument('--beta', type=float, default=10.0, help='The inverse-temperature parameter for the simulation')
    parser_ms.set_defaults(func=mixedstate)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
