# updates.py

import numpy as np
from typing import Tuple, List, Callable
from collections import deque
from .hamiltonian import TFIM

# Helper functions for operator types
def isdiagonal(op: Tuple[int, int]) -> bool:
    return op[0] != -2

def isidentity(op: Tuple[int, int]) -> bool:
    return op[0] == 0

def issiteoperator(op: Tuple[int, int]) -> bool:
    return op[0] < 0

def isbondoperator(op: Tuple[int, int]) -> bool:
    return op[0] > 0

def mc_step(f: Callable, qmc_state, H):
    diagonal_update(qmc_state, H)
    cluster_data = linked_list_update(qmc_state, H)
    f(cluster_data, qmc_state, H)
    cluster_update(cluster_data, qmc_state, H)

def mc_step_beta(f: Callable, qmc_state, H, beta: float, eq: bool = False):
    num_ops = diagonal_update_beta(qmc_state, H, beta, eq=eq)
    cluster_data = linked_list_update_beta(qmc_state, H)
    f(cluster_data, qmc_state, H)
    cluster_update_beta(cluster_data, qmc_state, H)
    return num_ops

def insert_diagonal_operator(qmc_state, H, spin_prop, n):
    op = H.op_sampler.rand()
    site1, site2 = op
    if issiteoperator(op) or spin_prop[site1] == spin_prop[site2]:
        qmc_state.operator_list[n] = op
        return True
    return False

def diagonal_update(qmc_state, H):
    spin_prop = qmc_state.propagated_config = qmc_state.left_config.copy()

    for n, op in enumerate(qmc_state.operator_list):
        if not isdiagonal(op):
            spin_prop[op[1]] ^= 1  # spinflip
        else:
            while not insert_diagonal_operator(qmc_state, H, spin_prop, n):
                pass

def linked_list_update_beta(qmc_state, H):
    Ns = H.nspins()
    print(f"Number of spins (Ns): {Ns}")
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config

    # Calculate the length of the linked list
    len_list = sum(2 if issiteoperator(op) else 4 for op in qmc_state.operator_list if not isidentity(op))
    len_list += Ns  # Add extra space for safety
    print(f"Calculated len_list: {len_list}")

    # Ensure the arrays are large enough
    if len(qmc_state.linked_list) < len_list:
        qmc_state.linked_list = np.zeros(len_list, dtype=int)
        qmc_state.leg_types = np.zeros(len_list, dtype=bool)
        qmc_state.associates = [(0, 0, 0) for _ in range(len_list)]

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates

    First = qmc_state.first = np.zeros(Ns, dtype=int)
    Last = qmc_state.last = np.zeros(Ns, dtype=int)
    idx = 0

    spin_prop = qmc_state.propagated_config = spin_left.copy()

    print(f"Initial array sizes - LinkList: {len(LinkList)}, LegType: {len(LegType)}, Associates: {len(Associates)}, First: {len(First)}, Last: {len(Last)}")

    for op_idx, op in enumerate(qmc_state.operator_list):
        print(f"Processing operator {op_idx}: {op}")
        print(f"Current idx: {idx}")
        
        if idx >= len(LinkList) - 4:
            new_size = len(LinkList) * 2
            print(f"Resizing arrays to {new_size}")
            LinkList = np.resize(LinkList, new_size)
            LegType = np.resize(LegType, new_size)
            Associates.extend([(0, 0, 0)] * (new_size - len(Associates)))

        if issiteoperator(op):
            site = op[1]
            print(f"Site operator at site {site}")
            
            if site >= Ns:
                print(f"Warning: site {site} is out of bounds for Ns {Ns}. Skipping this operator.")
                continue

            LinkList[idx] = First[site]
            LegType[idx] = spin_prop[site]
            
            if not isdiagonal(op):
                spin_prop[site] ^= 1

            if First[site] != 0:
                LinkList[First[site] - 1] = idx + 1
            else:
                Last[site] = idx + 1
            
            First[site] = idx + 1
            Associates[idx] = (0, 0, 0)
            idx += 1

            LegType[idx] = spin_prop[site]
            Associates[idx] = (0, 0, 0)
            idx += 1

        elif isbondoperator(op):
            site1, site2 = op
            print(f"Bond operator at sites {site1} and {site2}")
            
            if site1 >= Ns or site2 >= Ns:
                print(f"Warning: sites {site1} or {site2} are out of bounds for Ns {Ns}. Skipping this operator.")
                continue

            # lower left
            LinkList[idx] = First[site1]
            LegType[idx] = spin_prop[site1]
            
            if First[site1] != 0:
                LinkList[First[site1] - 1] = idx + 1
            else:
                Last[site1] = idx + 1

            First[site1] = idx + 2
            vertex1 = idx
            Associates[idx] = (vertex1 + 1, vertex1 + 2, vertex1 + 3)
            idx += 1

            # lower right
            LinkList[idx] = First[site2]
            LegType[idx] = spin_prop[site2]
            
            if First[site2] != 0:
                LinkList[First[site2] - 1] = idx + 1
            else:
                Last[site2] = idx + 1

            First[site2] = idx + 2
            Associates[idx] = (vertex1, vertex1 + 2, vertex1 + 3)
            idx += 1

            # upper left
            LegType[idx] = spin_prop[site1]
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 3)
            idx += 1

            # upper right
            LegType[idx] = spin_prop[site2]
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 2)
            idx += 1

        print(f"After processing, idx: {idx}")
        print(f"Current LinkList size: {len(LinkList)}")

    # Periodic boundary conditions for finite-beta
    for i in range(Ns):
        if First[i] != 0:  # This might be encountered at high temperatures
            LinkList[First[i] - 1] = Last[i]
            LinkList[Last[i] - 1] = First[i]

    print(f"Final idx: {idx}")
    print(f"Final LinkList size: {len(LinkList)}")

    qmc_state.linked_list = LinkList[:idx]
    qmc_state.leg_types = LegType[:idx]
    qmc_state.associates = Associates[:idx]

    return idx

def cluster_update(lsize: int, qmc_state, H):
    Ns = H.nspins()
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    operator_list = qmc_state.operator_list

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates

    in_cluster = np.zeros(lsize, dtype=int)
    cstack = deque()
    ccount = 0

    for i in range(lsize):
        if in_cluster[i] == 0 and Associates[i] == (0, 0, 0):
            ccount += 1
            cstack.append(i)
            in_cluster[i] = ccount

            flip = np.random.random() < 0.5
            if flip:
                LegType[i] ^= 1  # spinflip

            while cstack:
                leg = LinkList[cstack.pop()]

                if in_cluster[leg] == 0:
                    in_cluster[leg] = ccount
                    if flip:
                        LegType[leg] ^= 1

                    assoc = Associates[leg]
                    if assoc != (0, 0, 0):
                        for a in assoc:
                            cstack.append(a)
                            in_cluster[a] = ccount
                            if flip:
                                LegType[a] ^= 1

    for i in range(Ns):
        spin_left[i] = LegType[i]
        spin_right[i] = LegType[lsize - Ns + i]

    ocount = Ns + 1
    for n, op in enumerate(operator_list):
        if isbondoperator(op):
            ocount += 4
        else:
            if LegType[ocount] == LegType[ocount + 1]:  # diagonal
                operator_list[n] = (-1, op[1])
            else:  # off-diagonal
                operator_list[n] = (-2, op[1])
            ocount += 2

# Additional functions for finite beta case
def resize_op_list(qmc_state, new_size: int):
    operator_list = [op for op in qmc_state.operator_list if not isidentity(op)]
    len_list = len(operator_list)

    if len_list < new_size:
        tail = [(0, 0) for _ in range(new_size - len_list)]
        operator_list.extend(tail)

    qmc_state.operator_list = operator_list
    len_list = 4 * len(operator_list)
    qmc_state.linked_list = np.zeros(len_list, dtype=int)
    qmc_state.leg_types = np.zeros(len_list, dtype=bool)
    qmc_state.associates = [(0, 0, 0) for _ in range(len_list)]

def diagonal_update_beta(qmc_state, H, beta: float, eq: bool = False):
    P_norm = beta * H.P_normalization

    num_ids = sum(1 for op in qmc_state.operator_list if isidentity(op))
    P_remove = (num_ids + 1) / P_norm
    P_accept = P_norm / num_ids

    spin_prop = qmc_state.propagated_config = qmc_state.left_config.copy()

    for n, op in enumerate(qmc_state.operator_list):
        if not isdiagonal(op):
            spin_prop[op[1]] ^= 1  # spinflip
        elif not isidentity(op):
            if np.random.random() < P_remove:
                qmc_state.operator_list[n] = (0, 0)
                num_ids += 1
                P_remove = (num_ids + 1) / P_norm
                P_accept = P_norm / num_ids
        else:
            if np.random.random() < P_accept:
                if insert_diagonal_operator(qmc_state, H, spin_prop, n):
                    P_remove = num_ids / P_norm
                    num_ids -= 1
                    P_accept = P_norm / num_ids

    total_list_size = len(qmc_state.operator_list)
    num_ops = total_list_size - num_ids

    if eq and 1.2 * num_ops > total_list_size:
        resize_op_list(qmc_state, int(1.5 * num_ops + 0.5))

    return num_ops

def linked_list_update_beta(qmc_state, H: TFIM):
    Ns = H.nspins()
    print(f"Number of spins (Ns): {Ns}")
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config

    # Calculate the length of the linked list
    len_list = sum(2 if issiteoperator(op) else 4 for op in qmc_state.operator_list if not isidentity(op))
    len_list += Ns  # Add extra space for safety
    print(f"Calculated len_list: {len_list}")

    # Find the maximum site index
    max_site = max(max(op[1] if issiteoperator(op) else max(op) for op in qmc_state.operator_list if not isidentity(op)), Ns - 1)
    print(f"Maximum site index: {max_site}")

    # Ensure the arrays are large enough
    if len(qmc_state.linked_list) < len_list:
        qmc_state.linked_list = np.zeros(len_list, dtype=int)
        qmc_state.leg_types = np.zeros(len_list, dtype=bool)
        qmc_state.associates = [(0, 0, 0) for _ in range(len_list)]

    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates

    # Increase the size of First and Last arrays
    First = qmc_state.first = np.zeros(max_site + 1, dtype=int)
    Last = qmc_state.last = np.zeros(max_site + 1, dtype=int)
    idx = 0

    spin_prop = qmc_state.propagated_config = spin_left.copy()

    print(f"Initial array sizes - LinkList: {len(LinkList)}, LegType: {len(LegType)}, Associates: {len(Associates)}, First: {len(First)}, Last: {len(Last)}")

    for op_idx, op in enumerate(qmc_state.operator_list):
        print(f"Processing operator {op_idx}: {op}")
        print(f"Current idx: {idx}")
        
        if idx >= len(LinkList) - 4:
            new_size = len(LinkList) * 2
            print(f"Resizing arrays to {new_size}")
            LinkList = np.resize(LinkList, new_size)
            LegType = np.resize(LegType, new_size)
            Associates.extend([(0, 0, 0)] * (new_size - len(Associates)))

        if issiteoperator(op):
            site = op[1]
            print(f"Site operator at site {site}")
            
            if site >= len(First):
                print(f"Warning: site {site} is out of bounds for First array with size {len(First)}. Skipping this operator.")
                continue

            LinkList[idx] = First[site]
            LegType[idx] = spin_prop[site] if site < len(spin_prop) else False
            
            if not isdiagonal(op):
                if site < len(spin_prop):
                    spin_prop[site] ^= 1

            if First[site] != 0:
                LinkList[First[site] - 1] = idx + 1
            else:
                Last[site] = idx + 1
            
            First[site] = idx + 1
            Associates[idx] = (0, 0, 0)
            idx += 1

            LegType[idx] = spin_prop[site] if site < len(spin_prop) else False
            Associates[idx] = (0, 0, 0)
            idx += 1

        elif isbondoperator(op):
            site1, site2 = op
            print(f"Bond operator at sites {site1} and {site2}")
            
            if site1 >= len(First) or site2 >= len(First):
                print(f"Warning: sites {site1} or {site2} are out of bounds for First array with size {len(First)}. Skipping this operator.")
                continue

            # lower left
            LinkList[idx] = First[site1]
            LegType[idx] = spin_prop[site1] if site1 < len(spin_prop) else False
            
            if First[site1] != 0:
                LinkList[First[site1] - 1] = idx + 1
            else:
                Last[site1] = idx + 1

            First[site1] = idx + 2
            vertex1 = idx
            Associates[idx] = (vertex1 + 1, vertex1 + 2, vertex1 + 3)
            idx += 1

            # lower right
            LinkList[idx] = First[site2]
            LegType[idx] = spin_prop[site2] if site2 < len(spin_prop) else False
            
            if First[site2] != 0:
                LinkList[First[site2] - 1] = idx + 1
            else:
                Last[site2] = idx + 1

            First[site2] = idx + 2
            Associates[idx] = (vertex1, vertex1 + 2, vertex1 + 3)
            idx += 1

            # upper left
            LegType[idx] = spin_prop[site1] if site1 < len(spin_prop) else False
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 3)
            idx += 1

            # upper right
            LegType[idx] = spin_prop[site2] if site2 < len(spin_prop) else False
            Associates[idx] = (vertex1, vertex1 + 1, vertex1 + 2)
            idx += 1

        print(f"After processing, idx: {idx}")
        print(f"Current LinkList size: {len(LinkList)}")

    # Periodic boundary conditions for finite-beta
    for i in range(min(Ns, len(First))):
        if First[i] != 0:  # This might be encountered at high temperatures
            LinkList[First[i] - 1] = Last[i]
            LinkList[Last[i] - 1] = First[i]

    print(f"Final idx: {idx}")
    print(f"Final LinkList size: {len(LinkList)}")

    qmc_state.linked_list = LinkList[:idx]
    qmc_state.leg_types = LegType[:idx]
    qmc_state.associates = Associates[:idx]

    return idx

def cluster_update_beta(lsize: int, qmc_state, H):
    Ns = H.nspins()
    spin_left, spin_right = qmc_state.left_config, qmc_state.right_config
    operator_list = qmc_state.operator_list
    LinkList = qmc_state.linked_list
    LegType = qmc_state.leg_types
    Associates = qmc_state.associates

    # Ensure lsize does not exceed the actual length of LinkList
    lsize = min(lsize, len(LinkList))
    in_cluster = np.zeros(lsize, dtype=int)
    cstack = deque()
    ccount = 0  # cluster number counter

    for i in range(lsize):
        # Add a new leg onto the cluster
        if in_cluster[i] == 0 and Associates[i] == (0, 0, 0):
            ccount += 1
            cstack.append(i)
            in_cluster[i] = ccount
            flip = np.random.random() < 0.5  # flip a coin for the SW cluster flip

            if flip:
                LegType[i] ^= 1  # spinflip

            while cstack:
                leg = LinkList[cstack.pop()]
                
                # Add safety check
                if leg >= lsize:
                    print(f"Warning: leg {leg} is out of bounds. Skipping.")
                    continue

                if in_cluster[leg] == 0:
                    in_cluster[leg] = ccount  # add the new leg and flip it
                    if flip:
                        LegType[leg] ^= 1
                    
                    # now check all associates and add to cluster
                    assoc = Associates[leg]
                    if assoc != (0, 0, 0):
                        for a in assoc:
                            # Add safety check
                            if a < lsize:
                                cstack.append(a)
                                in_cluster[a] = ccount
                                if flip:
                                    LegType[a] ^= 1
                            else:
                                print(f"Warning: associate {a} is out of bounds. Skipping.")

    # map back basis states and operator list
    First = qmc_state.first
    Last = qmc_state.last
    for i in range(Ns):
        if First[i] != 0:
            # Add safety checks
            last_index = min(Last[i] - 1, lsize - 1)
            first_index = min(First[i] - 1, lsize - 1)
            spin_left[i] = LegType[last_index]
            spin_right[i] = LegType[first_index]
        else:
            # randomly flip spins not connected to operators
            spin_left[i] = spin_right[i] = np.random.random() < 0.5

    ocount = 0  # first leg (Python uses 0-based indexing)
    for n, op in enumerate(operator_list):
        if isbondoperator(op):
            ocount += 4
        elif not isidentity(op):
            # Add safety check
            if ocount + 1 < lsize:
                if LegType[ocount] == LegType[ocount + 1]:  # diagonal
                    operator_list[n] = (-1, op[1])
                else:  # off-diagonal
                    operator_list[n] = (-2, op[1])
                ocount += 2
            else:
                print(f"Warning: ocount {ocount} is out of bounds. Skipping operator update.")

    # Update relevant arrays in qmc_state
    qmc_state.linked_list = LinkList[:lsize]
    qmc_state.leg_types = LegType[:lsize]
    qmc_state.associates = Associates[:lsize]
