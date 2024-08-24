# lattice.py
#
# Defines the spatial lattice; supports 1D and 2D square, open and periodic boundaries
# The main data structure is the bond-spin index array, bond_spin[nBond,2]

from typing import List, Tuple

def lattice_bond_spins_1d(nX: int, pbc: bool = True) -> Tuple[List[Tuple[int, int]], int, int]:
    Nb = nX if pbc else nX - 1
    Ns = nX

    bond_spin = [(0, 0) for _ in range(Nb)]
    for i in range(Nb):
        bond_spin[i] = (i, (i + 1) % nX if pbc else i + 1)

    return bond_spin, Ns, Nb

def lattice_bond_spins_2d(nX: Tuple[int, int], pbc: bool = True) -> Tuple[List[Tuple[int, int]], int, int]:
    if nX[0] != nX[1]:
        raise ValueError("Currently only supports square lattices!")
    L = nX[0]

    Ns = L * L
    Nb = 2 * Ns if pbc else 2 * (Ns - L)
    bond_spin = [(0, 0) for _ in range(Nb)]

    if pbc:
        for i in range(Ns):
            # horizontal
            bond1 = 2 * i

            if (i + 1) % L != 0:
                bond_spin[bond1] = (i, i + 1)
            else:
                bond_spin[bond1] = (i, i + 1 - L)

            # vertical
            bond2 = 2 * i + 1

            if i < (Ns - L):
                bond_spin[bond2] = (i, i + L)
            else:
                bond_spin[bond2] = (i, i + L - Ns)
    else:
        # horizontal
        cnt = 0
        for i in range(Nb // 2):
            if (i + 1) % (L - 1) != 0:
                bond_spin[i] = (cnt, cnt + 1)
                cnt += 1
            else:
                bond_spin[i] = (cnt, cnt + 1)
                cnt += 2

        # vertical
        cnt = 0
        for i in range(Nb // 2, Nb):
            bond_spin[i] = (cnt, cnt + L)
            cnt += 1

    return bond_spin, Ns, Nb

def lattice_bond_spins(nX: Union[int, Tuple[int, int]], pbc: bool = True) -> Tuple[List[Tuple[int, int]], int, int]:
    if isinstance(nX, int):
        return lattice_bond_spins_1d(nX, pbc)
    elif isinstance(nX, tuple) and len(nX) == 2:
        return lattice_bond_spins_2d(nX, pbc)
    else:
        raise ValueError("nX must be an integer for 1D or a tuple of two integers for 2D")
