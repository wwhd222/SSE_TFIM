# hamiltonian.py

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Type, TypeVar

from .op_sampler import HierarchicalOperatorSampler, make_prob_vector

T = TypeVar('T', bound='Hamiltonian')

class Hamiltonian(ABC):
    def __init__(self, D: int, N: int):
        self.D = D
        self.N = N

    @classmethod
    def localdim(cls: Type[T]) -> int:
        return cls.D

    @classmethod
    def dim(cls: Type[T]) -> int:
        return cls.N

    @abstractmethod
    def nspins(self) -> int:
        pass

    @abstractmethod
    def nbonds(self) -> int:
        pass

    def zero(self):
        if self.D == 2:
            return np.zeros(self.nspins(), dtype=bool)
        return np.zeros(self.nspins())

    def one(self):
        if self.D == 2:
            return np.ones(self.nspins(), dtype=bool)
        return np.ones(self.nspins())

class TFIM(Hamiltonian):
    def __init__(self, bond_spin: List[Tuple[int, int]], Dim: int, Ns: int, Nb: int, h: float, J: float):
        super().__init__(2, Dim)
        ops, p = make_prob_vector(bond_spin, Ns, J, h)
        self.op_sampler = HierarchicalOperatorSampler(ops, p)
        self.h = h
        self.J = J
        self.P_normalization = sum(p)
        self.Ns = Ns
        self.Nb = Nb

    def nspins(self) -> int:
        return self.Ns

    def nbonds(self) -> int:
        return self.Nb

def make_tfim(bond_spin: List[Tuple[int, int]], Dim: int, Ns: int, Nb: int, h: float, J: float) -> TFIM:
    return TFIM(bond_spin, Dim, Ns, Nb, h, J)
