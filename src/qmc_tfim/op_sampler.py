# op_sampler.py

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union
from probability_vector import probability_vector, AbstractProbabilityVector

def make_prob_vector(bond_spins: List[Tuple[int, int]], Ns: int, J: float, h: float) -> Tuple[List[Tuple[int, int]], List[float]]:
    ops = []
    p = []

    if h != 0:
        for i in range(1, Ns + 1):
            ops.append((-1, i))
            p.append(h)

    if J != 0:
        for op in bond_spins:
            ops.append(op)
            p.append(2 * abs(J))

    return ops, p

class AbstractOperatorSampler(ABC):
    def __init__(self, N: int, T: type, P: type):
        self.N = N
        self.T = T
        self.P = P

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def rand(self):
        pass

class OperatorSampler(AbstractOperatorSampler):
    def __init__(self, operators: List[Tuple[int, ...]], p: List[float]):
        assert len(operators) == len(p), "Given lists must have the same length!"
        N = len(operators[0])
        T = type(p[0])
        pvec = probability_vector(p)
        super().__init__(N, T, type(pvec))
        self.operators = operators
        self.pvec = pvec

    def __len__(self):
        return len(self.operators)

    def __getitem__(self, index):
        return self.operators[index]

    def rand(self):
        return self.operators[self.pvec.rand()]

def cluster_probs_vec(operators: List[Tuple[int, ...]], p: List[float]) -> Tuple[List[List[Tuple[int, ...]]], List[float]]:
    sorted_indices = np.argsort(p)
    p = [p[i] for i in sorted_indices]
    operators = [operators[i] for i in sorted_indices]

    uniq_p = []
    uniq_ops = []

    for i in range(len(p)):
        if not uniq_p or not np.isclose(uniq_p[-1], p[i]):
            uniq_p.append(p[i])
            uniq_ops.append([operators[i]])
        else:
            uniq_ops[-1].append(operators[i])

    # rescale uniq_p
    for i in range(len(uniq_p)):
        uniq_p[i] *= len(uniq_ops[i])

    return uniq_ops, uniq_p

class HierarchicalOperatorSampler(AbstractOperatorSampler):
    def __init__(self, operators: List[Tuple[int, ...]], p: List[float]):
        assert len(operators) == len(p), "Given lists must have the same length!"
        N = len(operators[0])
        T = type(p[0])
        operator_bins, p = cluster_probs_vec(operators, p)
        pvec = probability_vector(p)
        super().__init__(N, T, type(pvec))
        self.operator_bins = operator_bins
        self.pvec = pvec

    def __len__(self):
        return sum(len(bin) for bin in self.operator_bins)

    def __getitem__(self, index):
        for bin in self.operator_bins:
            if index < len(bin):
                return bin[index]
            index -= len(bin)
        raise IndexError("Index out of range")

    def rand(self):
        ops_list = self.operator_bins[self.pvec.rand()]
        return ops_list[np.random.randint(len(ops_list))]
