# qmc_state.py

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union

from .hamiltonian import Hamiltonian, TFIM

def init_op_list(length: int) -> List[Tuple[int, int]]:
    return [(0, 0) for _ in range(length)]

class AbstractQMCState(ABC):
    def __init__(self, D: int, N: int):
        self.D = D
        self.N = N

class AbstractGroundState(AbstractQMCState):
    pass

class AbstractThermalState(AbstractQMCState):
    pass

class BinaryGroundState(AbstractGroundState):
    def __init__(self, left_config: np.ndarray, right_config: np.ndarray, operator_list: List[Tuple[int, int]]):
        super().__init__(2, left_config.ndim)
        assert left_config is not right_config, "left_config and right_config can't be the same array!"

        self.left_config = left_config
        self.right_config = right_config
        self.propagated_config = left_config.copy()
        self.operator_list = operator_list

        len_config = len(left_config)
        len_op_list = len(operator_list)
        len_total = 2 * len_config + 4 * len_op_list

        self.linked_list = np.zeros(len_total, dtype=int)
        self.leg_types = np.zeros(len_total, dtype=bool)
        self.associates = [(0, 0, 0) for _ in range(len_total)]
        self.first = np.zeros(len_config, dtype=int)

    @classmethod
    def from_hamiltonian(cls, H: Hamiltonian, M: int):
        operator_list = init_op_list(2 * M)
        left_config = H.zero()
        right_config = H.zero()
        return cls(left_config, right_config, operator_list)

class BinaryThermalState(AbstractThermalState):
    def __init__(self, H: TFIM, M: int, operator_list: List[Tuple[int, int]]):
        super().__init__(2, H.N)
        self.left_config = np.random.choice([0, 1], size=H.Ns)
        self.right_config = self.left_config.copy()
        self.propagated_config = self.left_config.copy()
        self.operator_list = operator_list

        len_list = 4 * len(operator_list) + H.Ns  # Add extra space
        self.linked_list = np.zeros(len_list, dtype=int)
        self.leg_types = np.zeros(len_list, dtype=bool)
        self.associates = [(0, 0, 0) for _ in range(len_list)]

        self.first = np.zeros(H.Ns, dtype=int)
        self.last = np.zeros(H.Ns, dtype=int)

    @classmethod
    def from_hamiltonian(cls, H: Hamiltonian, cutoff: int):
        operator_list = init_op_list(cutoff)
        left_config = H.zero()
        right_config = H.zero()
        return cls(left_config, right_config, operator_list)

BinaryQMCState = Union[BinaryGroundState, BinaryThermalState]
