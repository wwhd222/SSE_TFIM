# probability_vector.py

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union

class AbstractProbabilityVector(ABC):
    @abstractmethod
    def __getitem__(self, i):
        pass

    @abstractmethod
    def __setitem__(self, i, w):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def normalization(self):
        pass

    def set_probability(self, p: float, i: int):
        self[i] = p * self.normalization()

    def get_probability(self, i: int) -> float:
        return self[i] / self.normalization()

class ProbabilityVector(AbstractProbabilityVector):
    def __init__(self, p: List[float]):
        if len(p) == 0:
            raise ValueError("probability vector must have non-zero length!")
        if any(x < 0 for x in p):
            raise ValueError("weights must be non-negative!")
        self.p = np.array(p, dtype=float)
        self.cdf = np.cumsum(self.p)

    def __len__(self):
        return len(self.p)

    def normalization(self):
        return self.cdf[-1]

    def __getitem__(self, i):
        return self.p[i]

    def __setitem__(self, i, w):
        diff = w - self.p[i]
        self.p[i] = w
        self.cdf[i:] += diff

    def __repr__(self):
        return f"ProbabilityVector({self.p.tolist()})"

    def rand(self) -> int:
        r = np.random.random() * self.cdf[-1]
        return np.searchsorted(self.cdf, r, side='right')

class ProbabilityHeap(AbstractProbabilityVector):
    def __init__(self, p: List[float]):
        if len(p) == 0:
            raise ValueError("probability vector must have non-zero length!")
        if any(x < 0 for x in p):
            raise ValueError("weights must be non-negative!")

        self.length = len(p)
        d = 2 ** int(np.ceil(np.log2(self.length)))

        self.prob_heap = np.zeros(2*d, dtype=float)
        self.prob_heap[d:d+self.length] = p

        for i in range(d-1, 0, -1):
            self.prob_heap[i] = self.prob_heap[2*i] + self.prob_heap[2*i + 1]

    def __len__(self):
        return self.length

    def normalization(self):
        return self.prob_heap[1]

    def __getitem__(self, i):
        d = len(self.prob_heap) // 2 - 1
        return self.prob_heap[d + i]

    def __setitem__(self, i, w):
        heap = self.prob_heap
        j = (len(heap) // 2) - 1 + i
        heap[j] = w
        while j > 1:
            j //= 2
            heap[j] = heap[2*j] + heap[2*j + 1]

    def __repr__(self):
        d = len(self.prob_heap) // 2
        pvec = self.prob_heap[d:d+self.length].tolist()
        return f"ProbabilityHeap({pvec})"

    def rand(self) -> int:
        heap = self.prob_heap
        l = len(heap) // 2
        r = np.random.random() * heap[1]

        i = 1
        while i < l:
            i *= 2
            left = heap[i]
            if r > left:
                r -= left
                i += 1
        return i - l + 1

CUTOFF = 50

def probability_vector(p: List[float]) -> AbstractProbabilityVector:
    if len(p) < CUTOFF:
        return ProbabilityVector(p)
    else:
        return ProbabilityHeap(p)
