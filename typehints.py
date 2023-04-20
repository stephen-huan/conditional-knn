from typing import Callable, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

# general

Sparse: TypeAlias = csc_matrix
Matrix = NDArray[np.double]
Vector = NDArray[np.double]
Points = NDArray[np.double]


class Empty:

    """Empty "list" simply holding the length of the data."""

    def __init__(self, n):
        self.n = n

    def __len__(self) -> int:
        return self.n


class Kernel(Protocol):
    """Wrapper over kernel for type hinting."""

    def __call__(
        self,
        X,  # pyright: ignore
        Y=None,  # pyright: ignore
        eval_gradient: bool = False,  # pyright: ignore
    ) -> Matrix:
        ...

    def diag(self, X) -> Matrix:  # pyright: ignore
        ...


# ordering

Ordering = NDArray[np.int_]
LengthScales = NDArray[np.double]
Sparsity = dict[int, list[int] | Empty]
Grouping = list[list[int]]

# selection

Indices = NDArray[np.int_]
PointSelect = Callable[[Points, Points, Kernel, int], Indices]
IndexSelect = Callable[[Points, Indices, Indices, Kernel, int], Indices]
GlobalSelect = Callable[
    [Points, Kernel, Sparsity, Sparsity, Grouping], Sparsity
]
Select = PointSelect | IndexSelect

# cholesky

CholeskyFactor = tuple[Sparse, Ordering]
CholeskySelect = Callable[[Points, Kernel, int, Grouping | None], Sparse]

# gp regression

Sample = Callable[[int], Matrix]
InvChol = Callable[[Points, Kernel], CholeskyFactor]
JointInvChol = Callable[[Points, Points, Kernel], CholeskyFactor]
