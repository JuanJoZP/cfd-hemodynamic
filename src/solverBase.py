from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from ufl import (
    Identity,
    nabla_grad,
    sym,
)
from dolfinx.mesh import Mesh


class SolverBase(ABC):
    @abstractmethod
    def __init__(
        self,
        domain: Mesh,
        dt: float,
        rho: float,
        mu: float,
        f: list,
        h: list = None,
        initial_velocity: Callable[[np.ndarray], np.ndarray] = None,
    ):
        pass

    @abstractmethod
    def assembleTimeIndependent(self, bcu: list, bcp: list) -> None:
        pass

    @abstractmethod
    def solveStep(self, bcu: list, bcp: list) -> None:
        pass

    @staticmethod
    def epsilon(u):
        return sym(nabla_grad(u))

    @staticmethod
    def sigma(u, p, mu):
        return 2 * mu * sym(nabla_grad(u)) - p * Identity(len(u))
