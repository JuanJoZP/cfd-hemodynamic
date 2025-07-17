from abc import ABC, abstractmethod
from typing import Callable
from basix import ElementFamily, CellType
from basix.ufl import element
import numpy as np
from petsc4py import PETSc
from ufl import (
    Identity,
    nabla_grad,
    sym,
)
from dolfinx.mesh import Mesh
from dolfinx.fem import Constant, functionspace, Function, FunctionSpace
from src.boundaryCondition import BoundaryCondition


class SolverBase(ABC):
    @abstractmethod
    def __init__(
        self,
        mesh: Mesh,
        dt: float,
        rho: float,
        mu: float,
        f: list,
        initial_velocity: Callable[[np.ndarray], np.ndarray] = None,
    ):
        self.mesh = mesh
        self.dt = Constant(mesh, PETSc.ScalarType(dt))
        self.rho = Constant(mesh, PETSc.ScalarType(rho))
        self.mu = Constant(mesh, PETSc.ScalarType(mu))
        self.f = Constant(mesh, PETSc.ScalarType(f))
        self._u_sol: Function | None = None
        self._p_sol: Function | None = None
        self._u_prev: Function | None = None
        self._p_prev: Function | None = None
        self._V: FunctionSpace | None = None
        self._Q: FunctionSpace | None = None

    @property
    def u_sol(self):
        assert (
            self._u_sol is not None
        ), "Velocity solution function is not initialized. call initVelocitySpace() first."

        return self._u_sol

    @property
    def p_sol(self):
        assert (
            self._p_sol is not None
        ), "Pressure solution function is not initialized. call initPressureSpace() first."

        return self._p_sol

    @property
    def u_prev(self):
        assert (
            self._u_prev is not None
        ), "Velocity solution function is not initialized. call initVelocitySpace() first."

        return self._u_prev

    @property
    def p_prev(self):
        assert (
            self._p_prev is not None
        ), "Pressure solution function is not initialized. call initPressureSpace() first."

        return self._p_sol

    @property
    def V(self):
        assert (
            self._V is not None
        ), "Velocity function space is not initialized. call initVelocitySpace() first."

        return self._V

    @property
    def Q(self):
        assert (
            self._Q is not None
        ), "Pressure function space is not initialized. call initPressureSpace() first."

        return self._Q

    @abstractmethod
    def setup(self, bcu: list[BoundaryCondition], bcp: list[BoundaryCondition]) -> None:
        pass

    @abstractmethod
    def solveStep(self) -> None:
        pass

    def initVelocitySpace(
        self,
        family: ElementFamily | str,
        cell: CellType | str,
        deegre: int,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        """
        Initialize the function space for velocity `self.V`.
        Initialize the solution function `self.u_sol` in the velocity space, this is used by simulation to save the solution to a file.
        """
        element_v = element(family, cell, deegre, shape=shape)
        self._V = functionspace(self.mesh, element_v)
        self._u_sol = Function(self.V)
        self._u_sol.name = "velocity"
        self._u_prev = Function(self.V)

    def initPressureSpace(
        self,
        family: ElementFamily | str,
        cell: CellType | str,
        deegre: int,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        """
        Initialize the function space for pressure `self.Q`.
        Initialize the solution function `self.p_sol` in the pressure space, this is used by simulation to save the solution to a file.
        """
        element_p = element(family, cell, deegre, shape=shape)
        self._Q = functionspace(self.mesh, element_p)
        self._p_sol = Function(self.Q)
        self._p_sol.name = "pressure"
        self._p_prev = Function(self.Q)

    @staticmethod
    def epsilon(u):
        return sym(nabla_grad(u))

    @staticmethod
    def sigma(u, p, mu):
        return 2 * mu * sym(nabla_grad(u)) - p * Identity(len(u))
