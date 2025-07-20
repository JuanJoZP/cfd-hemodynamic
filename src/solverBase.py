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
    FacetNormal,
    inner,
    TestFunction,
    ds,
    FacetArea,
)
from dolfinx.mesh import Mesh
from dolfinx.fem import Constant, functionspace, Function, FunctionSpace, form
from dolfinx.fem.petsc import assemble_vector
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

    def initStressForm(self):
        scalar = functionspace(
            self.mesh, element("CG", self.mesh.topology.cell_name(), 1)
        )
        vector = functionspace(
            self.mesh,
            element(
                "CG", self.mesh.topology.cell_name(), 1, shape=(self.mesh.geometry.dim,)
            ),
        )

        self.normal_stress = Function(scalar)
        self.normal_stress.name = "normal_stress"
        self.shear_stress = Function(vector)
        self.shear_stress.name = "shear_stress"

        # stress forms
        n = FacetNormal(self.mesh)
        T = -self.sigma(self.u_sol, self.p_sol, self.mu) * n

        Tn = inner(T, n)
        Tt = T - Tn * n

        v = TestFunction(scalar)
        w = TestFunction(vector)

        self.Ln = (1 / FacetArea(self.mesh)) * v * Tn * ds
        self.Lt = (1 / FacetArea(self.mesh)) * inner(w, Tt) * ds

    @staticmethod
    def epsilon(u):
        return sym(nabla_grad(u))

    @staticmethod
    def sigma(u, p, mu):
        return 2 * mu * sym(nabla_grad(u)) - p * Identity(len(u))

    def assemble_wss(self):
        # self.normal_stress.x.petsc_vec.zeroEntries()
        # assemble_vector(self.normal_stress.x.petsc_vec, form(self.Ln))
        # self.normal_stress.x.petsc_vec.ghostUpdate(
        #     addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        # )
        self.shear_stress.x.petsc_vec.zeroEntries()
        assemble_vector(self.shear_stress.x.petsc_vec, form(self.Lt))
        self.shear_stress.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )
