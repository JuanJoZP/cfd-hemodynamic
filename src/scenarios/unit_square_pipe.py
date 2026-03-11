"""
2-D pressure-driven channel flow on a quadrilateral rectangle mesh.

Geometry (all dimensions in mm):
  x ∈ [0, L=80]   – axial (flow) direction
  y ∈ [0, H=1.5]  – channel height

Mesh: 587×11 quadrilateral elements  (square elements, h ≈ 0.136 mm)
  → (588×12) = 7 056 nodes
  → ~21 k DOFs  (2 velocity + 1 pressure, P1-P1)

Boundary conditions:
  Inlet  (x = 0)    : natural pressure = p_inlet  [Pa]
  Outlet (x = L)    : natural pressure = p_outlet [Pa]
  Walls  (y = 0, H) : Dirichlet velocity = 0  (no-slip)

Blood parameters (mm-g-s system):
  rho  = 1.06e-3  g/mm³   ≡ 1060   kg/m³
  mu   = 3.5e-3   g/(mm·s) ≡ 3.5e-3 Pa·s

Target p_inlet:
  Plane Poiseuille (H=1.5 mm, L=80 mm):
    ΔP = 12 · μ · U_mean · L / H²
       = 12 × 3.5e-3 × 5 × 80 / 2.25  ≈ 7.47 Pa
"""

import numpy as np
from dolfinx.fem import Function
from dolfinx.mesh import (
    CellType,
    Mesh,
    create_rectangle,
    locate_entities_boundary,
    meshtags,
)
from mpi4py import MPI
from petsc4py import PETSc

from src.boundaryCondition import BoundaryCondition
from src.scenario import Scenario

# Geometry constants (mm)
_L = 80.0  # channel length (axial, x)
_H = 1.5   # channel height (y)

# Mesh resolution → ~21 k DOFs  (588×12 = 7 056 nodes × 3 = 21 168 DOFs)
# Square elements: h = H/NY = 1.5/11 ≈ 0.136 mm; NX = round(L/h) = 587
_NX = 587
_NY = 11


class UnitSquarePipeSimulation(Scenario):
    """Pressure-driven 2-D channel flow on a structured quadrilateral mesh."""

    inlet_marker = 1
    outlet_marker = 2
    wall_marker = 3

    def __init__(
        self,
        solver_name: str,
        dt: float,
        T: float,
        f: tuple = (0.0, 0.0),
        *,
        rho: float = 1.06e-3,
        mu: float = 3.5e-3,
        p_inlet: float,
        p_outlet: float,
        early_stop_tolerance: float = 1e-5,
    ):
        self.p_inlet = float(p_inlet)
        self.p_outlet = float(p_outlet)

        self._mesh: Mesh | None = None
        self._ft = None
        self._bcu = None
        self._bcp = None

        super().__init__(
            solver_name,
            "unit_square_pipe",
            rho,
            mu,
            dt,
            T,
            list(f),
            early_stop_tolerance=early_stop_tolerance,
            p_inlet=self.p_inlet,
            p_outlet=self.p_outlet,
        )

        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )
        self.setup()

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self._mesh = create_rectangle(
                MPI.COMM_WORLD,
                [[0.0, 0.0], [_L, _H]],
                [_NX, _NY],
                cell_type=CellType.quadrilateral,
            )

            fdim = self._mesh.topology.dim - 1

            inlet_facets = locate_entities_boundary(
                self._mesh, fdim, lambda x: np.isclose(x[0], 0.0)
            )
            outlet_facets = locate_entities_boundary(
                self._mesh, fdim, lambda x: np.isclose(x[0], _L)
            )
            wall_facets = locate_entities_boundary(
                self._mesh,
                fdim,
                lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], _H),
            )

            indices = np.concatenate([inlet_facets, outlet_facets, wall_facets])
            values = np.concatenate([
                np.full(len(inlet_facets),  self.inlet_marker,  dtype=np.int32),
                np.full(len(outlet_facets), self.outlet_marker, dtype=np.int32),
                np.full(len(wall_facets),   self.wall_marker,   dtype=np.int32),
            ])
            sorted_idx = np.argsort(indices)
            self._ft = meshtags(
                self._mesh, fdim, indices[sorted_idx], values[sorted_idx]
            )

        return self._mesh

    @property
    def bcu(self) -> list[BoundaryCondition]:
        """No-slip at top and bottom walls; inlet and outlet are free (pressure-driven)."""
        if self._bcu is None:
            fdim = self.mesh.topology.dim - 1
            u_noslip = Function(self.solver.V)
            u_noslip.x.array[:] = 0.0
            wall_facets = self._ft.find(self.wall_marker)
            bc_walls = BoundaryCondition(u_noslip)
            bc_walls.initTopological(fdim, wall_facets)
            self._bcu = [bc_walls]
        return self._bcu

    @property
    def bcp(self) -> list[BoundaryCondition]:
        """Dirichlet pressure at inlet (p_inlet) and outlet (p_outlet)."""
        if self._bcp is None:
            fdim = self.mesh.topology.dim - 1

            p_in = Function(self.solver.Q)
            p_in.x.array[:] = self.p_inlet
            inlet_facets = self._ft.find(self.inlet_marker)
            bc_in = BoundaryCondition(p_in)
            bc_in.initTopological(fdim, inlet_facets)

            p_out = Function(self.solver.Q)
            p_out.x.array[:] = self.p_outlet
            outlet_facets = self._ft.find(self.outlet_marker)
            bc_out = BoundaryCondition(p_out)
            bc_out.initTopological(fdim, outlet_facets)

            self._bcp = [bc_in, bc_out]
        return self._bcp

    def initial_velocity(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
