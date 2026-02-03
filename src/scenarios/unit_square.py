from src.scenario import Scenario
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.mesh import create_unit_square, locate_entities_boundary, Mesh
from dolfinx.fem import Function

from src.boundaryCondition import BoundaryCondition

class UnitSquareSimulation(Scenario):
    def __init__(
        self, solver_name, dt, T, f: tuple[float, float] = (0, 0), *, rho=1, mu=1
    ):
        self._mesh: Mesh = None
        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None
        super().__init__(solver_name, "unit_square", rho, mu, dt, T, f)

        self.setup()

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh = create_unit_square(MPI.COMM_WORLD, 32, 32)

        return self._mesh

    @property
    def bcu(self):
        if not self._bcu:
            uD = Function(self.solver.V)
            uD.x.array[:] = 0
            fdim = self.mesh.topology.dim - 1
            walls_facets = locate_entities_boundary(self.mesh, fdim, self.walls)
            bc_noslip = BoundaryCondition(uD)
            bc_noslip.initTopological(fdim, walls_facets)

            ul = Function(self.solver.V)
            ul.interpolate(self.exact_velocity(0))
            inflow_facets = locate_entities_boundary(self.mesh, fdim, self.inflow)
            bc_inflow = BoundaryCondition(ul)
            bc_inflow.initTopological(fdim, inflow_facets)

            self._bcu = [bc_inflow, bc_noslip]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            fdim = self.mesh.topology.dim - 1
            pr = Function(self.solver.Q)
            pr.x.array[:] = 0
            outflow_facets = locate_entities_boundary(self.mesh, fdim, self.outflow)
            bc_outflow = BoundaryCondition(pr)
            bc_outflow.initTopological(fdim, outflow_facets)

            self._bcp = [bc_outflow]

        return self._bcp

    def initial_velocity(self, x):
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 1
        return values

    def exact_velocity(self, t):
        def velocity(x):
            return np.vstack((4.0 * x[1] * (1.0 - x[1]), 0.0 * x[0]))

        return velocity

    @staticmethod
    def inflow(x):
        return np.isclose(x[0], 0)

    @staticmethod
    def outflow(x):
        return np.isclose(x[0], 1)

    @staticmethod
    def walls(x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
