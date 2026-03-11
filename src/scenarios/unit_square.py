import numpy as np
from dolfinx.fem import Function
from dolfinx.mesh import (
    CellType,
    Mesh,
    create_unit_square,
    locate_entities_boundary,
    meshtags,
)
from mpi4py import MPI
from petsc4py import PETSc

from src.boundaryCondition import BoundaryCondition
from src.scenario import Scenario


class UnitSquareSimulation(Scenario):
    inlet_marker = 1
    outlet_marker = 2
    wall_marker = 3

    def __init__(
        self, solver_name, dt, T, f: tuple[float, float] = (0, 0), *, rho=1, mu=1
    ):
        self._mesh: Mesh = None
        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None
        self._ft = None
        super().__init__(solver_name, "unit_square", rho, mu, dt, T, f)

        self.setup()

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh = create_unit_square(
                MPI.COMM_WORLD, 32, 32, cell_type=CellType.quadrilateral
            )

            # Create facet tags
            fdim = self._mesh.topology.dim - 1
            inflow_facets = locate_entities_boundary(self._mesh, fdim, self.inflow)
            outflow_facets = locate_entities_boundary(self._mesh, fdim, self.outflow)
            wall_facets = locate_entities_boundary(self._mesh, fdim, self.walls)

            indices = np.concatenate([inflow_facets, outflow_facets, wall_facets])
            values = np.concatenate(
                [
                    np.full(len(inflow_facets), self.inlet_marker, dtype=np.int32),
                    np.full(len(outflow_facets), self.outlet_marker, dtype=np.int32),
                    np.full(len(wall_facets), self.wall_marker, dtype=np.int32),
                ]
            )
            sorted_indices = np.argsort(indices)
            self._ft = meshtags(
                self._mesh, fdim, indices[sorted_indices], values[sorted_indices]
            )

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
