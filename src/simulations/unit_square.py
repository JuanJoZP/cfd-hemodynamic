from src.simulation import Simulation
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from dolfinx.fem import (
    dirichletbc,
    locate_dofs_topological,
    Constant,
)


n_cells = 32
solver_name = "solver1"
simulation_name = "unit_square"
rho = 1
mu = 1
dt = 1 / 200
T = 5

class UnitSquareSimulation(Simulation):
    def __init__(
        self, solver_name, rho=1, mu=1, dt=1 / 100, T=5, f: tuple[float, float] = (0, 0)
    ):
        self._mesh = None
        self._bcu = None
        self._bcp = None
        super().__init__(solver_name, simulation_name, rho, mu, dt, T, f)

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh = create_unit_square(MPI.COMM_WORLD, n_cells, n_cells)

        return self._mesh

    @property
    def bcu(self):
        if not self._bcu:
            fdim = self.mesh.topology.dim - 1
            walls_facets = locate_entities_boundary(self.mesh, fdim, self.walls)
            dofs_walls = locate_dofs_topological(self.solver.V, fdim, walls_facets)
            bc_noslip = dirichletbc(
                Constant(self.mesh, PETSc.ScalarType((0, 0))), dofs_walls, self.solver.V
            )
            self._bcu = [bc_noslip]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            fdim = self.mesh.topology.dim - 1
            inflow_facets = locate_entities_boundary(self.mesh, fdim, self.inflow)
            dofs_inflow = locate_dofs_topological(self.solver.Q, fdim, inflow_facets)
            bc_inflow = dirichletbc(
                Constant(self.mesh, PETSc.ScalarType(8)),
                dofs_inflow,
                self.solver.Q,
            )

            outflow_facets = locate_entities_boundary(self.mesh, fdim, self.outflow)
            dofs_outflow = locate_dofs_topological(self.solver.Q, fdim, outflow_facets)
            bc_outflow = dirichletbc(
                Constant(self.mesh, PETSc.ScalarType(0)),
                dofs_outflow,
                self.solver.Q,
            )

            self._bcp = [bc_inflow, bc_outflow]

        return self._bcp

    def initial_velocity(self, x):
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        values[1] = 10
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


simulation = UnitSquareSimulation(solver_name, rho, mu, dt, T)
simulation.solve()
