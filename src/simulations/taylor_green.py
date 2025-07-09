from src.simulationBase import SimulationBase
from mpi4py import MPI
import numpy as np
from dolfinx.mesh import (
    create_unit_cube,
    exterior_facet_indices,
    Mesh,
)
from dolfinx.fem import Function

from src.boundaryCondition import BoundaryCondition

solver_name = "solver1"
simulation_name = "taylor_green"
n_cells = 32
rho = 1
mu = 1 / 50  # Re = 50
dt = 1 / 1000
T = 0.01


class TaylorGreenSimulation(SimulationBase):
    def __init__(
        self, solver_name, rho, mu, dt, T, f: tuple[float, float, float] = (0, 0, 0)
    ):
        self._mesh: Mesh = None
        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None
        self._boundary_facets = None

        super().__init__(solver_name, simulation_name, rho, mu, dt, T, f)

        self._u_bc = Function(self.solver.V)
        self._p_bc = Function(self.solver.Q)
        self._u_bc.interpolate(self.exact_velocity(0))
        self._p_bc.interpolate(self.exact_pressure(0))

        self.setup()

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh = create_unit_cube(MPI.COMM_WORLD, n_cells, n_cells, n_cells)
            self._mesh.topology.create_connectivity(
                self._mesh.topology.dim - 1, self._mesh.topology.dim
            )
            self._boundary_facets = exterior_facet_indices(self._mesh.topology)

        return self._mesh

    @property
    def bcu(self):
        if not self._bcu:
            bcu = BoundaryCondition(self._u_bc)
            bcu.initTopological(self.mesh.topology.dim - 1, self._boundary_facets)

            self._bcu = [bcu]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            bcp = BoundaryCondition(self._p_bc)
            bcp.initTopological(self.mesh.topology.dim - 1, self._boundary_facets)

            self._bcp = [bcp]

        return self._bcp

    def initial_velocity(self, x):
        return self.exact_velocity(0)(x)

    def solve(self):
        def update_boundary_conditions(t):
            self._u_bc.interpolate(self.exact_velocity(t))
            self._p_bc.interpolate(self.exact_pressure(t))

        return super().solve(update_boundary_conditions)

    def exact_velocity(self, t):
        def velocity(x):
            x, y, z = x[0], x[1], x[2]
            a = np.pi / 4
            d = np.pi / 2
            return np.vstack(
                (
                    -a
                    * (
                        np.exp(a * x) * np.sin(a * y + d * z)
                        + np.exp(a * z) * np.cos(a * x + d * y)
                    )
                    * np.exp(-1 * d * d * t),
                    -a
                    * (
                        np.exp(a * y) * np.sin(a * z + d * x)
                        + np.exp(a * x) * np.cos(a * y + d * z)
                    )
                    * np.exp(-1 * d * d * t),
                    -a
                    * (
                        np.exp(a * z) * np.sin(a * x + d * y)
                        + np.exp(a * y) * np.cos(a * z + d * x)
                    )
                    * np.exp(-1 * d * d * t),
                )
            )

        return velocity

    def exact_pressure(self, t):
        def pressure(x):
            x, y, z = x[0], x[1], x[2]
            a = np.pi / 4
            d = np.pi / 2
            return (
                -1
                * a
                * a
                * (1 / 2)
                * (
                    np.exp(2 * a * x)
                    + np.exp(2 * a * y)
                    + np.exp(2 * a * z)
                    + 2
                    * np.sin(a * x + d * y)
                    * np.cos(a * z + d * x)
                    * np.exp(a * y + a * z)
                    + 2
                    * np.sin(a * y + d * z)
                    * np.cos(a * x + d * y)
                    * np.exp(a * z + a * x)
                    + 2
                    * np.sin(a * z + d * x)
                    * np.cos(a * y + d * z)
                    * np.exp(a * x + a * y)
                )
                * np.exp(-2 * d * d * t)
            )

        return pressure


simulation = TaylorGreenSimulation(solver_name, rho, mu, dt, T)
results_path = simulation.solve()
print(f"Resultados guardados en: {results_path}")
