import numpy as np
import pandas as pd
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
from src.simulation import Simulation
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.mesh import create_unit_square, locate_entities_boundary, Mesh
from dolfinx.fem import Function

from src.boundaryCondition import BoundaryCondition

solver_name = "stabilized_schur"
simulation_name = "lid_driven2D"
n_cells = 50
rho = 1
T = 10
dt = 1 / 200


class LidDriven2DSimulation(Simulation):
    def __init__(
        self, solver_name, rho=1, mu=1, dt=1 / 100, T=5, f: tuple[float, float] = (0, 0)
    ):
        self._mesh: Mesh = None
        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None
        self.Re = str(int(1 / mu))
        super().__init__(solver_name, simulation_name, rho, mu, dt, T, f)

        self.setup()

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh = create_unit_square(MPI.COMM_WORLD, n_cells, n_cells)

        return self._mesh

    @property
    def bcu(self):
        if not self._bcu:
            u_noslip = Function(self.solver.V)
            u_noslip.x.array[:] = 0
            fdim = self.mesh.topology.dim - 1
            walls_facets = locate_entities_boundary(self.mesh, fdim, self.walls)
            bc_noslip = BoundaryCondition(u_noslip)
            bc_noslip.initTopological(fdim, walls_facets)

            u_lid = Function(self.solver.V)
            u_lid.interpolate(
                lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1])))
            )
            lid_facets = locate_entities_boundary(self.mesh, fdim, self.lid)
            bc_lid = BoundaryCondition(u_lid)
            bc_lid.initTopological(fdim, lid_facets)

            self._bcu = [bc_noslip, bc_lid]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            self._bcp = []

        return self._bcp

    def initial_velocity(self, x):
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        return values

    @staticmethod
    def lid(x):
        return np.isclose(x[1], 1.0) & (x[0] > 1e-10) & (x[0] < 1.0 - 1e-10)

    @staticmethod
    def walls(x):
        return np.logical_or.reduce(
            (np.isclose(x[0], 0), np.isclose(x[0], 1), np.isclose(x[1], 0))
        )

    def save_benchmark_plot(self, results_path):
        data = pd.read_csv(f"src/benchmark_data/lid_driven2D/plot_u_y_Ghia{self.Re}.csv")
        points = np.column_stack(
            (
                np.array((0.5,) * data["y"].size, dtype=np.float64),
                data["y"].to_numpy(),
                np.array((0,) * data["y"].size, dtype=np.float64),
            )
        )

        tree = bb_tree(self.mesh, self.mesh.geometry.dim)
        cell_candidates = compute_collisions_points(tree, points)
        colliding_cells = compute_colliding_cells(self.mesh, cell_candidates, points)

        val_sol = np.array([])
        val_bench = np.array([])

        for i, p in enumerate(points):
            val_sol = np.append(
                val_sol, self.solver.u_sol.eval(p, colliding_cells.links(i)[:1])[0]
            )
            val_bench = np.append(val_bench, data["u"][i])

        fig, ax = plt.subplots()
        ax.plot(val_sol, label="Del solver")
        ax.plot(val_bench, label="Del benchmark")
        ax.legend()
        ax.set_ylabel("componente x de la velocidad")
        ax.set_xticks(
            np.arange(data["y"].size),
            map(lambda x: round(x, 2), data["y"].to_list()),
            rotation=30,
        )
        ax.set_title("Componente x de la velocidad en x=0.5")
        fig.savefig(f"{results_path}/benchmark_{self.Re}.png")


simulation = LidDriven2DSimulation(solver_name, rho, 1 / 100, dt, T)
results_path = simulation.solve()
simulation.save_benchmark_plot(results_path)
print(f"Resultados guardados en: {results_path}")

simulation = LidDriven2DSimulation(solver_name, rho, 1 / 400, dt, T)
results_path = simulation.solve()
simulation.save_benchmark_plot(results_path)
print(f"Resultados guardados en: {results_path}")

simulation = LidDriven2DSimulation(solver_name, rho, 1 / 1000, dt, T)
results_path = simulation.solve()
simulation.save_benchmark_plot(results_path)
print(f"Resultados guardados en: {results_path}")
