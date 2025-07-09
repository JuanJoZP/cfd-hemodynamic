from importlib import import_module
from abc import ABC, abstractmethod
from typing import Callable

import os
import numpy as np
from src.boundaryCondition import BoundaryCondition
from src.solverBase import SolverBase
from datetime import datetime, timezone, timedelta
from mpi4py import MPI
from ufl import inner, dx
from dolfinx.io import VTXWriter
from dolfinx.mesh import Mesh
from dolfinx.fem import DirichletBC, form, assemble_scalar, Function, Expression


class SimulationBase(ABC):
    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        pass

    @property
    @abstractmethod
    def bcu(self) -> list[BoundaryCondition]:
        pass

    @property
    @abstractmethod
    def bcp(self) -> list[BoundaryCondition]:
        pass

    @abstractmethod
    def initial_velocity(self, x: np.ndarray) -> np.ndarray:
        pass

    def exact_velocity(self, t) -> Function | Expression | Callable:
        pass

    def __init__(
        self,
        solver_name: str,
        simulation_name: str,
        rho: float,
        mu: float,
        dt: float,
        T: float,
        f: list,
    ):
        self.solver_name = solver_name
        self.solverClass: type[SolverBase] = getattr(
            import_module(f"src.solvers.{solver_name}"), "Solver"
        )
        self.solver = self.solverClass(
            self.mesh, dt, rho, mu, f, initial_velocity=self.initial_velocity
        )

        self.num_steps = int(T / dt)
        self.has_exact_solution = (
            self.__class__.exact_velocity is not SimulationBase.exact_velocity
        )

        self.dt = dt
        self.simulation_name = simulation_name

    def setup(self):
        self.solver.setup(self.bcu, self.bcp)

    def solve(self, afterStepCallback: Callable[[float], None] = None) -> str:
        """
        Runs the time-stepping simulation, returns the path to directory with results in VTX
        format and an error log if the method `exact_velocity` is implemented.

        Args:
            afterStepCallback (Callable[[float], None], optional): A function to be called
                after each time step, receiving the current simulation time as an argument.

        Returns:
            str: The absolute path to the directory where simulation results are stored.
        """

        tqdm = self.get_tqdm()
        mesh = self.mesh
        num_steps = self.num_steps
        solver = self.solver

        progress = tqdm(desc="Solving", total=num_steps) if mesh.comm.rank == 0 else None

        date = (
            datetime.now(tz=timezone(-timedelta(hours=5))).isoformat(timespec="seconds")
            if mesh.comm.rank == 0
            else None
        )
        date = mesh.comm.bcast(date, root=0)

        safe_date = date.replace(":", ".")
        parent_route = f"results/{self.simulation_name}/{self.solver_name}/{safe_date}"
        u_file = VTXWriter(mesh.comm, f"{parent_route}/v.bp", self.solver.u_sol)
        p_file = VTXWriter(mesh.comm, f"{parent_route}/p.bp", self.solver.p_sol)

        t = 0.0
        u_file.write(t)
        p_file.write(t)

        error_log = None
        if self.has_exact_solution:
            error_log = (
                open(f"{parent_route}/err.txt", "w") if mesh.comm.rank == 0 else None
            )
            u_e = Function(solver.V)
            u_e.interpolate(lambda x: self.exact_velocity(t)(x))
            error = self.compute_error(solver.u_sol, u_e, mesh)

        if error_log:
            error_log.write("t = %.3f: error = %.3g" % (t, error) + "\n")

        for _ in range(num_steps):
            solver.solveStep()

            if progress:
                progress.update()

            t += self.dt

            if error_log:
                u_e.interpolate(self.exact_velocity(t))
                error = self.compute_error(u_e, solver.u_sol, mesh)
                error_log.write("t = %.3f: error = %.3g" % (t, error) + "\n")

            u_file.write(t)
            p_file.write(t)

            if afterStepCallback:
                afterStepCallback(t)

        u_file.close()
        p_file.close()
        if error_log:
            error_log.close()
        if progress:
            progress.close()

        return os.path.abspath(parent_route)

    @staticmethod
    def get_tqdm():
        try:
            # Check if inside Jupyter notebook
            from IPython import get_ipython

            shell = get_ipython().__class__.__name__
            if shell in ["ZMQInteractiveShell"]:
                from tqdm.notebook import tqdm as notebook_tqdm

                return notebook_tqdm
        except:
            pass
        from tqdm import tqdm

        return tqdm

    @staticmethod
    def compute_error(u: Function, u_aprox: Function, mesh: Mesh) -> float:
        """Compute the L2 relative error between u and u_aprox."""

        error_form = form(inner(u_aprox - u, u_aprox - u) * dx)
        error_abs = np.sqrt(mesh.comm.allreduce(assemble_scalar(error_form), op=MPI.SUM))
        norm_form = form(inner(u, u) * dx)
        norm = np.sqrt(mesh.comm.allreduce(assemble_scalar(norm_form), op=MPI.SUM))
        return error_abs / norm
