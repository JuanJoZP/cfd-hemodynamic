import os
import sys
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Callable

import numpy as np
from dolfinx.fem import Expression, Function, assemble_scalar, form
from dolfinx.io import VTXWriter
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx, inner

from src.boundaryCondition import BoundaryCondition
from src.solverBase import SolverBase


class Scenario(ABC):
    EARLY_STOP_TOLERANCE = (
        1e-5  # if |(u_sol-u_prev)|_inf < EARLY_STOP_TOLERANCE, stop simulation
    )

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
        scenario_name: str,
        rho: float,
        mu: float,
        dt: float,
        T: float,
        f: list,
    ):
        self.solver_name = solver_name
        self.scenario_name = scenario_name

        # Load the solver class
        try:
            solver_module = import_module(f"src.solvers.{solver_name}")
        except ImportError as e:
            available = self._list_available_solvers()
            raise ImportError(
                f"Could not import solver '{solver_name}'. "
                f"Ensure src/solvers/{solver_name}.py exists and all its dependencies are available.\n"
                f"Underlying error: {e}\n"
                f"Available solvers: {available}"
            ) from e

        if not hasattr(solver_module, "Solver"):
            raise ValueError(
                f"Solver module 'src/solvers/{solver_name}.py' does not define a 'Solver' class."
            )

        self.solverClass: type[SolverBase] = solver_module.Solver

        # Instantiate the solver
        try:
            self.solver = self.solverClass(
                self.mesh, dt, rho, mu, f, initial_velocity=self.initial_velocity
            )
        except TypeError as e:
            raise RuntimeError(
                f"Failed to instantiate solver '{solver_name}': {e}. "
                f"Check that the Solver class has the correct constructor signature."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error while initializing solver '{solver_name}': {type(e).__name__}: {e}"
            ) from e

        self.T = T
        self.has_exact_solution = (
            self.__class__.exact_velocity is not Scenario.exact_velocity
        )

        self.dt = dt

    @staticmethod
    def _list_available_solvers():
        """List available solver modules."""
        import os

        solvers_dir = os.path.join(os.path.dirname(__file__), "solvers")
        try:
            files = os.listdir(solvers_dir)
            solvers = [
                f[:-3] for f in files if f.endswith(".py") and not f.startswith("_")
            ]
            return solvers if solvers else ["(none found)"]
        except OSError:
            return ["(could not list)"]

    def setup(self):
        self.solver.setup(self.bcu, self.bcp)

        if self.mesh.comm.rank == 0:
            num_dofs_V = (
                self.solver.V.dofmap.index_map.size_global
                * self.solver.V.dofmap.index_map_bs
            )
            num_dofs_Q = (
                self.solver.Q.dofmap.index_map.size_global
                * self.solver.Q.dofmap.index_map_bs
            )
            total_dofs = num_dofs_V + num_dofs_Q
            print(
                f"DOFs: {total_dofs} (Velocity: {num_dofs_V}, Pressure: {num_dofs_Q})"
            )
            print(f"Suggested cores: {total_dofs / 20000:.1f}")

    def solve(
        self, output_folder: str, afterStepCallback: Callable[[float], None] = None
    ) -> str:
        """
        Runs the time-stepping simulation, returns the path to directory with results in VTX
        format and an error log if the method `exact_velocity` is implemented.

        Args:
            output_folder (str): The absolute path to the directory where simulation results are stored.
            afterStepCallback (Callable[[float], None], optional): A function to be called
                after each time step, receiving the current simulation time as an argument.

        Returns:
            str: The absolute path to the directory where simulation results are stored.
        """

        tqdm = self.get_tqdm()
        mesh = self.mesh
        T = self.T
        solver = self.solver

        progress = (
            tqdm(
                desc="Solving",
                total=float(T),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                file=sys.stdout,
            )
            if mesh.comm.rank == 0
            else None
        )

        # Ensure output folder exists
        if mesh.comm.rank == 0:
            os.makedirs(output_folder, exist_ok=True)

        mesh.comm.barrier()

        u_file = VTXWriter(mesh.comm, f"{output_folder}/v.bp", self.solver.u_sol)
        p_file = VTXWriter(mesh.comm, f"{output_folder}/p.bp", self.solver.p_sol)
        solver.initStressForm()
        wss_file = VTXWriter(
            mesh.comm, f"{output_folder}/wss.bp", self.solver.shear_stress
        )

        t = 0.0
        self.solver.u_sol.interpolate(self.initial_velocity)
        solver.assemble_wss()
        u_file.write(t)
        p_file.write(t)
        wss_file.write(t)

        error_log = None
        if self.has_exact_solution:
            error_log = (
                open(f"{output_folder}/err.txt", "w") if mesh.comm.rank == 0 else None
            )
            u_e = Function(solver.V)
            u_e.interpolate(lambda x: self.exact_velocity(t)(x))
            error = self.compute_error(solver.u_sol, u_e, mesh)

        if error_log:
            error_log.write("t = %.3f: error = %.3g" % (t, error) + "\n")

        i = 0
        while t < T:
            solver.solveStep()

            if progress:
                progress.update(self.dt)

            i += 1
            t += self.dt

            if self.has_exact_solution:
                u_e.interpolate(self.exact_velocity(t))
                error = self.compute_error(u_e, solver.u_sol, mesh)
                if error_log:
                    error_log.write("t = %.3f: error = %.3g" % (t, error) + "\n")

            solver.assemble_wss()
            u_file.write(t)
            p_file.write(t)
            wss_file.write(t)

            if afterStepCallback:
                afterStepCallback(t)

            if (i + 1) % 10 == 0:
                u_diff = solver.u_sol.x.array - solver.u_prev.x.array
                u_diff_norm = np.linalg.norm(u_diff, ord=np.inf)
                u_diff_norm = mesh.comm.allreduce(u_diff_norm, op=MPI.MAX)
                if u_diff_norm < self.EARLY_STOP_TOLERANCE:
                    print(
                        f"Early stopping at t={t:.3f}, "
                        f"because ||u_sol - u_prev||_inf = {u_diff_norm:.3g} < "
                        f"{self.EARLY_STOP_TOLERANCE}"
                    )
                    break

            solver.u_prev.x.array[:] = solver.u_sol.x.array[:]
            solver.p_prev.x.array[:] = solver.p_sol.x.array[:]

        u_file.close()
        p_file.close()
        wss_file.close()
        if error_log:
            error_log.close()
        if progress:
            progress.close()

        return output_folder

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
        error_abs = np.sqrt(
            mesh.comm.allreduce(assemble_scalar(error_form), op=MPI.SUM)
        )
        norm_form = form(inner(u, u) * dx)
        norm = np.sqrt(mesh.comm.allreduce(assemble_scalar(norm_form), op=MPI.SUM))
        return error_abs / norm
