import numpy as np
from dolfinx.fem import Function
from dolfinx.io import gmshio
from mpi4py import MPI

from src.boundaryCondition import BoundaryCondition
from src.geom.stenosis.stenosis import INLET_TAG, OUTLET_TAG, WALL_TAG
from src.scenario import Scenario


def create_experiment_scenario_class(mesh_path, experiment_params, base_params):
    """
    Crea dinámicamente una clase de escenario (LADExperimentScenario)
    con los parámetros del experimento 'congelados'.
    """

    class LADExperimentScenario(Scenario):
        def __init__(
            self,
            solver_name,
            T,
            dt,
            rho=1.06e-3,
            mu=3.5e-3,
            f=[0.0, 0.0, 0.0],
            **kwargs,
        ):
            self._mesh_path = mesh_path
            self.experiment_params = experiment_params
            self.base_params = base_params

            # Cargar malla de GMSH
            self._mesh, self.mt, self.ft = gmshio.read_from_msh(
                str(self._mesh_path), MPI.COMM_WORLD, 0, gdim=3
            )

            # Tags definidos en stenosis.py
            self.INLET_TAG = INLET_TAG
            self.OUTLET_TAG = OUTLET_TAG
            self.WALL_TAG = WALL_TAG

            super().__init__(
                solver_name=solver_name,
                scenario_name="LAD_Experiment",
                rho=rho,
                mu=mu,
                dt=dt,
                T=T,
                f=f,
            )

        @property
        def mesh(self):
            return self._mesh

        @property
        def bcu(self):
            # Determinamos el flujo según si es hiperemia o no
            is_hyper = self.experiment_params.get("hiperemia", False)
            q_val = (
                self.base_params["q_in_hyper"] if is_hyper else self.base_params["q_in"]
            )

            fdim = self.mesh.topology.dim - 1

            # Inlet: Parabolic velocity profile
            # q_val is flow rate (m^3/s). We need to calculate v_max.
            # Q = Area * v_avg = (pi * r^2) * v_avg
            # v_max = 2 * v_avg (for Poiseuille flow)
            # => v_max = 2 * Q / (pi * r^2)

            r_in = self.base_params["radius_in"]
            area = np.pi * r_in**2
            v_avg = q_val / area
            v_max = 2.0 * v_avg

            u_inlet = Function(self.solver.V)

            def inlet_profile_expression(x):
                # x[0] = x (axial), x[1] = y, x[2] = z
                # Center is at (0, 0, 0) assuming mesh generation
                r_sq = x[1] ** 2 + x[2] ** 2
                # Parabolic profile: v = v_max * (1 - r^2 / R^2)
                # Clip negative values just in case
                val = v_max * (1.0 - r_sq / (r_in**2))
                # val = np.maximum(val, 0.0) # Optional
                return np.stack((val, np.zeros_like(val), np.zeros_like(val)))

            u_inlet.interpolate(inlet_profile_expression)

            entities_inflow = self.ft.find(self.INLET_TAG)
            bcu_inflow = BoundaryCondition(u_inlet)
            bcu_inflow.initTopological(fdim, entities_inflow)

            # No-slip en paredes
            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0.0

            entities_walls = self.ft.find(self.WALL_TAG)
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, entities_walls)

            return [bcu_inflow, bcu_walls]

        @property
        def bcp(self):
            # Outlet: Presión (p_terminal)
            fdim = self.mesh.topology.dim - 1
            p_val = self.base_params.get("p_terminal", 0.0)

            p_out = Function(self.solver.Q)
            p_out.x.array[:] = float(p_val)

            outflow_entities = self.ft.find(self.OUTLET_TAG)
            bc_outflow = BoundaryCondition(p_out)
            bc_outflow.initTopological(fdim, outflow_entities)

            return [bc_outflow]

        def initial_velocity(self, x):
            return np.zeros((3, x.shape[1]), dtype=np.float64)

    return LADExperimentScenario
