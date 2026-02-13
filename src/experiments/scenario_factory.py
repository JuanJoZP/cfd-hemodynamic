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

            bc_type = self.experiment_params.get("bc_type", "default")

            fdim = self.mesh.topology.dim - 1

            # Common: Wall No-Slip
            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0.0
            entities_walls = self.ft.find(self.WALL_TAG)
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, entities_walls)

            bcs = [bcu_walls]

            # Inlet Configs
            entities_inflow = self.ft.find(self.INLET_TAG)

            if "inlet_velocity_parabolic" in bc_type or bc_type == "default":
                # Parabolic velocity profile
                r_in = self.base_params["radius_in"]
                area = np.pi * r_in**2
                v_avg = q_val / area
                v_max = 2.0 * v_avg  # Poiseuille

                u_inlet = Function(self.solver.V)

                def inlet_profile_expression(x):
                    r_sq = x[1] ** 2 + x[2] ** 2
                    val = v_max * (1.0 - r_sq / (r_in**2))
                    return np.stack((val, np.zeros_like(val), np.zeros_like(val)))

                u_inlet.interpolate(inlet_profile_expression)
                bcu_inflow = BoundaryCondition(u_inlet)
                bcu_inflow.initTopological(fdim, entities_inflow)
                bcs.append(bcu_inflow)

            elif "inlet_velocity_constant" in bc_type:
                # Constant velocity profile
                r_in = self.base_params["radius_in"]
                area = np.pi * r_in**2
                v_avg = q_val / area

                u_inlet = Function(self.solver.V)
                u_inlet.x.array[:] = 0.0  # reset

                # We need to set x-component to v_avg everywhere (assuming flow is along X)
                # However, Function.x.array is flat.
                # Use interpolate with constant value
                def constant_profile(x):
                    return np.stack(
                        (
                            np.full_like(x[0], v_avg),
                            np.zeros_like(x[0]),
                            np.zeros_like(x[0]),
                        )
                    )

                u_inlet.interpolate(constant_profile)

                bcu_inflow = BoundaryCondition(u_inlet)
                bcu_inflow.initTopological(fdim, entities_inflow)
                bcs.append(bcu_inflow)

            elif "inlet_pressure" in bc_type:
                # Inlet Pressure -> No Dirichlet BC for Velocity on Inlet
                pass

            # Outlet Configs for Velocity
            if "outlet_velocity_zero" in bc_type:
                # Zero velocity at outlet (Blocked / Wall-like)
                u_outlet = Function(self.solver.V)
                u_outlet.x.array[:] = 0.0
                entities_outlet = self.ft.find(self.OUTLET_TAG)
                bcu_outlet = BoundaryCondition(u_outlet)
                bcu_outlet.initTopological(fdim, entities_outlet)
                bcs.append(bcu_outlet)

            return bcs

        @property
        def bcp(self):
            bc_type = self.experiment_params.get("bc_type", "default")
            fdim = self.mesh.topology.dim - 1
            bcs = []

            # Outlet Pressure Configs
            if "outlet_pressure" in bc_type or bc_type == "default":
                p_val = self.base_params.get("p_terminal", 0.0)
                p_out = Function(self.solver.Q)
                p_out.x.array[:] = float(p_val)
                outflow_entities = self.ft.find(self.OUTLET_TAG)
                bc_outflow = BoundaryCondition(p_out)
                bc_outflow.initTopological(fdim, outflow_entities)
                bcs.append(bc_outflow)

            # Inlet Pressure Configs
            if "inlet_pressure" in bc_type:
                # Set inlet pressure
                # We assume p_inlet is provided or derived.
                # For now let's say p_inlet is another param, or repurpose a param.
                p_in_val = self.experiment_params.get(
                    "p_inlet", 13332.2
                )  # ~100 mmHg default? Or just higher than terminal.

                p_in = Function(self.solver.Q)
                p_in.x.array[:] = float(p_in_val)
                inflow_entities = self.ft.find(self.INLET_TAG)
                bc_inflow = BoundaryCondition(p_in)
                bc_inflow.initTopological(fdim, inflow_entities)
                bcs.append(bc_inflow)

            return bcs

        def initial_velocity(self, x):
            return np.zeros((3, x.shape[1]), dtype=np.float64)

    return LADExperimentScenario
