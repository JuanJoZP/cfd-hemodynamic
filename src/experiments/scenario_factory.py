import numpy as np
from dolfinx.fem import Function
from dolfinx.io import gmshio
from mpi4py import MPI

from src.boundaryCondition import BoundaryCondition
from src.geom.stenosis.stenosis import INLET_TAG, OUTLET_TAG, WALL_TAG
from src.scenario import Scenario


def _parse_bc_type(bc_type_raw):
    """
    Parse bc_type dict into (bc_inlet, bc_outlet) strings.

    Expected YAML format:
        bc_type:
          inlet: "velocity_parabolic"  # velocity_parabolic | velocity_constant | pressure
          outlet: "pressure"           # pressure | none | velocity_zero
    """
    bc_inlet = bc_type_raw.get("inlet", "velocity_parabolic")
    bc_outlet = bc_type_raw.get("outlet", "pressure")
    return bc_inlet, bc_outlet


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
            is_hyper = self.experiment_params.get(
                "hyperemia", self.base_params.get("hyperemia", False)
            )
            q_val = (
                self.base_params["q_in_hyper"] if is_hyper else self.base_params["q_in"]
            )

            bc_type_raw = self.experiment_params.get(
                "bc_type", self.base_params.get("bc_type", {})
            )
            bc_inlet, bc_outlet = _parse_bc_type(bc_type_raw)

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

            if bc_inlet in ("velocity_parabolic", "default"):
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

            elif bc_inlet == "velocity_constant":
                # Constant (plug) velocity profile
                r_in = self.base_params["radius_in"]
                area = np.pi * r_in**2
                v_avg = q_val / area

                u_inlet = Function(self.solver.V)
                u_inlet.x.array[:] = 0.0

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

            elif bc_inlet == "pressure":
                # Inlet Pressure -> No Dirichlet BC for Velocity on Inlet
                pass

            # Outlet Velocity BC (only for velocity_zero)
            if bc_outlet == "velocity_zero":
                # Zero velocity at outlet (wall-like blockage)
                u_outlet = Function(self.solver.V)
                u_outlet.x.array[:] = 0.0
                entities_outlet = self.ft.find(self.OUTLET_TAG)
                bcu_outlet = BoundaryCondition(u_outlet)
                bcu_outlet.initTopological(fdim, entities_outlet)

                bcs.append(bcu_outlet)

            return bcs

        @property
        def bcp(self):
            bc_type_raw = self.experiment_params.get(
                "bc_type", self.base_params.get("bc_type", {})
            )
            bc_inlet, bc_outlet = _parse_bc_type(bc_type_raw)
            fdim = self.mesh.topology.dim - 1
            bcs = []

            # Outlet Pressure BC
            if bc_outlet in ("pressure", "default"):
                p_val = self.base_params.get("p_terminal", 0.0)
                p_out = Function(self.solver.Q)
                p_out.x.array[:] = float(p_val)
                outflow_entities = self.ft.find(self.OUTLET_TAG)
                bc_outflow = BoundaryCondition(p_out)
                bc_outflow.initTopological(fdim, outflow_entities)
                bcs.append(bc_outflow)

            # Inlet Pressure BC
            if bc_inlet == "pressure":
                p_in_val = self.experiment_params.get(
                    "p_inlet", self.base_params.get("p_inlet", 13332.2)
                )
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
