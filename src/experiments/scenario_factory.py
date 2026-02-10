import numpy as np
from src.scenario import Scenario
from src.boundaryCondition import BoundaryCondition
from dolfinx.io import gmshio
from mpi4py import MPI
import yaml
from pathlib import Path

def create_experiment_scenario_class(mesh_path, experiment_params, base_params):
    """
    Crea dinámicamente una clase de escenario (LADExperimentScenario) 
    con los parámetros del experimento 'congelados'.
    """
    
    class LADExperimentScenario(Scenario):
        def __init__(self, solver_name, T, dt, rho=1.06e-3, mu=3.5e-3, f=[0.0, 0.0, 0.0], **kwargs):
            self._mesh_path = mesh_path
            self.experiment_params = experiment_params
            self.base_params = base_params
            
            # Cargar malla de GMSH
            self._mesh, self.mt, self.ft = gmshio.read_from_msh(
                str(self._mesh_path), MPI.COMM_WORLD, 0, gdim=3
            )
            
            # Tags definidos en stenosis.py
            self.INLET_TAG = 1
            self.OUTLET_TAG = 2
            self.WALL_TAG = 3

            super().__init__(
                solver_name=solver_name,
                scenario_name="LAD_Experiment",
                rho=rho,
                mu=mu,
                dt=dt,
                T=T,
                f=f
            )

        @property
        def mesh(self):
            return self._mesh

        @property
        def bcu(self):
            # Determinamos el flujo según si es hiperemia o no
            is_hyper = self.experiment_params.get("hiperemia", False)
            q_val = self.base_params["q_in_hyper"] if is_hyper else self.base_params["q_in"]
            
            bcs = [
                # Inlet: Velocidad constante (ejemplo placeholder)
                BoundaryCondition("dirichlet", self.INLET_TAG, value=[float(q_val), 0.0, 0.0], variable="u"),
                # No-slip en paredes
                BoundaryCondition("dirichlet", self.WALL_TAG, value=[0.0, 0.0, 0.0], variable="u")
            ]
            return bcs

        @property
        def bcp(self):
            # Outlet: Presión (p_terminal)
            p_val = self.base_params.get("p_terminal", 0.0)
            return [
                BoundaryCondition("dirichlet", self.OUTLET_TAG, value=float(p_val), variable="p")
            ]

        def initial_velocity(self, x):
            return np.zeros((3, x.shape[1]), dtype=np.float64)

    return LADExperimentScenario
