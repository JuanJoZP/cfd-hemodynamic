from src.scenario import Scenario
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import Mesh
from dolfinx.fem import Function

from src.boundaryCondition import BoundaryCondition


class MicrovasculatureSimulation(Scenario):
    fluid_tag = 7
    inlet_tag = 8
    outlet1_tag = 9
    outlet2_tag = 10
    wall_tag = 11

    # Constants
    rho_real = 1055.0
    mu_real = 3.5e-3
    r_mesh_in = 0.003918604
    r_mesh_out2 = 0.000922768
    L_c = (100 / r_mesh_in) / 1e6
    U_c = 0.01

    def __init__(
        self,
        solver_name,
        dt,
        T,
        f: tuple[float, float, float] = (0, 0, 0),
        v_inlet=1.5,
        p_outlet1=0,
        p_outlet2=0,
        *,
        rho=None,
        mu=None,
        **kwargs,
    ):
        self._mesh: Mesh = None
        self._ft = None
        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None

        # Recalculate parameters
        Re = self.rho_real * self.U_c * self.L_c / self.mu_real
        rho_adim = 1
        mu_adim = 1 / Re
        p_c = self.rho_real * self.U_c**2

        self.v_inlet = float(v_inlet)
        self.p_outlet1_adim = float(p_outlet1) / p_c
        self.p_outlet2_adim = float(p_outlet2) / p_c

        if MPI.COMM_WORLD.rank == 0:
            print(f"MicrovasculatureSimulation (Simple Bifurcation): Reynolds = {Re}")
            print(f"Using calculated rho={rho_adim}, mu={mu_adim}")

        super().__init__(solver_name, "simple_bifurcation", rho_adim, mu_adim, dt, T, f)

        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )
        self.setup()

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh, _, self._ft = gmshio.read_from_msh(
                "simple_bifurcation.msh", MPI.COMM_WORLD, 0, gdim=3
            )

        return self._mesh

    @property
    def bcu(self):
        if not self._bcu:
            fdim = self.mesh.topology.dim - 1

            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0
            entities_walls = self._ft.find(self.wall_tag)
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, entities_walls)

            u_inlet = Function(self.solver.V)
            u_inlet.interpolate(self.inlet_velocity(self.v_inlet, self.r_mesh_in))
            entities_inflow = self._ft.find(self.inlet_tag)
            bcu_inflow = BoundaryCondition(u_inlet)
            bcu_inflow.initTopological(fdim, entities_inflow)

            self._bcu = [bcu_walls, bcu_inflow]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            fdim = self.mesh.topology.dim - 1

            # outlet 1
            p_outlet1_func = Function(self.solver.Q)
            p_outlet1_func.x.array[:] = self.p_outlet1_adim
            outlet1_entities = self._ft.find(self.outlet1_tag)
            bc_outlet1 = BoundaryCondition(p_outlet1_func)
            bc_outlet1.initTopological(fdim, outlet1_entities)

            # outlet 2
            p_outlet2_func = Function(self.solver.Q)
            p_outlet2_func.x.array[:] = self.p_outlet2_adim
            outlet2_entities = self._ft.find(self.outlet2_tag)
            bc_outlet2 = BoundaryCondition(p_outlet2_func)
            bc_outlet2.initTopological(fdim, outlet2_entities)

            self._bcp = [bc_outlet1, bc_outlet2]

        return self._bcp

    def initial_velocity(self, x):
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        return values

    @staticmethod
    def inlet_velocity(v_max, r_max):
        def velocity(x):
            values = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
            r = (x[0] ** 2 + x[2] ** 2) ** (1 / 2)
            values[1] = v_max * (1 - (r / r_max) ** 2)
            return values

        return velocity
