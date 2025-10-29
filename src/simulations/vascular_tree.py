from src.simulationBase import SimulationBase
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import Mesh
from dolfinx.fem import Function

from src.boundaryCondition import BoundaryCondition

solver_name = "stabilized_schur"
simulation_name = "vascular_tree"

dt = 1 / 1200
T = 10

rho_real = 1055.0  # kg/m^3
mu_real = 3.5e-3  # Pa·s


# radio de los vasos en unidades de la malla
r_mesh_in = 0.003918604
r_mesh_out2 = 0.000922768

# reescalamos la ecuación
# U_real = U * U_c (donde U es adimensional solucion de la ec), igual para L y p

# seteamos L_c y U_c arbitrariamente para ajustar la malla y la velocidad a valores fisiologicos
L_c = (100 / r_mesh_in) / 1e6  # setea r_mesh_in = 100 micrometros y pasa a metros
U_c = 0.01  # m/s

Re = rho_real * U_c * L_c / mu_real

# si dividimos la ecuacion entre rho*(U_c)^2/L_c, los coefs del termino temporal y convectivo
# se cancelan, queda p_c/(rho * (U_c)^2) acompañando al termino de gradiente de presión
# y queda mu / (rho * U_c * L_c) en vez de solo mu en el termino viscoso.
# es decir que si escojo p_c = rho * (U_c)^2, entonces rho y mu deben cambiar a:

rho_adim = 1
mu_adim = 1 / Re

p_c = rho_real * U_c**2

r_in = r_mesh_in * L_c
r_out2 = r_mesh_out2 * L_c

# presiones reales a preescribir (Pascales)
p_inlet = 5300  # ~40mmHg
p_outlet1 = 4200  # ~ 31mmHg
p_outlet2 = 1333  # ~ 10mmHg

# pasamos a valores adimensionales
p_inlet_adim = p_inlet / p_c
p_outlet1_adim = p_outlet1 / p_c
p_outlet2_adim = p_outlet2 / p_c

print("Número de Reynolds para los parametros dados:", Re)


class MicrovasculatureSimulation(SimulationBase):
    fluid_tag = 7
    inlet_tag = 8
    outlet1_tag = 9
    outlet2_tag = 10
    wall_tag = 11

    def __init__(
        self,
        solver_name,
        rho,
        mu,
        dt,
        T,
        f: tuple[float, float] = (0, 0),
    ):
        self._mesh: Mesh = None
        self._ft = None
        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None
        super().__init__(solver_name, simulation_name, rho, mu, dt, T, f)

        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )
        self.setup()

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh, _, self._ft = gmshio.read_from_msh(
                "vascular_tree.msh", MPI.COMM_WORLD, 0, gdim=3
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

            self._bcu = [bcu_walls]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            fdim = self.mesh.topology.dim - 1

            # inlet
            p_inlet_func = Function(self.solver.Q)
            p_inlet_func.x.array[:] = p_inlet_adim
            inlet_entities = self._ft.find(self.inlet_tag)
            bc_inlet = BoundaryCondition(p_inlet_func)
            bc_inlet.initTopological(fdim, inlet_entities)

            # outlet 1
            p_outlet1_func = Function(self.solver.Q)
            p_outlet1_func.x.array[:] = p_outlet1_adim
            outlet1_entities = self._ft.find(self.outlet1_tag)
            bc_outlet1 = BoundaryCondition(p_outlet1_func)
            bc_outlet1.initTopological(fdim, outlet1_entities)

            # outlet 2
            p_outlet2_func = Function(self.solver.Q)
            p_outlet2_func.x.array[:] = p_outlet2_adim
            outlet2_entities = self._ft.find(self.outlet2_tag)
            bc_outlet2 = BoundaryCondition(p_outlet2_func)
            bc_outlet2.initTopological(fdim, outlet2_entities)

            self._bcp = [bc_inlet, bc_outlet1, bc_outlet2]

        return self._bcp

    def initial_velocity(self, x):
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        return values


simulation = MicrovasculatureSimulation(
    solver_name,
    rho_adim,
    mu_adim,
    dt,
    T,
    f=(0, 0, 0),
)
print(
    simulation.solver.V.dofmap.index_map.size_global
    + simulation.solver.Q.dofmap.index_map.size_global
)
simulation.solve()
