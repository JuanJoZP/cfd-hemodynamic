from src.simulationBase import SimulationBase
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.io import gmshio
from dolfinx.mesh import Mesh
from dolfinx.fem import Function

from src.boundaryCondition import BoundaryCondition

solver_name = "stabilized_schur"
simulation_name = "vascular_tree"

dt = 1 / 200
T = 10

rho_real = 1055.0  # kg/m^3
mu_real = 3.5e-3  # Pa·s


# radio de los vasos en unidades de la malla
r_mesh_in = 0.003918604
# r_mesh_out2 = 0.000922768

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
# r_out2 = r_mesh_out2 * L_c


print("Número de Reynolds para los parametros dados:", Re)


class MicrovasculatureSimulation(SimulationBase):
    inlet_tag = 1
    outlet_tag = 2
    wall_tag = 3

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
            self._mesh, self._ft, _ = gmshio.read_from_msh(
                "src/geom/vessels.msh", MPI.COMM_WORLD, 0, gdim=3
            )

        return self._mesh

    @property
    def bcu(self):
        if not self._bcu:
            fdim = self.mesh.geometry.dim - 1
            self.mesh.topology.create_connectivity(2, 2)

            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0
            entities_walls = self._ft.find(self.wall_tag)
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, entities_walls)

            u_inlet = Function(self.solver.V)
            u_inlet.interpolate(self.inlet_velocity(U_c, r_mesh_in))
            entities_inflow = self._ft.find(self.inlet_tag)
            bcu_inflow = BoundaryCondition(u_inlet)
            bcu_inflow.initTopological(fdim, entities_inflow)

            self._bcu = [bcu_inflow, bcu_walls]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            fdim = self.mesh.topology.dim - 1

            # outlets
            p_outlet_func = Function(self.solver.Q)
            p_outlet_func.x.array[:] = 0
            outlet_entities = self._ft.find(self.outlet_tag)
            bc_outlet = BoundaryCondition(p_outlet_func)
            bc_outlet.initTopological(fdim, outlet_entities)

            self._bcp = [bc_outlet]

        return self._bcp

    def initial_velocity(self, x):
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        return values

    @staticmethod
    def inlet_velocity(v_max, r_max):
        inlet_normal = np.array(
            [[0.07961727999999998], [-0.10554240000000004], [0.015753600000000034]]
        )
        inlet_center = np.array([[0.0], [2.0], [2.0]])

        # TODO: este codigo esta sin probar
        def velocity(x):
            values = np.zeros((3, x.shape[1]), dtype=np.float64)
            inlet_normal_unit = inlet_normal / np.linalg.norm(inlet_normal)
            r_vector = x - inlet_center
            r = np.linalg.norm(r_vector)
            magnitude = v_max * (1 - (r / r_max) ** 2)
            values[:] = magnitude * inlet_normal_unit
            return values

        return velocity


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
