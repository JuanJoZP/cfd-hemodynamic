from src.simulationBase import SimulationBase
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import Mesh, locate_entities_boundary
from dolfinx.fem import Function
import os

from src.boundaryCondition import BoundaryCondition

solver_name = "stabilized_schur"
simulation_name = "stenosis"
# rho = 1055  # kg/m^3
# mu = 0.004  # 4 centipoise = 0.004 Pa.s
rho = 1.05
mu = 0.004
dt = 1 / 50
T = 50

# revisar unidades de todo esto
res = 0.02
L = 5
H = 1
major_axis = 0.4
minor_axis = 0.2
x_position_stenosis = 1
inlet_max_velocity = 3


class StenosisSimulation(SimulationBase):
    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4

    def __init__(
        self,
        solver_name,
        rho=1055,
        mu=0.004,
        dt=1 / 100,
        T=5,
        f: tuple[float, float] = (0, 0),
        inlet_max_velocity=1.5,
        mesh_options={},
    ):
        self._mesh: Mesh = None
        self.mesh_options = mesh_options
        self.inlet_max_velocity = inlet_max_velocity
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
            self._mesh, self._ft = self.generate_mesh(**self.mesh_options)

        return self._mesh

    @property
    def bcu(self):
        if not self._bcu:
            fdim = self.mesh.topology.dim - 1
            u_inlet = Function(self.solver.V)
            u_inlet.interpolate(
                self.inlet_velocity(
                    self.inlet_max_velocity, self.mesh_options.get("H", 1)
                )
            )
            entities_inflow = self._ft.find(self.inlet_marker)
            bcu_inflow = BoundaryCondition(u_inlet)
            bcu_inflow.initTopological(fdim, entities_inflow)

            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0
            entities_walls = self._ft.find(self.wall_marker)
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, entities_walls)

            self._bcu = [bcu_inflow, bcu_walls]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            fdim = self.mesh.topology.dim - 1
            pr = Function(self.solver.Q)
            pr.x.array[:] = 0
            outflow_entities = self._ft.find(self.outlet_marker)
            bc_outflow = BoundaryCondition(pr)
            bc_outflow.initTopological(fdim, outflow_entities)

            self._bcp = [bc_outflow]

        return self._bcp

    def initial_velocity(self, x):
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        return values

    def generate_mesh(self, **kwargs):
        gmsh.initialize()

        L = kwargs.get("L", 3)
        H = kwargs.get("H", 1)
        res = kwargs.get("res", 0.1)
        major_axis = kwargs.get("major_axis", 0.4)
        minor_axis = kwargs.get("minor_axis", 0.2)
        x_position_stenosis = kwargs.get("x_position_stenosis", 1)

        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0
        if mesh_comm.rank == model_rank:
            rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
            stenosis_lower = gmsh.model.occ.addEllipse(
                x_position_stenosis, 0, 0, major_axis, minor_axis
            )
            stenosis_upper = gmsh.model.occ.addEllipse(
                x_position_stenosis, H, 0, major_axis, minor_axis
            )
            loop1 = gmsh.model.occ.addCurveLoop([stenosis_lower])
            loop2 = gmsh.model.occ.addCurveLoop([stenosis_upper])

            surf1 = gmsh.model.occ.addPlaneSurface([loop1])
            surf2 = gmsh.model.occ.addPlaneSurface([loop2])

            gmsh.model.occ.synchronize()

            fluid = gmsh.model.occ.cut(
                [(gdim, rectangle)], [(gdim, surf1), (gdim, surf2)]
            )
            gmsh.model.occ.synchronize()

        inflow, outflow, walls = [], [], []
        if mesh_comm.rank == model_rank:
            volumes = gmsh.model.getEntities(dim=gdim)
            assert len(volumes) == 1
            gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], self.fluid_marker)
            gmsh.model.setPhysicalName(volumes[0][0], self.fluid_marker, "Fluid")

            boundaries = gmsh.model.getBoundary(volumes, oriented=False)
            for boundary in boundaries:
                center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
                if np.allclose(center_of_mass, [0, H / 2, 0]):
                    inflow.append(boundary[1])
                elif np.allclose(center_of_mass, [L, H / 2, 0]):
                    outflow.append(boundary[1])
                else:
                    walls.append(boundary[1])

            gmsh.model.addPhysicalGroup(1, walls, self.wall_marker)
            gmsh.model.setPhysicalName(1, self.wall_marker, "Walls")
            gmsh.model.addPhysicalGroup(1, inflow, self.inlet_marker)
            gmsh.model.setPhysicalName(1, self.inlet_marker, "Inlet")
            gmsh.model.addPhysicalGroup(1, outflow, self.outlet_marker)
            gmsh.model.setPhysicalName(1, self.outlet_marker, "Outlet")

        # resolucion
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res * 1.5)

        # variable resolution, finer near the walls
        # res_min = minor_axis / 5
        # if mesh_comm.rank == model_rank:
        #     distance_field = gmsh.model.mesh.field.add("Distance")
        #     gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", walls)
        #     threshold_field = gmsh.model.mesh.field.add("Threshold")
        #     gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        #     gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        #     gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", res_min * 1.5)
        #     gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", minor_axis / 2)
        #     gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 0.8 * H)
        #     min_field = gmsh.model.mesh.field.add("Min")
        #     gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        #     gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        if mesh_comm.rank == model_rank:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            # gmsh.option.setNumber("Mesh.RecombineAll", 1)
            # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh.model.mesh.generate(gdim)
            gmsh.model.mesh.setOrder(2)
            gmsh.model.mesh.optimize("Netgen")

        mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        mesh.name = "Grid"
        ft.name = "Facet markers"
        gmsh.finalize()

        with XDMFFile(
            mesh_comm, f"meshes/stenosis{major_axis}_{minor_axis}.xdmf", "w"
        ) as xdmf_file:
            xdmf_file.write_mesh(mesh)
            xdmf_file.write_meshtags(ft, mesh.geometry)

        from dolfinx.io import VTXWriter

        mesh_file = VTXWriter(
            mesh_comm, f"meshes/stenosis{major_axis}_{minor_axis}.bp", mesh
        )
        mesh_file.write(0)

        return mesh, ft

    @staticmethod
    def inlet_velocity(v_max, y_max):
        def velocity(x):
            values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = 4 * v_max * x[1] * (y_max - x[1]) / (y_max**2)
            return values

        return velocity


simulation = StenosisSimulation(
    solver_name,
    rho,
    mu,
    dt,
    T,
    mesh_options={
        "L": L,
        "H": H,
        "res": res,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "x_position_stenosis": x_position_stenosis,
    },
    inlet_max_velocity=inlet_max_velocity,
)
print(
    simulation.solver.V.dofmap.index_map.size_global
    + simulation.solver.Q.dofmap.index_map.size_global
)
simulation.solve()
