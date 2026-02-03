from src.scenario import Scenario
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from dolfinx.io import VTXWriter

from src.boundaryCondition import BoundaryCondition


class StenosisSimulation(Scenario):
    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4

    stenosis_grades = {
        "mild": {"major_axis": 0.0024, "minor_axis": 0.000375, "y_offset": 0.0},
        "moderate": {"major_axis": 0.0024, "minor_axis": 0.00075, "y_offset": 0.0},
        "severe": {"major_axis": 0.0044, "minor_axis": 0.001125, "y_offset": 0.001},
    }

    def __init__(
        self,
        solver_name,
        dt,
        T,
        f: tuple[float, float] = (0, 0),
        grade="severe",
        inlet_max_velocity=1.5,
        *,
        rho=1,
        mu=3.3e-6,
        **kwargs,
    ):
        self._mesh: Mesh = None
        self._ft = None
        self.inlet_max_velocity = float(inlet_max_velocity)

        # Setup mesh options
        self.mesh_options = kwargs.copy()

        # Defaults
        defaults = {"L": 0.03, "H": 0.003, "res": 0.0001, "x_position_stenosis": 0.005}
        for k, v in defaults.items():
            if k not in self.mesh_options:
                self.mesh_options[k] = v

        # Apply grade parameters
        grade_params = self.stenosis_grades.get(grade, self.stenosis_grades["severe"])
        for k, v in grade_params.items():
            # If user explicitly provided e.g. major_axis, keep it. Otherwise use grade default.
            if k not in kwargs:
                self.mesh_options[k] = v

        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None
        super().__init__(solver_name, "stenosis", rho, mu, dt, T, f)

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

    def solve(self, output_folder, afterStepCallback=None):
        def update_dt(t):
            if t > 10 * self.dt and self.dt < 1 / 1000:
                self.dt = 1 / 200
                self.solver.dt.value = self.dt
            if afterStepCallback:
                afterStepCallback(t)

        return super().solve(output_folder, update_dt)

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
        y_offset = kwargs.get("y_offset", 0)

        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0
        if mesh_comm.rank == model_rank:
            rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
            stenosis_lower = gmsh.model.occ.addEllipse(
                x_position_stenosis, -y_offset, 0, major_axis, minor_axis + y_offset
            )
            stenosis_upper = gmsh.model.occ.addEllipse(
                x_position_stenosis, H + y_offset, 0, major_axis, minor_axis + y_offset
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
            gmsh.model.addPhysicalGroup(
                volumes[0][0], [volumes[0][1]], self.fluid_marker
            )
            gmsh.model.setPhysicalName(volumes[0][0], self.fluid_marker, "Fluid")

            boundaries = gmsh.model.getBoundary(volumes, oriented=False)
            for boundary in boundaries:
                center_of_mass = gmsh.model.occ.getCenterOfMass(
                    boundary[0], boundary[1]
                )
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

        if mesh_comm.rank == model_rank:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.model.mesh.generate(gdim)
            gmsh.model.mesh.setOrder(1)
            gmsh.model.mesh.optimize("Netgen")

        mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        mesh.name = "Grid"
        ft.name = "Facet markers"
        gmsh.finalize()

        return mesh, ft

    @staticmethod
    def inlet_velocity(v_max, y_max):
        def velocity(x):
            values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = 4 * v_max * x[1] * (y_max - x[1]) / (y_max**2)
            return values

        return velocity
