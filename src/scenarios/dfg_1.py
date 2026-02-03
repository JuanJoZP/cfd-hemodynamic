import os

import gmsh
import numpy as np
from ufl import Measure, FacetNormal, grad, inner, as_vector
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.fem import Function, form, assemble_scalar
from dolfinx.io import XDMFFile
from dolfinx.io import gmshio
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc

from src.boundaryCondition import BoundaryCondition
from src.scenario import Scenario


class DFG1Benchmark(Scenario):
    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4
    obstacle_marker = 5

    def __init__(
        self, solver_name, dt, T, f: tuple[float, float] = (0, 0), *, rho=1, mu=1 / 1000
    ):
        self._mesh: Mesh = None
        self._ft = None
        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None
        self.mu = mu
        self.rho = rho
        super().__init__(solver_name, "dfg_1", rho, mu, dt, T, f)

        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )
        self.setup()

    @property
    def mesh(self):
        if not self._mesh:
            mesh_file = "meshes/pipe_cylinder.xdmf"
            if os.path.exists(mesh_file):
                mesh_comm = MPI.COMM_WORLD
                with XDMFFile(mesh_comm, mesh_file, "r") as xdmf:
                    self._mesh = xdmf.read_mesh(name="Grid")
                    self._ft = xdmf.read_meshtags(self._mesh, name="Facet markers")

            else:
                self._mesh, self._ft = self.generate_mesh()

        return self._mesh

    @property
    def bcu(self):
        if not self._bcu:
            fdim = self.mesh.topology.dim - 1
            u_inlet = Function(self.solver.V)
            u_inlet.interpolate(self.inlet_velocity)
            entities_inflow = self._ft.find(self.inlet_marker)
            bcu_inflow = BoundaryCondition(u_inlet)
            bcu_inflow.initTopological(fdim, entities_inflow)

            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0
            entities_walls = self._ft.find(self.wall_marker)
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, entities_walls)

            entities_obstacle = self._ft.find(self.obstacle_marker)
            bcu_obstacle = BoundaryCondition(u_nonslip)
            bcu_obstacle.initTopological(fdim, entities_obstacle)

            self._bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            fdim = self.mesh.topology.dim - 1
            pr = Function(self.solver.Q)
            pr.x.array[:] = 0
            outflow_facets = self._ft.find(self.outlet_marker)
            bc_outflow = BoundaryCondition(pr)
            bc_outflow.initTopological(fdim, outflow_facets)

            self._bcp = [bc_outflow]

        return self._bcp

    def initial_velocity(self, x):
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        return values

    def generate_mesh(self):
        gmsh.initialize()
        L = 2.2
        H = 0.41
        c_x = c_y = 0.2
        r = 0.05
        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0
        if mesh_comm.rank == model_rank:
            rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
            obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
            fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
            gmsh.model.occ.synchronize()

        inflow, outflow, walls, obstacle = [], [], [], []
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
                elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(
                    center_of_mass, [L / 2, 0, 0]
                ):
                    walls.append(boundary[1])
                else:
                    obstacle.append(boundary[1])
            gmsh.model.addPhysicalGroup(1, walls, self.wall_marker)
            gmsh.model.setPhysicalName(1, self.wall_marker, "Walls")
            gmsh.model.addPhysicalGroup(1, inflow, self.inlet_marker)
            gmsh.model.setPhysicalName(1, self.inlet_marker, "Inlet")
            gmsh.model.addPhysicalGroup(1, outflow, self.outlet_marker)
            gmsh.model.setPhysicalName(1, self.outlet_marker, "Outlet")
            gmsh.model.addPhysicalGroup(1, obstacle, self.obstacle_marker)
            gmsh.model.setPhysicalName(1, self.obstacle_marker, "Obstacle")

        # variable resolution, finer near the obstacle
        res_min = r / 6
        if mesh_comm.rank == model_rank:
            distance_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
            threshold_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
            gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", H / 13)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        if mesh_comm.rank == model_rank:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.RecombineAll", 0)
            gmsh.model.mesh.generate(gdim)
            gmsh.model.mesh.setOrder(1)
            gmsh.model.mesh.optimize("Netgen")

        mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        mesh.name = "Grid"
        ft.name = "Facet markers"

        return mesh, ft

    @staticmethod
    def inlet_velocity(x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * 0.3 * x[1] * (0.41 - x[1]) / (0.41**2)
        return values

    def solve(self, output_folder, afterStepCallback=None):
        out_path = super().solve(output_folder, afterStepCallback)

        # Post-processing calculations
        dObs = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self._ft,
            subdomain_id=self.obstacle_marker,
        )
        u = self.solver.u_sol
        p = self.solver.p_sol
        n = -FacetNormal(self.mesh)

        tangent = as_vector((n[1], -n[0]))
        u_t = inner(tangent, u)

        mu = self.mu

        F_D_form = form((mu * inner(grad(u_t), n) * n[1] - p * n[0]) * dObs)
        F_L_form = form(-(mu * inner(grad(u_t), n) * n[0] + p * n[1]) * dObs)

        F_D = self.mesh.comm.allreduce(assemble_scalar(F_D_form), op=MPI.SUM)
        F_L = self.mesh.comm.allreduce(assemble_scalar(F_L_form), op=MPI.SUM)

        if self.mesh.comm.rank == 0:
            print(f"Drag: {500*F_D}")
            print(f"Lift: {500*F_L}")

            # Save to file in output_folder
            with open(f"{out_path}/drag_lift.txt", "w") as f:
                f.write(f"Drag: {500*F_D}\n")
                f.write(f"Lift: {500*F_L}\n")

        tree = bb_tree(self.mesh, self.mesh.geometry.dim)
        points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
        cell_candidates = compute_collisions_points(tree, points)
        colliding_cells = compute_colliding_cells(self.mesh, cell_candidates, points)
        front_cells = colliding_cells.links(0)
        back_cells = colliding_cells.links(1)

        p_front = None
        if len(front_cells) > 0:
            p_front = p.eval(points[0], front_cells[:1])
        p_front = self.mesh.comm.gather(p_front, root=0)
        p_back = None
        if len(back_cells) > 0:
            p_back = p.eval(points[1], back_cells[:1])
        p_back = self.mesh.comm.gather(p_back, root=0)

        if self.mesh.comm.rank == 0:
            p_diff = 0
            # Simplify gathering logic
            found_front = False
            for pressure in p_front:
                if pressure is not None:
                    p_diff = pressure[0]
                    found_front = True
                    break

            found_back = False
            for pressure in p_back:
                if pressure is not None:
                    p_diff -= pressure[0]
                    found_back = True
                    break

            if found_front and found_back:
                print(f"Pressure difference: {p_diff}")
                with open(f"{out_path}/pressure_diff.txt", "w") as f:
                    f.write(f"Pressure difference: {p_diff}\n")
            else:
                print(
                    "Could not calculate pressure difference (points not found on rank 0 or gathered correctly)"
                )

        return out_path
