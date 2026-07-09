import os
from typing import Callable, Optional

import gmsh
import numpy as np
from dolfinx.fem import Constant, Function
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc

from src.boundaryCondition import BoundaryCondition
from src.scenario import Scenario


class DFG2D1(Scenario):
    """
    DFG 2D-1 Benchmark Scenario: Stationary Flow around a cylinder.
    Geometry: [0, 2.2] x [0, 0.41] with a cylinder at (0.2, 0.2) and r=0.05.
    L_char = 0.1 (diameter), U_max = 0.3, U_mean = 0.2, nu = 0.001 -> Re = 20.
    Inlet velocity: 4 * U_max * y * (H - y) / H^2.
    """

    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4
    obstacle_marker = 5

    def __init__(
        self,
        solver_name,
        dt,
        T,
        f: tuple[float, float] = (0, 0),
        *,
        rho=1.0,
        mu=0.001,
        res=0.005,
    ):
        self._mesh = None
        self._ft = None
        self._bcu = None
        self._bcp = None
        self.t = 0.0
        self.res = float(res)
        super().__init__(solver_name, "dfg_2d_1", rho, mu, dt, T, list(f))

        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh, self._ft = self.generate_mesh()
        return self._mesh

    def generate_mesh(self):
        gmsh.initialize()
        L, H = 2.2, 0.41
        c_x, c_y, r = 0.2, 0.2, 0.05
        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0

        if mesh_comm.rank == model_rank:
            rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
            obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
            gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
            gmsh.model.occ.synchronize()

        if mesh_comm.rank == model_rank:
            volumes = gmsh.model.getEntities(dim=gdim)
            gmsh.model.addPhysicalGroup(
                volumes[0][0], [volumes[0][1]], self.fluid_marker
            )
            gmsh.model.setPhysicalName(volumes[0][0], self.fluid_marker, "Fluid")

            boundaries = gmsh.model.getBoundary(volumes, oriented=False)
            inflow, outflow, walls, obstacle_facets = [], [], [], []
            for boundary in boundaries:
                mass_center = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
                if np.allclose(mass_center[0], 0, atol=1e-5):
                    inflow.append(boundary[1])
                elif np.allclose(mass_center[0], L, atol=1e-5):
                    outflow.append(boundary[1])
                elif np.allclose(mass_center[1], 0, atol=1e-5) or np.allclose(
                    mass_center[1], H, atol=1e-5
                ):
                    walls.append(boundary[1])
                else:
                    obstacle_facets.append(boundary[1])

            gmsh.model.addPhysicalGroup(1, walls, self.wall_marker)
            gmsh.model.addPhysicalGroup(1, inflow, self.inlet_marker)
            gmsh.model.addPhysicalGroup(1, outflow, self.outlet_marker)
            gmsh.model.addPhysicalGroup(1, obstacle_facets, self.obstacle_marker)

            # Refinement near obstacle
            res_min = self.res
            res_max = res_min * 3
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "EdgesList", obstacle_facets)
            thresh_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_field, "IField", dist_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "LcMin", res_min)
            gmsh.model.mesh.field.setNumber(thresh_field, "LcMax", res_max)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", r)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 2 * H)
            gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.model.mesh.generate(gdim)

        mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        gmsh.finalize()
        mesh.name = "Grid"
        ft.name = "Facet markers"
        return mesh, ft

    @property
    def bcu(self):
        if not self._bcu:
            fdim = self.mesh.topology.dim - 1

            # Inlet: Strong Dirichlet (Static)
            self.u_inlet_func = Function(self.solver.V)
            self.u_inlet_func.interpolate(self.inlet_velocity_expression)
            entities_in = self._ft.find(self.inlet_marker)
            bcu_in = BoundaryCondition(self.u_inlet_func)
            bcu_in.initTopological(fdim, entities_in)

            # Walls and Obstacle: No-slip
            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, self._ft.find(self.wall_marker))

            bcu_obs = BoundaryCondition(u_nonslip)
            bcu_obs.initTopological(fdim, self._ft.find(self.obstacle_marker))

            self._bcu = [bcu_in, bcu_walls, bcu_obs]
        return self._bcu

    @property
    def bcp(self):
        return []

    def inlet_velocity_expression(self, x):
        H = 0.41
        U_max = 0.3
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4.0 * U_max * x[1] * (H - x[1]) / (H**2)
        return values

    def initial_velocity(self, x):
        return np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
