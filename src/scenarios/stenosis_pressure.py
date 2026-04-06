"""Stenosis scenario with pressure-driven inlet (no velocity BC).

Uses stabilized_schur_pressure_backflow solver with:
- Inlet: weak pressure BC (p_inlet) + Nitsche tangential condition (u_T = 0)
- Outlet: resistance BC (p = R*Q) + backflow stabilization

This corresponds to a physiological pressure-driven flow where the inlet
pressure is prescribed instead of velocity.
"""

import os

import gmsh
import numpy as np
from dolfinx.fem import Function
from dolfinx.io import gmshio
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc

from src.boundaryCondition import BoundaryCondition
from src.scenario import Scenario
from src.scenarios.stenosis import _MMHG

# For 2D formulation, pressure is divided by 2
_MMHG_2D = _MMHG * 0.5


class StenosisPressureSimulation(Scenario):
    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4

    stenosis_grades = {
        "mild": {"severity": 0.25, "slope": 0.3},
        "moderate": {"severity": 0.50, "slope": 0.3},
        "severe": {"severity": 0.75, "slope": 0.3},
    }

    def __init__(
        self,
        solver_name,
        dt,
        T,
        f: tuple[float, float] = (0, 0),
        grade="severe",
        p_inlet: float = 80.0,
        R_resistance: float = None,
        v_max: float = None,
        *,
        rho: float = 1.060e-3,
        mu: float = 3.5e-3,
        **kwargs,
    ):
        self._mesh = None
        self._ft = None

        p_grade = kwargs.pop("p_grade", 1)
        beta_nitsche = kwargs.pop("beta_nitsche", 100.0)
        beta_backflow = kwargs.pop("beta_backflow", 0.2)
        alpha_damping = kwargs.pop("alpha_damping", 0.75)
        self.mesh_options = kwargs.copy()

        defaults = {
            "L": 138.0,
            "R_in": 1.57,
            "R_out": 1.2,
            "res": 0.15,
            "x_position_stenosis": 30.0,
            "severity": 0.567,
            "slope": 0.4,
            "tension": 0.5,
        }
        for k, v in defaults.items():
            if k not in self.mesh_options:
                self.mesh_options[k] = v

        stenosis_grades = {
            "mild": {"severity": 0.25, "slope": 0.3},
            "moderate": {"severity": 0.50, "slope": 0.3},
            "severe": {"severity": 0.75, "slope": 0.3},
        }
        grade_params = stenosis_grades.get(grade, stenosis_grades["severe"])
        for k, v in grade_params.items():
            if k not in self.mesh_options:
                self.mesh_options[k] = v

        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None
        self._v_max = v_max

        if R_resistance is None:
            raise ValueError(
                "R_resistance is required for pressure-driven inlet. "
                "Pass it via CLI: --R_resistance <value>"
            )

        solver_kwargs = {
            "p_inlet": float(p_inlet) * _MMHG_2D,
            "p_grade": p_grade,
            "beta_nitsche": beta_nitsche,
            "beta_backflow": beta_backflow,
            "R_resistance": float(R_resistance),
            "alpha_damping": alpha_damping,
        }

        super().__init__(
            solver_name,
            "stenosis_pressure",
            rho,
            mu,
            dt,
            T,
            list(f),
            **solver_kwargs,
        )

        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )
        self.setup()

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh, self._ft = self._generate_mesh(**self.mesh_options)
        return self._mesh

    @property
    def bcu(self):
        """Wall no-slip only. Inlet velocity is NOT prescribed (weak pressure BC)."""
        if not self._bcu:
            fdim = self.mesh.topology.dim - 1
            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0
            entities_walls = self._ft.find(self.wall_marker)
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, entities_walls)
            self._bcu = [bcu_walls]
        return self._bcu

    @property
    def bcp(self):
        if not self._bcp:
            self._bcp = []
        return self._bcp

    def initial_velocity(self, x):
        """Initial velocity profile respecting stenosis geometry.

        If v_max is provided, uses a parabolic profile that conserves flow rate
        and accounts for the stenosis. Otherwise returns zero velocity.
        """
        if self._v_max is None:
            return np.zeros(
                (self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType
            )

        R_in = self.mesh_options["R_in"]
        R_out = self.mesh_options["R_out"]
        L = self.mesh_options["L"]
        x_sten = self.mesh_options["x_position_stenosis"]
        severity = self.mesh_options["severity"]
        slope = self.mesh_options["slope"]
        center_y = R_in
        v_max = float(self._v_max)

        R_taper = R_in + (R_out - R_in) * (x[0] / L)

        r_taper_mid = R_in + (R_out - R_in) * (x_sten / L)
        h_sten = severity * r_taper_mid
        dist_x = h_sten / slope if slope > 0 else L / 4
        dist_x = max(dist_x, L * 0.05)
        dist_x = min(dist_x, min(x_sten, L - x_sten) * 0.95)

        dx_abs = np.abs(x[0] - x_sten)
        in_stenosis = dx_abs < dist_x
        bump = np.where(
            in_stenosis,
            h_sten * 0.5 * (1.0 + np.cos(np.pi * dx_abs / dist_x)),
            0.0,
        )

        R_local = np.maximum(R_taper - bump, 1e-6)
        v_max_local = v_max * R_in / R_local

        r = x[1] - center_y
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = np.maximum(v_max_local * (1.0 - (r / R_local) ** 2), 0.0)
        return values

    def _generate_mesh(self, **kwargs):
        """Generate stenosis mesh."""
        L = kwargs["L"]
        R_in = kwargs["R_in"]
        R_out = kwargs["R_out"]
        res = kwargs["res"]
        x_sten = kwargs["x_position_stenosis"]
        severity = kwargs["severity"]
        slope = kwargs["slope"]
        tension = kwargs["tension"]

        gmsh.initialize()

        r_taper_mid = R_in + (R_out - R_in) * (x_sten / L)
        R_min = (1.0 - severity) * r_taper_mid

        if R_min <= 0:
            raise ValueError("severity too large: stenosis would close the channel")

        h_sten = r_taper_mid - R_min

        dist_x = h_sten / slope if slope > 0 else L / 4
        dist_x = min(dist_x, min(x_sten, L - x_sten) * 0.95)

        cp1_x = x_sten - dist_x
        cp2_x = x_sten + dist_x

        cp1_r = R_in + (R_out - R_in) * (cp1_x / L)
        cp2_r = R_in + (R_out - R_in) * (cp2_x / L)

        slope_top = (R_out - R_in) / L
        slope_bot = (R_in - R_out) / L

        ha = tension * dist_x
        hb = tension * dist_x

        y_top_0 = 2.0 * R_in
        y_bot_0 = 0.0
        y_top_cp1 = R_in + cp1_r
        y_bot_cp1 = R_in - cp1_r
        y_top_mid = R_in + R_min
        y_bot_mid = R_in - R_min
        y_top_cp2 = R_in + cp2_r
        y_bot_cp2 = R_in - cp2_r
        y_top_L = R_in + R_out
        y_bot_L = R_in - R_out

        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0

        if mesh_comm.rank == model_rank:
            p_bl = gmsh.model.occ.addPoint(0, y_bot_0, 0)
            p_tl = gmsh.model.occ.addPoint(0, y_top_0, 0)
            p_tr = gmsh.model.occ.addPoint(L, y_top_L, 0)
            p_br = gmsh.model.occ.addPoint(L, y_bot_L, 0)

            p_top_cp1 = gmsh.model.occ.addPoint(cp1_x, y_top_cp1, 0)
            p_top_mid = gmsh.model.occ.addPoint(x_sten, y_top_mid, 0)
            p_top_cp2 = gmsh.model.occ.addPoint(cp2_x, y_top_cp2, 0)
            p_bot_cp1 = gmsh.model.occ.addPoint(cp1_x, y_bot_cp1, 0)
            p_bot_mid = gmsh.model.occ.addPoint(x_sten, y_bot_mid, 0)
            p_bot_cp2 = gmsh.model.occ.addPoint(cp2_x, y_bot_cp2, 0)

            pt_l1 = gmsh.model.occ.addPoint(cp1_x + ha, y_top_cp1 + ha * slope_top, 0)
            pt_l2 = gmsh.model.occ.addPoint(x_sten - hb, y_top_mid - hb * slope_top, 0)
            pt_r1 = gmsh.model.occ.addPoint(x_sten + hb, y_top_mid + hb * slope_top, 0)
            pt_r2 = gmsh.model.occ.addPoint(cp2_x - ha, y_top_cp2 - ha * slope_top, 0)

            pb_r1 = gmsh.model.occ.addPoint(cp2_x - ha, y_bot_cp2 - ha * slope_bot, 0)
            pb_r2 = gmsh.model.occ.addPoint(x_sten + hb, y_bot_mid + hb * slope_bot, 0)
            pb_l1 = gmsh.model.occ.addPoint(x_sten - hb, y_bot_mid - hb * slope_bot, 0)
            pb_l2 = gmsh.model.occ.addPoint(cp1_x + ha, y_bot_cp1 + ha * slope_bot, 0)

            l_inlet = gmsh.model.occ.addLine(p_bl, p_tl)
            l_top_pre = gmsh.model.occ.addLine(p_tl, p_top_cp1)
            l_top_post = gmsh.model.occ.addLine(p_top_cp2, p_tr)
            l_outlet = gmsh.model.occ.addLine(p_tr, p_br)
            l_bot_post = gmsh.model.occ.addLine(p_br, p_bot_cp2)
            l_bot_pre = gmsh.model.occ.addLine(p_bot_cp1, p_bl)

            bez_top1 = gmsh.model.occ.addBezier([p_top_cp1, pt_l1, pt_l2, p_top_mid])
            bez_top2 = gmsh.model.occ.addBezier([p_top_mid, pt_r1, pt_r2, p_top_cp2])
            bez_bot2 = gmsh.model.occ.addBezier([p_bot_cp2, pb_r1, pb_r2, p_bot_mid])
            bez_bot1 = gmsh.model.occ.addBezier([p_bot_mid, pb_l1, pb_l2, p_bot_cp1])

            loop = gmsh.model.occ.addCurveLoop(
                [
                    l_inlet,
                    l_top_pre,
                    bez_top1,
                    bez_top2,
                    l_top_post,
                    l_outlet,
                    l_bot_post,
                    bez_bot2,
                    bez_bot1,
                    l_bot_pre,
                ]
            )
            gmsh.model.occ.addPlaneSurface([loop])
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
                com = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
                if np.isclose(com[0], 0.0, atol=res):
                    inflow.append(boundary[1])
                elif np.isclose(com[0], L, atol=res):
                    outflow.append(boundary[1])
                else:
                    walls.append(boundary[1])

            gmsh.model.addPhysicalGroup(1, walls, self.wall_marker)
            gmsh.model.setPhysicalName(1, self.wall_marker, "Walls")
            gmsh.model.addPhysicalGroup(1, inflow, self.inlet_marker)
            gmsh.model.setPhysicalName(1, self.inlet_marker, "Inlet")
            gmsh.model.addPhysicalGroup(1, outflow, self.outlet_marker)
            gmsh.model.setPhysicalName(1, self.outlet_marker, "Outlet")

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

    def solve(self, output_folder, afterStepCallback=None):
        """Run simulation and compute FFR."""
        result = super().solve(output_folder, afterStepCallback)
        self._compute_ffr(output_folder)
        return result

    def _compute_ffr(self, output_folder):
        """Compute FFR = p_distal / p_proximal at the channel centerline."""
        from dolfinx.geometry import (
            bb_tree,
            compute_collisions_points,
            compute_colliding_cells,
        )

        mesh = self.mesh
        R_in = self.mesh_options["R_in"]
        L = self.mesh_options["L"]
        center_y = R_in

        points = np.array(
            [
                [0.0, center_y, 0.0],
                [L, center_y, 0.0],
            ]
        )

        tree = bb_tree(mesh, mesh.topology.dim)
        cell_candidates = compute_collisions_points(tree, points)
        colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)

        p_values = np.full(len(points), np.nan)
        for i, point in enumerate(points):
            cells = colliding_cells.links(i)
            if len(cells) > 0:
                p_values[i] = self.solver.p_sol.eval(point, cells[0])[0]

        comm = mesh.comm
        all_p = comm.allreduce(np.where(np.isnan(p_values), 0.0, p_values), op=MPI.SUM)

        if comm.rank == 0:
            p_proximal = all_p[0]
            p_distal = all_p[1]
            ffr = p_distal / p_proximal if abs(p_proximal) > 1e-12 else float("nan")

            lines = [
                f"p_proximal (inlet center):  {p_proximal:.6f}",
                f"p_distal   (outlet center): {p_distal:.6f}",
                f"FFR = p_distal / p_proximal: {ffr:.6f}",
            ]
            txt = "\n".join(lines)
            print(f"\n[FFR] {txt}", flush=True)
            with open(os.path.join(output_folder, "ffr.txt"), "w") as f:
                f.write(txt + "\n")
