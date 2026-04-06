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

# Unit system: distances in mm, mass in g, time in s
#   1 Pa   = 1 g/(mm·s²)
#   1 Pa·s = 1 g/(mm·s)
#   1 mmHg = 133.322 g/(mm·s²)
_MMHG = 133.322


class StenosisSimulation(Scenario):
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
        p_inlet: float = 75.0,  # mmHg
        p_outlet: float = 10.0,  # mmHg
        *,
        rho: float = 1.060e-3,  # g/mm³  (blood density)
        mu: float = 3.5e-3,  # g/(mm·s) = Pa·s  (blood viscosity)
        **kwargs,
    ):
        self._mesh: Mesh = None
        self._ft = None

        p_grade = kwargs.pop("p_grade", 1)
        beta_nitsche = kwargs.pop("beta_nitsche", 100.0)
        beta_backflow = kwargs.pop("beta_backflow", None)
        R_resistance = kwargs.pop("R_resistance", None)
        initial_ffr = kwargs.pop("initial_ffr", 0.8)
        v_max = kwargs.pop("v_max", None)
        self.mesh_options = kwargs.copy()

        # Defaults: 80 mm long, inlet radius 1.5 mm, outlet radius 0.65 mm,
        # stenosis at first quarter (x = 20 mm)
        defaults = {
            "L": 138.0,
            "R_in": 1.57,  # inlet radius (mm)
            "R_out": 1.2,  # outlet radius (mm)
            "res": 0.15,
            "x_position_stenosis": 30.0,
            "severity": 0.567,
            "slope": 0.4,
            "tension": 0.5,
        }
        for k, v in defaults.items():
            if k not in self.mesh_options:
                self.mesh_options[k] = v

        grade_params = self.stenosis_grades.get(grade, self.stenosis_grades["severe"])
        for k, v in grade_params.items():
            if k not in self.mesh_options:
                self.mesh_options[k] = v

        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None

        self._v_max = v_max

        solver_kwargs = {
            "p_inlet": float(p_inlet) * _MMHG,
            "p_grade": p_grade,
            "beta_nitsche": beta_nitsche,
        }
        # For backflow solvers: pass beta_backflow
        if beta_backflow is not None:
            solver_kwargs["beta_backflow"] = float(beta_backflow)
        # For vascularbc_cbc solver: pass v_max (mm/s)
        if v_max is not None:
            solver_kwargs["v_max"] = float(v_max)
        # For vascularbc solver: pass R_resistance and initial_ffr instead of p_outlet
        if R_resistance is not None:
            solver_kwargs["R_resistance"] = float(R_resistance)
            solver_kwargs["initial_ffr"] = initial_ffr
        else:
            solver_kwargs["p_outlet"] = float(p_outlet) * _MMHG

        super().__init__(
            solver_name,
            "stenosis",
            rho,
            mu,
            dt,
            T,
            f,
            **solver_kwargs,
        )

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
        """Wall no-slip. When v_max is set (CBC solver), also adds parabolic
        Dirichlet velocity at the inlet."""
        if not self._bcu:
            fdim = self.mesh.topology.dim - 1
            u_nonslip = Function(self.solver.V)
            u_nonslip.x.array[:] = 0
            entities_walls = self._ft.find(self.wall_marker)
            bcu_walls = BoundaryCondition(u_nonslip)
            bcu_walls.initTopological(fdim, entities_walls)
            self._bcu = [bcu_walls]

            if self._v_max is not None:
                R_in = self.mesh_options["R_in"]
                center_y = R_in
                v_max = float(self._v_max)

                def parabolic_inlet(x):
                    values = np.zeros(
                        (self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType
                    )
                    r = x[1] - center_y
                    values[0] = v_max * (1.0 - (r / R_in) ** 2)
                    return values

                u_inlet = Function(self.solver.V)
                u_inlet.interpolate(parabolic_inlet)
                entities_inlet = self._ft.find(self.inlet_marker)
                bcu_inlet = BoundaryCondition(u_inlet)
                bcu_inlet.initTopological(fdim, entities_inlet)
                self._bcu.append(bcu_inlet)
        return self._bcu

    def solve(self, output_folder, afterStepCallback=None):
        result = super().solve(output_folder, afterStepCallback)
        self._compute_ffr(output_folder)
        return result

    def _compute_ffr(self, output_folder):
        """Compute FFR = p_distal / p_proximal at the channel centerline (y = R_in)."""
        from dolfinx.geometry import (
            bb_tree,
            compute_collisions_points,
            compute_colliding_cells,
        )

        mesh = self.mesh
        R_in = self.mesh_options["R_in"]
        L = self.mesh_options["L"]
        center_y = R_in

        # Proximal (inlet) and distal (outlet) points at channel center
        points = np.array(
            [
                [0.0, center_y, 0.0],  # proximal (inlet)
                [L, center_y, 0.0],  # distal (outlet)
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

        # Gather to rank 0 (only one rank owns each point)
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

    @property
    def bcp(self):
        """Empty: pressure BCs are applied naturally by the solver via p_inlet/p_outlet."""
        if not self._bcp:
            self._bcp = []
        return self._bcp

    def initial_velocity(self, x):
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

        # Linear taper radius
        R_taper = R_in + (R_out - R_in) * (x[0] / L)

        # Stenosis bump (cosine approximation of the Bezier profile)
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

        # Conserve 2D flow rate: v_max_local * R_local = v_max * R_in
        v_max_local = v_max * R_in / R_local

        r = x[1] - center_y
        values = np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = np.maximum(v_max_local * (1.0 - (r / R_local) ** 2), 0.0)
        return values

    def generate_mesh(self, **kwargs):
        gmsh.initialize()

        L = kwargs["L"]
        R_in = kwargs["R_in"]
        R_out = kwargs["R_out"]
        res = kwargs["res"]
        x_sten = kwargs["x_position_stenosis"]
        severity = kwargs["severity"]
        slope = kwargs["slope"]
        tension = kwargs["tension"]

        # Geometry follows the same logic as src/geom/stenosis/stenosis.py, but 2D:
        #   - channel tapers linearly from diameter 2*R_in (inlet) to 2*R_out (outlet)
        #   - at x_sten the radius is further reduced by severity relative to the
        #     local taper radius:  R_min = (1 - severity) * r_taper(x_sten)

        # Linear taper radius at the stenosis centre
        r_taper_mid = R_in + (R_out - R_in) * (x_sten / L)
        R_min = (1.0 - severity) * r_taper_mid

        if R_min <= 0:
            raise ValueError("severity too large: stenosis would close the channel")

        # Depth of stenosis above the taper line
        h_sten = r_taper_mid - R_min  # = severity * r_taper_mid

        # Half-width of the stenosis region (same formula as the 3D code)
        dist_x = h_sten / slope if slope > 0 else L / 4
        dist_x = min(dist_x, min(x_sten, L - x_sten) * 0.95)

        cp1_x = x_sten - dist_x
        cp2_x = x_sten + dist_x

        # Taper radius at the Bezier junction points
        cp1_r = R_in + (R_out - R_in) * (cp1_x / L)
        cp2_r = R_in + (R_out - R_in) * (cp2_x / L)

        # Overall taper slope (dy_top/dx < 0, dy_bot/dx > 0 for a narrowing channel)
        slope_top = (R_out - R_in) / L  # negative
        slope_bot = (R_in - R_out) / L  # positive

        # Bezier handle length along x
        ha = tension * dist_x
        hb = tension * dist_x

        # Key y-coordinates.
        # Channel centre axis sits at y = R_in (fixed).
        # Top wall:    y_top(x) = R_in + R_profile(x)
        # Bottom wall: y_bot(x) = R_in - R_profile(x)
        y_top_0 = 2.0 * R_in  # top-left  corner (inlet)
        y_bot_0 = 0.0  # bottom-left corner (inlet)
        y_top_cp1 = R_in + cp1_r
        y_bot_cp1 = R_in - cp1_r
        y_top_mid = R_in + R_min
        y_bot_mid = R_in - R_min
        y_top_cp2 = R_in + cp2_r
        y_bot_cp2 = R_in - cp2_r
        y_top_L = R_in + R_out  # top-right  corner (outlet)
        y_bot_L = R_in - R_out  # bottom-right corner (outlet)

        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0

        if mesh_comm.rank == model_rank:
            # Corner points
            p_bl = gmsh.model.occ.addPoint(0, y_bot_0, 0)
            p_tl = gmsh.model.occ.addPoint(0, y_top_0, 0)
            p_tr = gmsh.model.occ.addPoint(L, y_top_L, 0)
            p_br = gmsh.model.occ.addPoint(L, y_bot_L, 0)

            # Stenosis junction points (where Bezier meets the straight taper lines)
            p_top_cp1 = gmsh.model.occ.addPoint(cp1_x, y_top_cp1, 0)
            p_top_mid = gmsh.model.occ.addPoint(x_sten, y_top_mid, 0)
            p_top_cp2 = gmsh.model.occ.addPoint(cp2_x, y_top_cp2, 0)
            p_bot_cp1 = gmsh.model.occ.addPoint(cp1_x, y_bot_cp1, 0)
            p_bot_mid = gmsh.model.occ.addPoint(x_sten, y_bot_mid, 0)
            p_bot_cp2 = gmsh.model.occ.addPoint(cp2_x, y_bot_cp2, 0)

            # Bezier control handles.
            # Tangent at the junction points follows the taper slope.
            # Tangent at the stenosis peak also follows the taper slope → C1 continuity.
            #
            # Top wall (traversed left→right in the loop): cp1 → mid → cp2
            pt_l1 = gmsh.model.occ.addPoint(cp1_x + ha, y_top_cp1 + ha * slope_top, 0)
            pt_l2 = gmsh.model.occ.addPoint(x_sten - hb, y_top_mid - hb * slope_top, 0)
            pt_r1 = gmsh.model.occ.addPoint(x_sten + hb, y_top_mid + hb * slope_top, 0)
            pt_r2 = gmsh.model.occ.addPoint(cp2_x - ha, y_top_cp2 - ha * slope_top, 0)

            # Bottom wall (traversed right→left in the loop): cp2 → mid → cp1
            pb_r1 = gmsh.model.occ.addPoint(cp2_x - ha, y_bot_cp2 - ha * slope_bot, 0)
            pb_r2 = gmsh.model.occ.addPoint(x_sten + hb, y_bot_mid + hb * slope_bot, 0)
            pb_l1 = gmsh.model.occ.addPoint(x_sten - hb, y_bot_mid - hb * slope_bot, 0)
            pb_l2 = gmsh.model.occ.addPoint(cp1_x + ha, y_bot_cp1 + ha * slope_bot, 0)

            # Straight boundary segments
            l_inlet = gmsh.model.occ.addLine(p_bl, p_tl)
            l_top_pre = gmsh.model.occ.addLine(p_tl, p_top_cp1)
            l_top_post = gmsh.model.occ.addLine(p_top_cp2, p_tr)
            l_outlet = gmsh.model.occ.addLine(p_tr, p_br)
            l_bot_post = gmsh.model.occ.addLine(p_br, p_bot_cp2)
            l_bot_pre = gmsh.model.occ.addLine(p_bot_cp1, p_bl)

            # Cubic Beziers — top wall (left→right), bottom wall (right→left)
            bez_top1 = gmsh.model.occ.addBezier([p_top_cp1, pt_l1, pt_l2, p_top_mid])
            bez_top2 = gmsh.model.occ.addBezier([p_top_mid, pt_r1, pt_r2, p_top_cp2])
            bez_bot2 = gmsh.model.occ.addBezier([p_bot_cp2, pb_r1, pb_r2, p_bot_mid])
            bez_bot1 = gmsh.model.occ.addBezier([p_bot_mid, pb_l1, pb_l2, p_bot_cp1])

            # Counterclockwise loop:
            # inlet (up) → top wall (right) → outlet (down) → bottom wall (left)
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

            # Inlet centre: (0, R_in);  outlet centre: (L, R_in)
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
