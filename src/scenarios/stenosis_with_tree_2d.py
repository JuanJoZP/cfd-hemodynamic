"""Stenosis + 2D vascular tree scenario.

Uses the pure-Python VascularTree2D generator (Murray's law, flow
conservation) — no external VascuSynth binary required.

The tree attaches at the stenosis outlet via a trapezoidal coupling
region that transitions from the artery diameter down to the tree
root channel width.
"""

import gmsh
import numpy as np
from dolfinx.fem import Constant, Function, dirichletbc, locate_dofs_topological
from dolfinx.io import gmshio
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc

from src.boundaryCondition import BoundaryCondition
from src.geom.tree.tree_2d import VascularTree2D
from src.scenario import Scenario
from src.solvers_aux.stokes import StokesSolver

_MMHG = 133.322


class StenosisWithTree2DSimulation(Scenario):
    """2-D stenosis with a procedurally-generated vascular tree at the outlet.

    Boundary markers
    ----------------
    fluid_marker  = 1
    inlet_marker  = 2
    outlet_marker = 3  (terminal ends of the tree)
    wall_marker   = 4
    """

    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4

    def __init__(
        self,
        solver_name,
        dt,
        T,
        f: tuple[float, float] = (0, 0),
        grade="severe",
        p_inlet: float = 75.0,
        p_outlet: float = 10.0,
        *,
        rho: float = 1.060e-3,
        mu: float = 3.5e-3,
        # Tree parameters
        n_generations: int = 3,
        tree_gamma: float = 3.0,
        tree_angle: float = 35.0,
        tree_length_ratio: float = 8.0,
        tree_asymmetry: float = 0.5,
        coupling_slope: float = 0.3,
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

        # Stenosis mesh defaults (same as stenosis.py)
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
            if k not in kwargs:
                self.mesh_options[k] = v

        # Tree config
        self.tree_config = {
            "n_generations": int(n_generations),
            "gamma": float(tree_gamma),
            "bifurcation_angle": float(tree_angle),
            "length_ratio": float(tree_length_ratio),
            "asymmetry": float(tree_asymmetry),
            "coupling_slope": float(coupling_slope),
        }

        self._bcu: list[BoundaryCondition] = None
        self._bcp: list[BoundaryCondition] = None

        self._v_max = v_max

        solver_kwargs = {
            "p_inlet": float(p_inlet) * _MMHG,
            "p_grade": p_grade,
            "beta_nitsche": beta_nitsche,
        }
        if beta_backflow is not None:
            solver_kwargs["beta_backflow"] = float(beta_backflow)
        if v_max is not None:
            solver_kwargs["v_max"] = float(v_max)
        if R_resistance is not None:
            # Subtract the Poiseuille resistance of the included tree
            # so the solver prescribes only the remaining downstream resistance.
            R_included = self._compute_tree_resistance(
                r_root=self.mesh_options["R_out"],
                n_gen=self.tree_config["n_generations"],
                length_ratio=self.tree_config["length_ratio"],
                mu=mu,
            )
            R_effective = float(R_resistance) - R_included
            if R_effective < 0:
                R_effective = 0.0
                print(
                    f"[WARN] R_resistance ({R_resistance:.4e}) < R_included "
                    f"({R_included:.4e}); clamping outlet resistance to 0"
                )
            else:
                print(
                    f"[INFO] R_resistance={R_resistance:.4e}, "
                    f"R_included(tree Poiseuille)={R_included:.4e}, "
                    f"R_effective(outlets)={R_effective:.4e}"
                )
            solver_kwargs["R_resistance"] = R_effective
            solver_kwargs["initial_ffr"] = initial_ffr
        else:
            solver_kwargs["p_outlet"] = float(p_outlet) * _MMHG

        super().__init__(
            solver_name,
            "stenosis_with_tree_2d",
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
        self._solve_stokes_initial()

    @staticmethod
    def _compute_tree_resistance(r_root, n_gen, length_ratio, mu):
        """Equivalent Poiseuille resistance of a symmetric 2D tree (seen from root).

        For a symmetric Murray-law tree, the pressure drop from root to any
        terminal is the same.  Along one root-to-terminal path the flow at
        generation j is Q_total/2^j and the segment radius is r_root/2^(j/3).

        R_included = ΔP_tree / Q_total
                   = (3·μ·α / (2·r₀²)) · Σ_{j=0}^{n-1} 2^{-j/3}

        where α = length_ratio, r₀ = r_root.
        """
        alpha = float(length_ratio)
        r0 = float(r_root)
        coeff = 3.0 * float(mu) * alpha / (2.0 * r0**2)
        s = sum(2.0 ** (-j / 3.0) for j in range(int(n_gen)))
        return coeff * s

    def _solve_stokes_initial(self):
        """Solve Stokes on the full domain and use the result as initial condition."""
        mesh = self.mesh
        fdim = mesh.topology.dim - 1

        rho_const = Constant(mesh, PETSc.ScalarType(self.solver.rho.value))
        mu_const = Constant(mesh, PETSc.ScalarType(self.solver.mu.value))
        f_const = Constant(mesh, PETSc.ScalarType((0.0,) * mesh.geometry.dim))

        stokes = StokesSolver(mesh, rho_const, mu_const, f_const)

        # ── Build Dirichlet BCs on the Stokes velocity space ──────────────
        bcs = []

        # Wall no-slip
        u_noslip = Function(stokes.V)
        u_noslip.x.array[:] = 0.0
        wall_facets = self._ft.find(self.wall_marker)
        wall_dofs = locate_dofs_topological(stokes.V, fdim, wall_facets)
        bcs.append(dirichletbc(u_noslip, wall_dofs))

        # Inlet: parabolic velocity if v_max is set
        if self._v_max is not None:
            R_in = self.mesh_options["R_in"]
            center_y = R_in
            v_max = float(self._v_max)

            def parabolic_inlet(x):
                values = np.zeros(
                    (mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType
                )
                r = x[1] - center_y
                values[0] = v_max * (1.0 - (r / R_in) ** 2)
                return values

            u_inlet = Function(stokes.V)
            u_inlet.interpolate(parabolic_inlet)
            inlet_facets = self._ft.find(self.inlet_marker)
            inlet_dofs = locate_dofs_topological(stokes.V, fdim, inlet_facets)
            bcs.append(dirichletbc(u_inlet, inlet_dofs))

        # ── Solve Stokes ──────────────────────────────────────────────────
        stokes.solve(bcs=bcs)

        if mesh.comm.rank == 0:
            print("[INFO] Stokes initial condition solved")

        # ── Copy Stokes velocity (P3) → NS velocity space (P1) ───────────
        u_init = Function(self.solver.V)
        u_init.interpolate(stokes.u_sol)

        self.solver.u_sol.x.array[:] = u_init.x.array
        self.solver.u_prev.x.array[:] = u_init.x.array

        # Replace initial_velocity so that solve() line
        # `u_sol.interpolate(self.initial_velocity)` uses the Stokes Function
        self.initial_velocity = u_init

    # ── lazy properties ───────────────────────────────────────────────────

    @property
    def mesh(self):
        if not self._mesh:
            self._mesh, self._ft = self._generate_mesh()
        return self._mesh

    @property
    def bcu(self):
        """Wall no-slip.  When v_max is set, also adds parabolic Dirichlet
        velocity at the inlet."""
        if not self._bcu:
            fdim = self.mesh.topology.dim - 1
            u_noslip = Function(self.solver.V)
            u_noslip.x.array[:] = 0
            bcu_walls = BoundaryCondition(u_noslip)
            bcu_walls.initTopological(fdim, self._ft.find(self.wall_marker))
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

    @property
    def bcp(self):
        if not self._bcp:
            self._bcp = []
        return self._bcp

    def initial_velocity(self, x):
        return np.zeros((self.mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)

    # ── mesh generation ───────────────────────────────────────────────────

    def _generate_mesh(self):
        L = self.mesh_options["L"]
        R_in = self.mesh_options["R_in"]
        R_out = self.mesh_options["R_out"]
        res = self.mesh_options["res"]
        x_sten = self.mesh_options["x_position_stenosis"]
        severity = self.mesh_options["severity"]
        slope = self.mesh_options["slope"]
        tension = self.mesh_options["tension"]

        coupling_slope = self.tree_config["coupling_slope"]

        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0

        # ── 1. Generate vascular tree (rank 0) ─────────────────────────────
        tree = None
        r_root = 0.0
        coupling_length = 0.0
        terminal_positions = {}

        if mesh_comm.rank == model_rank:
            # Tree root radius = outlet radius of the stenosis
            r_root = R_out

            print(
                f"[DEBUG] Tree params: length_ratio={self.tree_config['length_ratio']}, "
                f"r_root={r_root}, n_gen={self.tree_config['n_generations']}, "
                f"angle={self.tree_config['bifurcation_angle']}, "
                f"asymmetry={self.tree_config['asymmetry']}"
            )

            tree = VascularTree2D(
                r_root=r_root,
                n_generations=self.tree_config["n_generations"],
                gamma=self.tree_config["gamma"],
                bifurcation_angle=self.tree_config["bifurcation_angle"],
                length_ratio=self.tree_config["length_ratio"],
                asymmetry=self.tree_config["asymmetry"],
            )

            # Coupling region: trapezoid from R_out down to r_root
            # (here R_out == r_root so coupling narrows from R_in taper end)
            coupling_length = max((R_out - r_root) / coupling_slope, R_out * 2.0)

            # Tree origin = end of coupling region, at channel centre
            tree_origin = (L + coupling_length, R_in)
            tree.generate(origin=tree_origin, direction=0.0)

            terminal_positions = {nid: tree.nodes[nid] for nid in tree.terminals}

            # Map terminal node -> radius of the edge arriving at it
            terminal_radii = {}
            for e in tree.edges:
                if e["to"] in tree.terminals:
                    terminal_radii[e["to"]] = e["radius"]

        # ── 2. Build gmsh geometry ─────────────────────────────────────────
        gmsh.initialize()

        if mesh_comm.rank == model_rank:
            # ── 2a. Stenosis (same as stenosis.py) ─────────────────────────
            r_taper_mid = R_in + (R_out - R_in) * (x_sten / L)
            R_min = (1.0 - severity) * r_taper_mid
            if R_min <= 0:
                raise ValueError("severity too large: stenosis would close")

            h_sten = r_taper_mid - R_min
            dist_x = h_sten / slope if slope > 0 else L / 4
            dist_x = max(dist_x, L * 0.05)
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

            # Corner points
            p_bl = gmsh.model.occ.addPoint(0, y_bot_0, 0)
            p_tl = gmsh.model.occ.addPoint(0, y_top_0, 0)
            p_tr = gmsh.model.occ.addPoint(L, y_top_L, 0)
            p_br = gmsh.model.occ.addPoint(L, y_bot_L, 0)

            # Stenosis junction points
            p_top_cp1 = gmsh.model.occ.addPoint(cp1_x, y_top_cp1, 0)
            p_top_mid = gmsh.model.occ.addPoint(x_sten, y_top_mid, 0)
            p_top_cp2 = gmsh.model.occ.addPoint(cp2_x, y_top_cp2, 0)
            p_bot_cp1 = gmsh.model.occ.addPoint(cp1_x, y_bot_cp1, 0)
            p_bot_mid = gmsh.model.occ.addPoint(x_sten, y_bot_mid, 0)
            p_bot_cp2 = gmsh.model.occ.addPoint(cp2_x, y_bot_cp2, 0)

            # Bezier handles
            pt_l1 = gmsh.model.occ.addPoint(cp1_x + ha, y_top_cp1 + ha * slope_top, 0)
            pt_l2 = gmsh.model.occ.addPoint(x_sten - hb, y_top_mid - hb * slope_top, 0)
            pt_r1 = gmsh.model.occ.addPoint(x_sten + hb, y_top_mid + hb * slope_top, 0)
            pt_r2 = gmsh.model.occ.addPoint(cp2_x - ha, y_top_cp2 - ha * slope_top, 0)

            pb_r1 = gmsh.model.occ.addPoint(cp2_x - ha, y_bot_cp2 - ha * slope_bot, 0)
            pb_r2 = gmsh.model.occ.addPoint(x_sten + hb, y_bot_mid + hb * slope_bot, 0)
            pb_l1 = gmsh.model.occ.addPoint(x_sten - hb, y_bot_mid - hb * slope_bot, 0)
            pb_l2 = gmsh.model.occ.addPoint(cp1_x + ha, y_bot_cp1 + ha * slope_bot, 0)

            # Lines
            l_inlet = gmsh.model.occ.addLine(p_bl, p_tl)
            l_top_pre = gmsh.model.occ.addLine(p_tl, p_top_cp1)
            l_top_post = gmsh.model.occ.addLine(p_top_cp2, p_tr)
            l_outlet = gmsh.model.occ.addLine(p_tr, p_br)
            l_bot_post = gmsh.model.occ.addLine(p_br, p_bot_cp2)
            l_bot_pre = gmsh.model.occ.addLine(p_bot_cp1, p_bl)

            # Beziers
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
            steno_surf = gmsh.model.occ.addPlaneSurface([loop])

            # ── 2b. Coupling trapezoid ─────────────────────────────────────
            p_cp_bl = gmsh.model.occ.addPoint(L, y_bot_L, 0)
            p_cp_tl = gmsh.model.occ.addPoint(L, y_top_L, 0)
            p_cp_tr = gmsh.model.occ.addPoint(L + coupling_length, R_in + r_root, 0)
            p_cp_br = gmsh.model.occ.addPoint(L + coupling_length, R_in - r_root, 0)

            l_cp_left = gmsh.model.occ.addLine(p_cp_bl, p_cp_tl)
            l_cp_top = gmsh.model.occ.addLine(p_cp_tl, p_cp_tr)
            l_cp_right = gmsh.model.occ.addLine(p_cp_tr, p_cp_br)
            l_cp_bot = gmsh.model.occ.addLine(p_cp_br, p_cp_bl)

            cp_loop = gmsh.model.occ.addCurveLoop(
                [
                    l_cp_left,
                    l_cp_top,
                    l_cp_right,
                    l_cp_bot,
                ]
            )
            coupling_surf = gmsh.model.occ.addPlaneSurface([cp_loop])

            # ── 2c. Tree branch channels ───────────────────────────────────
            N_SAMPLES = 12
            HANDLE = 0.4

            # Parent direction for smooth Bezier channels
            adj_out = {}
            for e in tree.edges:
                adj_out.setdefault(e["from"], []).append(e["to"])

            parent_dir = {}
            # Root edge: first edge's from node -> direction +x
            root_from = tree.edges[0]["from"]
            parent_dir[root_from] = np.array([1.0, 0.0])
            queue = [root_from]
            visited = {root_from}
            while queue:
                curr = queue.pop(0)
                for child in adj_out.get(curr, []):
                    if child in visited:
                        continue
                    visited.add(child)
                    seg = tree.nodes[child] - tree.nodes[curr]
                    norm = np.linalg.norm(seg)
                    parent_dir[child] = (
                        seg / norm
                        if norm > 1e-10
                        else parent_dir.get(curr, np.array([1.0, 0.0]))
                    )
                    queue.append(child)

            def bezier_channel_polygon(A, B, tang_in, r_start, r_end):
                """Closed CCW polygon around the Bezier channel from A to B.

                The radius varies smoothly from r_start (at A) to r_end
                (at B) using a cosine blend, giving a C1-smooth taper at
                each bifurcation junction.
                """
                seg = B - A
                seg_len = np.linalg.norm(seg)
                if seg_len < 1e-10:
                    return []
                tang_out = seg / seg_len
                h = seg_len * HANDLE
                p0, p1 = A, A + h * tang_in
                p3, p2 = B, B - h * tang_out
                top, bot = [], []
                for i in range(N_SAMPLES + 1):
                    t = i / N_SAMPLES
                    mt = 1.0 - t
                    pt = (
                        mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3
                    )
                    tan = (
                        3 * mt**2 * (p1 - p0)
                        + 6 * mt * t * (p2 - p1)
                        + 3 * t**2 * (p3 - p2)
                    )
                    n_t = np.linalg.norm(tan)
                    if n_t < 1e-12:
                        continue
                    perp = np.array([-tan[1], tan[0]]) / n_t
                    # Smooth cosine taper: r(0)=r_start, r(1)=r_end
                    blend = 0.5 * (1.0 - np.cos(np.pi * t))
                    r_local = r_start + (r_end - r_start) * blend
                    top.append(pt + r_local * perp)
                    bot.append(pt - r_local * perp)
                return top + bot[::-1]

            tree_surf_dimtags = []
            for e in tree.edges:
                frm, to = e["from"], e["to"]
                r_child = e["radius"]
                r_par = e["r_parent"]
                if frm not in tree.nodes or to not in tree.nodes:
                    continue
                if r_child < res * 0.1:
                    continue
                poly = bezier_channel_polygon(
                    tree.nodes[frm],
                    tree.nodes[to],
                    parent_dir.get(frm, np.array([1.0, 0.0])),
                    r_start=r_par,
                    r_end=r_child,
                )
                if len(poly) < 3:
                    continue
                ptags = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in poly]
                ltags = [
                    gmsh.model.occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)])
                    for i in range(len(ptags))
                ]
                tloop = gmsh.model.occ.addCurveLoop(ltags)
                tsurf = gmsh.model.occ.addPlaneSurface([tloop])
                tree_surf_dimtags.append((2, tsurf))

            # ── 2d. Boolean fuse ───────────────────────────────────────────
            tools = [(2, coupling_surf)] + tree_surf_dimtags
            if tools:
                gmsh.model.occ.fuse(
                    [(2, steno_surf)],
                    tools,
                    removeObject=True,
                    removeTool=True,
                )

            # ── 2e. Fillet inner corners at bifurcations ──────────────────
            # Create small wedge surfaces at each crotch (where sibling
            # branches diverge) and fuse them with the main surface.
            # Each wedge is: crotch vertex → point A along wall 1 →
            # arc A-B → point B along wall 2 → back to crotch.
            FILLET_FRACTION = 1
            MIN_ANGLE_DEG = 10.0
            MAX_ANGLE_DEG = 160.0
            gmsh.model.occ.synchronize()

            bif_list = tree.bifurcation_points
            default_r_fillet = (
                FILLET_FRACTION * min(e["radius"] for e in tree.edges)
                if tree.edges
                else res
            )

            all_surfs = gmsh.model.occ.getEntities(2)

            # Get all vertices and their coordinates
            all_pts = gmsh.model.occ.getEntities(0)
            pt_coords = {}
            for _, ptag in all_pts:
                bb = gmsh.model.occ.getBoundingBox(0, ptag)
                pt_coords[ptag] = np.array([bb[0], bb[1]])

            # Build point -> curves map
            pt_to_curves = {}
            for _, ctag in gmsh.model.occ.getEntities(1):
                endpts = gmsh.model.getBoundary(
                    [(1, ctag)], combined=False, oriented=False
                )
                for _, ep in endpts:
                    pt_to_curves.setdefault(ep, []).append(ctag)

            def curve_tangent_at_point(ctag, ptag):
                par_bounds = gmsh.model.getParametrizationBounds(1, ctag)
                u_min, u_max = par_bounds[0][0], par_bounds[1][0]
                p0 = np.array(gmsh.model.getValue(1, ctag, [u_min])[:2])
                p1 = np.array(gmsh.model.getValue(1, ctag, [u_max])[:2])
                pcoord = pt_coords[ptag]
                du = (u_max - u_min) * 0.01
                if np.linalg.norm(p0 - pcoord) < np.linalg.norm(p1 - pcoord):
                    p_next = np.array(gmsh.model.getValue(1, ctag, [u_min + du])[:2])
                    tang = p_next - p0
                else:
                    p_next = np.array(gmsh.model.getValue(1, ctag, [u_max - du])[:2])
                    tang = p_next - p1
                n = np.linalg.norm(tang)
                return tang / n if n > 1e-8 else np.array([1.0, 0.0])

            # Detect crotch vertices
            x_tree_start = L
            wedge_surfs = []

            # Terminal positions to exclude from fillet
            term_positions = list(terminal_positions.values())

            for ptag, pxy in pt_coords.items():
                if pxy[0] < x_tree_start:
                    continue
                # Skip vertices near terminal outlets
                near_terminal = any(
                    np.linalg.norm(pxy - tp) < default_r_fillet * 3.0
                    for tp in term_positions
                )
                if near_terminal:
                    continue
                curves = pt_to_curves.get(ptag, [])
                if len(curves) != 2:
                    continue
                t0 = curve_tangent_at_point(curves[0], ptag)
                t1 = curve_tangent_at_point(curves[1], ptag)
                cos_a = np.clip(np.dot(t0, t1), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_a))
                if not (MIN_ANGLE_DEG < angle_deg < MAX_ANGLE_DEG):
                    continue

                # Compute fillet radius from nearest bifurcation
                r_fillet = default_r_fillet
                for bif in bif_list:
                    d = np.linalg.norm(pxy - bif["pos"])
                    if d < bif["r_min"] * 5.0:
                        r_fillet = FILLET_FRACTION * bif["r_min"]
                        break

                # Build wedge: crotch P, point A along t0, point B along t1,
                # arc from A to B curving away from P.
                # Push A slightly toward the opposite wall (t1 direction)
                # and B toward t0, so the wedge penetrates into both channels.
                INSET = 0.05
                P = pxy
                A = P + r_fillet * t0 - INSET * r_fillet * t1
                B = P + r_fillet * t1 - INSET * r_fillet * t0
                C = P + r_fillet * 0.3 * (t0 + t1)  # center opposite to P

                pP = gmsh.model.occ.addPoint(P[0], P[1], 0)
                pA = gmsh.model.occ.addPoint(A[0], A[1], 0)
                pB = gmsh.model.occ.addPoint(B[0], B[1], 0)
                pC = gmsh.model.occ.addPoint(C[0], C[1], 0)

                lPA = gmsh.model.occ.addLine(pP, pA)
                arc = gmsh.model.occ.addCircleArc(pA, pC, pB, center=False)
                lBP = gmsh.model.occ.addLine(pB, pP)

                try:
                    wloop = gmsh.model.occ.addCurveLoop([lPA, arc, lBP])
                    wsurf = gmsh.model.occ.addPlaneSurface([wloop])
                    wedge_surfs.append((2, wsurf))
                    print(
                        f"  -> wedge at ({P[0]:.2f}, {P[1]:.2f}), "
                        f"r={r_fillet:.4f}, angle={angle_deg:.1f}°"
                    )
                except Exception as exc:
                    print(f"  -> wedge FAILED at vertex {ptag}: {exc}")

            # Fuse wedges with main surface
            if wedge_surfs:
                gmsh.model.occ.fuse(
                    all_surfs,
                    wedge_surfs,
                    removeObject=True,
                    removeTool=True,
                )
                # Remove duplicate/overlapping edges left by the fuse
                gmsh.model.occ.removeAllDuplicates()
                gmsh.model.occ.healShapes()

            gmsh.model.occ.synchronize()

        # ── 3. Physical groups ─────────────────────────────────────────────
        inflow, outflow, walls = [], [], []
        if mesh_comm.rank == model_rank:
            all_surfs = gmsh.model.getEntities(dim=gdim)
            gmsh.model.addPhysicalGroup(
                gdim, [s[1] for s in all_surfs], self.fluid_marker
            )
            gmsh.model.setPhysicalName(gdim, self.fluid_marker, "Fluid")

            boundaries = gmsh.model.getBoundary(all_surfs, oriented=False)

            # Classify boundaries: a terminal cap has both endpoints
            # at distance ≈ r_terminal from the terminal node (it straddles
            # the node perpendicularly), while adjacent wall curves have one
            # endpoint near the terminal and the other far away.
            inlet_tol = res * 5.0

            for bnd in boundaries:
                ctag = bnd[1]
                com = gmsh.model.occ.getCenterOfMass(bnd[0], bnd[1])
                cxy = np.array([com[0], com[1]])

                if cxy[0] <= inlet_tol:
                    inflow.append(ctag)
                    continue

                # Get curve endpoints
                endpts = gmsh.model.getBoundary(
                    [(1, ctag)], combined=False, oriented=False
                )
                ep_coords = []
                for _, ptag in endpts:
                    bb = gmsh.model.occ.getBoundingBox(0, ptag)
                    ep_coords.append(np.array([bb[0], bb[1]]))

                is_outlet = False
                if len(ep_coords) == 2:
                    for nid, t_pos in terminal_positions.items():
                        r_term = terminal_radii.get(nid, res)
                        d0 = np.linalg.norm(ep_coords[0] - t_pos)
                        d1 = np.linalg.norm(ep_coords[1] - t_pos)
                        # Both endpoints near the terminal (within 1.5*r)
                        # and balanced (ratio close to 1 — cap straddles the node)
                        tol = r_term * 1.8
                        if d0 < tol and d1 < tol:
                            ratio = min(d0, d1) / max(d0, d1) if max(d0, d1) > 1e-12 else 1.0
                            if ratio > 0.3:
                                is_outlet = True
                                break

                (outflow if is_outlet else walls).append(ctag)

            if inflow:
                gmsh.model.addPhysicalGroup(1, inflow, self.inlet_marker)
                gmsh.model.setPhysicalName(1, self.inlet_marker, "Inlet")
            else:
                print("[WARN] No inlet curves detected")
            if outflow:
                gmsh.model.addPhysicalGroup(1, outflow, self.outlet_marker)
                gmsh.model.setPhysicalName(1, self.outlet_marker, "Outlets")
            else:
                print("[WARN] No outlet curves detected")
            if walls:
                gmsh.model.addPhysicalGroup(1, walls, self.wall_marker)
                gmsh.model.setPhysicalName(1, self.wall_marker, "Walls")

        # ── 4. Mesh ────────────────────────────────────────────────────────
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res * 1.5)

        if mesh_comm.rank == model_rank:
            gmsh.write("stenosis_tree_pre_mesh.brep")
            print("[DEBUG] wrote stenosis_tree_pre_mesh.brep")
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.model.mesh.generate(gdim)
            gmsh.model.mesh.setOrder(1)
            gmsh.model.mesh.optimize("Netgen")
            gmsh.write("stenosis_tree_tagged.msh")
            print("[DEBUG] wrote stenosis_tree_tagged.msh (with physical groups)")

        mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        mesh.name = "Grid"
        ft.name = "Facet markers"
        gmsh.finalize()

        return mesh, ft
