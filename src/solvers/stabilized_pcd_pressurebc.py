# Curl-curl (rotational) formulation with natural pressure BCs and Nitsche inlet tangential BC.
# SUPG/PSPG/LSIC stabilization, implicit Euler, SNES + PCD Schur preconditioner.

from typing import Callable

import dolfinx.fem as fem
import numpy as np
from dolfinx.fem import Constant, DirichletBC, Function, form, functionspace
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    assemble_vector_block,
    create_matrix_block,
    create_vector_block,
)
from dolfinx.mesh import Mesh
from fenicsx_pctools.mat import create_splittable_matrix_block
from petsc4py import PETSc
from petsc4py import typing as petsc_typing
from ufl import (
    FacetNormal,
    Measure,
    MixedFunctionSpace,
    TestFunctions,
    TrialFunctions,
    as_vector,
    conditional,
    cross,
    curl,
    derivative,
    div,
    dot,
    dx,
    extract_blocks,
    ge,
    grad,
    inner,
    le,
    nabla_grad,
    sqrt,
)

from src.boundaryCondition import BoundaryCondition
from src.solverBase import SolverBase


class Solver(SolverBase):
    MAX_ITER = 20

    def __init__(
        self,
        mesh: Mesh,
        dt: float,
        rho: float,
        mu: float,
        f: list,
        initial_velocity: Callable[[np.ndarray], np.ndarray] = None,
        p_inlet: float = None,
        p_outlet: float = None,
        beta_nitsche: float = 100.0,
        **kwargs,
    ):
        if p_inlet is None or p_outlet is None:
            raise ValueError(
                "p_inlet and p_outlet are required for stabilized_pcd_pressurebc. "
                "Pass them via CLI: --p_inlet <value> --p_outlet <value>"
            )
        self._p_inlet_val = float(p_inlet) / 2
        self._p_outlet_val = float(p_outlet) / 2
        self.beta_nitsche = beta_nitsche

        super().__init__(mesh, dt, rho, mu, f)

        super().initVelocitySpace(
            "Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,)
        )
        super().initPressureSpace("Lagrange", mesh.topology.cell_name(), 1)

        self.VQ = MixedFunctionSpace(self.V, self.Q)
        v, q = TestFunctions(self.VQ)

        if initial_velocity:
            self.u_prev.interpolate(initial_velocity)

        u_sol = self.u_sol
        p_sol = self.p_sol
        u_prev = self.u_prev
        u_mid = 0.5 * (u_sol + u_prev)
        self.u_mid = u_mid

        # Dimension-dependent curl/cross helpers.
        # 3D: curl(u) is a vector; cross products use UFL curl/cross directly.
        # 2D: curl(u) is a scalar ω = ∂u_y/∂x - ∂u_x/∂y; cross products are
        #     computed by embedding in 3D and projecting: (0,0,ω)×(a,b,0) = (-ω·b, ω·a).
        gdim = mesh.geometry.dim
        if gdim == 2:

            def _rot(w):
                return w[1].dx(0) - w[0].dx(1)

            def _curl_curl_inner(u, v):
                return _rot(u) * _rot(v)

            def _cross_curl_vec(w):
                omega = _rot(w)
                return as_vector([-omega * w[1], omega * w[0]])

            def _cross_curl_n(w, n):
                omega = _rot(w)
                return as_vector([-omega * n[1], omega * n[0]])

        else:

            def _curl_curl_inner(u, v):
                return inner(curl(u), curl(v))

            def _cross_curl_vec(w):
                return cross(curl(w), w)

            def _cross_curl_n(w, n):
                return cross(curl(w), n)

        self._cross_curl_n = _cross_curl_n  # used in setup() for Nitsche BCs

        # Implicit Euler, curl-curl viscous term, skew-symmetric convection
        F = self.rho * inner(v, (u_sol - u_prev) / self.dt) * dx
        F += self.mu * _curl_curl_inner(u_mid, v) * dx  # a(u,v)
        F -= p_sol * div(v) * dx  # b(v,p)
        F += self.rho * dot(_cross_curl_vec(u_mid), v) * dx  # N part 1
        F -= self.rho * 0.5 * dot(u_mid, u_mid) * div(v) * dx  # N part 2
        F -= self.rho * inner(v, self.f) * dx
        F += inner(q, div(u_mid)) * dx  # continuity

        # Stabilization — h via DG0 function (works for all cell types incl. hexahedra)
        V_dg0 = functionspace(mesh, ("DG", 0))
        h = Function(V_dg0)
        h.x.array[:] = mesh.h(
            mesh.topology.dim,
            np.arange(h.x.index_map.size_local + h.x.index_map.num_ghosts),
        )
        vnorm = sqrt(inner(u_prev, u_prev))

        # Strong residual — viscous term omitted per standard SUPG practice
        R = self.rho * ((u_sol - u_prev) / self.dt + _cross_curl_vec(u_mid))
        R += grad(p_sol) - self.rho * self.f

        eps = Constant(mesh, np.finfo(PETSc.ScalarType()).resolution)
        tau1 = h / conditional(ge(2.0 * vnorm, eps), 2.0 * vnorm, eps)
        tau2 = self.dt / 2.0
        tau3 = (h * h) / (4.0 * (self.mu / self.rho))
        tau = (1 / tau1**2 + 1 / tau2**2 + 1 / tau3**2) ** (-1 / 2)

        F_supg = inner(tau * R, dot(u_mid, nabla_grad(v))) * dx
        F_pspg = (1 / self.rho) * inner(tau * R, grad(q)) * dx

        Re = (vnorm * h) / (2.0 * (self.mu / self.rho))
        z = conditional(le(Re, 3), Re / 3, 1.0)
        tau_lsic = (vnorm * h * z) / 2.0
        F_lsic = tau_lsic * inner(div(u_mid), self.rho * div(v)) * dx

        F += F_supg + F_pspg + F_lsic

        self.F = F
        self._h = h
        self._v_test = v

    def updateSolution(self, x: petsc_typing.Vec) -> None:
        "Updates the solution functions u_sol and p_sol with the values in x."
        start_u, end_u = self.u_prev.x.petsc_vec.getOwnershipRange()
        start_p, end_p = self.p_prev.x.petsc_vec.getOwnershipRange()
        u_size_local = self.u_prev.x.petsc_vec.getLocalSize()

        self.u_sol.x.petsc_vec.setValues(
            range(start_u, end_u), x.array_r[:u_size_local]
        )
        self.p_sol.x.petsc_vec.setValues(
            range(start_p, end_p), x.array_r[u_size_local:]
        )
        self.u_sol.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.p_sol.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    def assembleJacobian(
        self,
        snes: petsc_typing.SNES,
        x: petsc_typing.Vec,
        J: petsc_typing.Mat,
        P: petsc_typing.Mat,
        bcs: list[DirichletBC] = [],
    ) -> None:
        "Assembles the Jacobian matrix evaluated at u_sol and p_sol."
        J_mat = J.getPythonContext().Mat if J.getType() == "python" else J
        J.zeroEntries()
        fem.petsc.assemble_matrix_block(J_mat, self.J_form, bcs, diagonal=1.0)
        J.assemble()
        if P is not None and P != J:
            P_mat = P.getPythonContext().Mat if P.getType() == "python" else P
            P.zeroEntries()
            fem.petsc.assemble_matrix_block(P_mat, self.J_form, bcs, diagonal=1.0)
            P.assemble()

    def assembleResidual(
        self,
        snes: PETSc.SNES,
        x: PETSc.Vec,
        F_vec: PETSc.Vec,
        bcs: list[DirichletBC] = [],
    ) -> None:
        "Assembles the residual vector evaluated at u_sol and p_sol, applies lifting and set_bcs so that the constrained dofs are = x_n - g."
        with F_vec.localForm() as F_local:
            F_local.set(0.0)

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.updateSolution(x)
        [bc.update() for bc in bcs]

        assemble_vector_block(
            F_vec, self.F_form, self.J_form, bcs=bcs, x0=x, alpha=-1.0
        )
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def setup(
        self,
        bcu: list[BoundaryCondition],
        bcp: list[BoundaryCondition],
        facet_tags=None,
        tags=None,
    ) -> None:
        # 1. Subdomain measures
        ds_in = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=facet_tags,
            subdomain_id=tags["inlet"],
        )
        ds_out = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=facet_tags,
            subdomain_id=tags["outlet"],
        )

        # 2. Natural pressure BCs (weak Neumann — adds p·v·n on inlet/outlet)
        n = FacetNormal(self.mesh)
        p_in = Constant(self.mesh, PETSc.ScalarType(self._p_inlet_val))
        p_out = Constant(self.mesh, PETSc.ScalarType(self._p_outlet_val))
        v = self._v_test
        self.F += p_in * dot(v, n) * ds_in
        self.F += p_out * dot(v, n) * ds_out

        # 3. Nitsche for tangential velocity (u_T = 0) on inlet and outlet
        u = self.u_mid
        h = self._h
        u_T = u - dot(u, n) * n
        v_T = v - dot(v, n) * n
        for ds_bc in (ds_in, ds_out):
            self.F += (
                -self.mu * dot(self._cross_curl_n(u, n), v_T) * ds_bc
                - self.mu * dot(self._cross_curl_n(v, n), u_T) * ds_bc
                + (self.beta_nitsche * self.mu / h) * dot(u_T, v_T) * ds_bc
            )

        # 4. Linearize and compile
        du, dp = TrialFunctions(self.VQ)
        J = derivative(self.F, (self.u_sol, self.p_sol), (du, dp))
        self.F_form = form(extract_blocks(self.F))
        self.J_form = form(extract_blocks(J))
        self.A = create_matrix_block(self.J_form)
        self.b = create_vector_block(self.F_form)
        self.x_n = self.b.duplicate()
        self.offset = (
            self.V.dofmap.index_map.size_local + self.V.dofmap.index_map.num_ghosts
        ) * self.V.dofmap.index_map_bs

        # 5. Wall Dirichlet BCs only; pressure BCs are handled naturally
        self.bcu_d = [bc.getBC(self.V) for bc in bcu]
        self.bcp_d = []
        bcs = [*self.bcu_d, *self.bcp_d]

        # 6. PCD secondary boundary conditions
        inlet_dofs_p = fem.locate_dofs_topological(
            self.Q, self.mesh.topology.dim - 1, facet_tags.find(tags["inlet"])
        )
        outlet_dofs_p = fem.locate_dofs_topological(
            self.Q, self.mesh.topology.dim - 1, facet_tags.find(tags["outlet"])
        )

        pcd_type = "PCDPC_vY"
        bcs_pcd = {
            "PCDPC_vX": [fem.dirichletbc(fem.Function(self.Q), inlet_dofs_p)],
            "PCDPC_vY": [fem.dirichletbc(fem.Function(self.Q), outlet_dofs_p)],
        }[pcd_type]

        appctx = {
            "nu": self.mu / self.rho,
            "v": self.u_sol,
            "bcs_pcd": bcs_pcd,
            "ds_in": ds_in,
        }

        # Initialize Jacobian matrix
        assemble_matrix_block(self.A, self.J_form, bcs)
        self.A.assemble()

        # Wrap matrix for FieldSplit / PCD
        J_splittable = create_splittable_matrix_block(
            self.A, extract_blocks(J), **appctx
        )

        # PETSc options configuration
        problem_prefix = "ns_"
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        opts["snes_type"] = "newtonls"
        opts["snes_rtol"] = 1.0e-04
        opts["snes_max_it"] = 50
        opts["snes_ksp_ew"] = True

        opts["ksp_converged_reason"] = ""
        opts["ksp_max_it"] = 10000
        opts["ksp_type"] = "fgmres"
        opts["ksp_gmres_restart"] = 150
        opts["ksp_pc_side"] = "right"

        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.pc.WrappedPC"
        opts.prefixPush("wrapped_")
        opts["pc_type"] = "fieldsplit"
        opts["pc_fieldsplit_type"] = "schur"
        opts["pc_fieldsplit_schur_fact_type"] = "upper"
        opts["pc_fieldsplit_schur_precondition"] = "user"
        opts["pc_fieldsplit_0_fields"] = 0  # velocity
        opts["pc_fieldsplit_1_fields"] = 1  # pressure

        opts["fieldsplit_0_ksp_type"] = "gmres"
        opts["fieldsplit_0_pc_type"] = "hypre"

        opts["fieldsplit_1_ksp_type"] = "preonly"
        opts["fieldsplit_1_pc_type"] = "python"
        opts["fieldsplit_1_pc_python_type"] = f"fenicsx_pctools.pc.{pcd_type}"
        opts["fieldsplit_1_pcd_Mp_ksp_type"] = "preonly"
        opts["fieldsplit_1_pcd_Mp_pc_type"] = "jacobi"
        opts["fieldsplit_1_pcd_Ap_ksp_type"] = "cg"
        opts["fieldsplit_1_pcd_Ap_pc_type"] = "hypre"
        opts.prefixPop()  # wrapped_
        opts.prefixPop()  # ns_

        # Newton solver
        snes = PETSc.SNES().create(self.mesh.comm)
        snes.setFunction(self.assembleResidual, f=self.b, kargs={"bcs": bcs})
        snes.setJacobian(
            self.assembleJacobian, J=J_splittable, P=None, kargs={"bcs": bcs}
        )
        snes.setOptionsPrefix(problem_prefix)
        snes.setFromOptions()

        # x is the initial guess for the newton iteration = solution at previous time step
        start, end = self.x_n.getOwnershipRange()
        u_size_local = self.u_prev.x.petsc_vec.getLocalSize()
        self.x_n.setValues(range(start, start + u_size_local), self.u_prev.x.petsc_vec)
        self.x_n.setValues(
            range(start + u_size_local, end),
            self.p_prev.x.petsc_vec,
        )
        self.x_n.assemble()

        snes.setUp()

        print("\n--- CONFIGURACIÓN DEL SOLVER ---")
        viewer = PETSc.Viewer.STDOUT(self.mesh.comm)
        snes.view(viewer)
        print("--------------------------------\n")

        self.solver = snes

    def solveStep(self):
        PETSc.Sys.Print("Solving the nonlinear problem with SNES")
        self.solver.solve(None, self.x_n)
        its_snes = self.solver.getIterationNumber()
        its_ksp = self.solver.getLinearSolveIterations()
        PETSc.Sys.Print(
            f"Solver converged in {its_snes} nonlinear iterations"
            f" (with total number of {its_ksp} linear iterations)"
        )
        self.updateSolution(self.x_n)

        reason = self.solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"Did not converge, reason: {reason}.")
