# Curl-curl (rotational) formulation with natural pressure BCs and Nitsche inlet tangential BC.
# SUPG/PSPG/LSIC stabilization, implicit Euler, SNES + Schur fieldsplit preconditioner.

from typing import Callable

import numpy as np
from dolfinx.fem import Constant, DirichletBC, Function, form, functionspace
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    assemble_vector_block,
    create_matrix_block,
    create_vector_block,
)
from dolfinx.mesh import Mesh
from petsc4py import PETSc
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
    ):
        if p_inlet is None or p_outlet is None:
            raise ValueError(
                "p_inlet and p_outlet are required for stabilized_schur_pressurebc. "
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

        # 6. SNES + Schur fieldsplit (identical structure to stabilized_schur.py)
        snes = PETSc.SNES().create(self.mesh.comm)
        snes.setOptionsPrefix("nonlinear_")
        snes.setType("newtonls")
        snes.setFunction(
            self.assembleResidual, f=self.b, kargs={"bcs": [*self.bcu_d, *self.bcp_d]}
        )
        snes.setJacobian(
            self.assembleJacobian,
            J=self.A,
            P=None,
            kargs={"bcs": [*self.bcu_d, *self.bcp_d]},
        )

        start, end = self.x_n.getOwnershipRange()
        u_size_local = self.u_prev.x.petsc_vec.getLocalSize()
        self.x_n.setValues(range(start, start + u_size_local), self.u_prev.x.petsc_vec)
        self.x_n.setValues(range(start + u_size_local, end), self.p_prev.x.petsc_vec)
        self.x_n.assemble()

        ksp = snes.getKSP()
        ksp.setType("fgmres")
        snes.computeJacobian(self.x_n, self.A)
        ksp.setOperators(self.A)

        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)

        V_map = self.V.dofmap.index_map
        Q_map = self.Q.dofmap.index_map
        offset_u = (
            V_map.local_range[0] * self.V.dofmap.index_map_bs + Q_map.local_range[0]
        )
        offset_p = offset_u + V_map.size_local * self.V.dofmap.index_map_bs
        is_u = PETSc.IS().createStride(
            V_map.size_local * self.V.dofmap.index_map_bs,
            offset_u,
            1,
            comm=self.mesh.comm,
        )
        is_p = PETSc.IS().createStride(
            Q_map.size_local, offset_p, 1, comm=self.mesh.comm
        )
        pc.setFieldSplitIS(("u", is_u), ("p", is_p))
        pc.setUp()

        ksp_u, ksp_p = pc.getFieldSplitSchurGetSubKSP()
        ksp_u.setType("gmres")
        ksp_u.getPC().setType("asm")
        ksp_p.setType("preonly")
        ksp_p.getPC().setType("asm")

        ksp_u.getPC().setUp()
        ksp_p.getPC().setUp()

        snes.setFromOptions()
        snes.setUp()

        viewer = PETSc.Viewer.STDOUT(self.mesh.comm)
        snes.view(viewer)

        self.solver = snes

        # Constant pressure null space
        vec_const = self.A.createVecs()[0]
        vec_const.set(0.0)
        indices_p = is_p.getIndices()
        for i in indices_p:
            vec_const.setValue(i, 1.0)
        vec_const.assemble()
        norm = vec_const.norm(PETSc.NormType.NORM_2)
        vec_const.scale(1.0 / norm)
        self.nullsp = PETSc.NullSpace().create(vectors=[vec_const], comm=self.mesh.comm)

    def updateSolution(self, x: PETSc.Vec) -> None:
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
        snes: PETSc.SNES,
        x: PETSc.Vec,
        J_mat: PETSc.Mat,
        P_mat: PETSc.Mat,
        bcs: list[DirichletBC] = [],
    ) -> None:
        J_mat.zeroEntries()
        assemble_matrix_block(J_mat, self.J_form, bcs)
        J_mat.assemble()

    def assembleResidual(
        self,
        snes: PETSc.SNES,
        x: PETSc.Vec,
        F_vec: PETSc.Vec,
        bcs: list[DirichletBC] = [],
    ) -> None:
        with F_vec.localForm() as F_local:
            F_local.set(0.0)

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.updateSolution(x)
        [bc.update() for bc in bcs]

        assemble_vector_block(
            F_vec, self.F_form, self.J_form, bcs=bcs, x0=x, alpha=-1.0
        )
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def solveStep(self):
        if self.nullsp.test(self.A):
            self.A.setNullSpace(self.nullsp)

        self.nullsp.remove(self.x_n)

        self.solver.solve(None, self.x_n)
        self.updateSolution(self.x_n)

        its_snes = self.solver.getIterationNumber()
        its_ksp = self.solver.getLinearSolveIterations()
        PETSc.Sys.Print(
            f"Solver converged in {its_snes} nonlinear iterations"
            f" (with total number of {its_ksp} linear iterations)"
        )

        reason = self.solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"Did not converge, reason: {reason}.")
