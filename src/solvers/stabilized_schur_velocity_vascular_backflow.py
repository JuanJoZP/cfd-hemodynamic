# SUPG/PSPG/LSIC stabilization, Newton linearization, Schur fieldsplit preconditioner.
# Inlet: strong Dirichlet parabolic velocity (from scenario).
# Outlet: prescribed pressure p_c + backflow stabilization (Moghadam et al. 2011, Eq. 10).
#
# Outlet traction: sigma·n = -p_c·n - rho*theta*(u·n)_- * u
#   where p_c is fixed (default 1000), theta = beta_backflow,
#   (u·n)_- = (u·n - |u·n|) / 2  (active only when flow reverses at the outlet).

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
    conditional,
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
        v_max: float = None,
        p_grade: int = 1,
        beta_backflow: float = 0.2,
        p_outlet: float = 1000.0,
    ):
        if v_max is None:
            raise ValueError(
                "v_max is required for stabilized_schur_velocity_vascular_backflow. "
                "Pass it via CLI: --v_max <value>"
            )

        self.v_max = float(v_max)
        self.beta_backflow = float(beta_backflow)
        self._p_outlet_val = float(p_outlet)

        if mesh.comm.rank == 0:
            print(
                f"[Solver] p_grade={p_grade}, v_max={self.v_max:.4f}, "
                f"beta_backflow={self.beta_backflow:.2f}, "
                f"p_outlet={self._p_outlet_val:.4f}",
                flush=True,
            )

        super().__init__(mesh, dt, rho, mu, f)

        super().initVelocitySpace(
            "Lagrange", mesh.topology.cell_name(), p_grade, shape=(mesh.geometry.dim,)
        )
        super().initPressureSpace("Lagrange", mesh.topology.cell_name(), p_grade)

        self.VQ = MixedFunctionSpace(self.V, self.Q)
        v, q = TestFunctions(self.VQ)

        if initial_velocity:
            self.u_prev.interpolate(initial_velocity)

        # weak form
        u_sol = self.u_sol
        p_sol = self.p_sol
        u_prev = self.u_prev
        u_mid = 0.5 * (u_sol + u_prev)
        self.u_mid = u_mid
        n = FacetNormal(self.mesh)

        F = self.rho * inner(v, (u_sol - u_prev) / self.dt) * dx
        F += self.rho * dot(v, dot(u_mid, nabla_grad(u_mid))) * dx
        F -= inner(v, self.rho * self.f) * dx
        F += inner(self.epsilon(v), self.sigma(u_mid, p_sol, self.mu)) * dx
        F += inner(q, div(u_mid)) * dx

        # stabilization terms
        V_dg0 = functionspace(mesh, ("DG", 0))
        h = Function(V_dg0)
        h.x.array[:] = mesh.h(
            mesh.topology.dim,
            np.arange(h.x.index_map.size_local + h.x.index_map.num_ghosts),
        )

        vnorm = sqrt(inner(u_prev, u_prev))

        R = self.rho * ((u_sol - u_prev) / self.dt + dot(u_mid, nabla_grad(u_mid)))
        R -= div(self.sigma(u_mid, p_sol, self.mu))
        R -= self.rho * self.f

        # SUPG
        eps = Constant(self.mesh, np.finfo(PETSc.ScalarType()).resolution)
        tau_supg1 = h / conditional(ge((2.0 * vnorm), eps), (2.0 * vnorm), eps)
        tau_supg2 = self.dt / 2.0
        tau_supg3 = (h * h) / (4.0 * (self.mu / self.rho))
        tau_supg = (
            1 / (tau_supg1**2) + 1 / (tau_supg2**2) + 1 / (tau_supg3**2)
        ) ** (-1 / 2)
        F_supg = inner(tau_supg * R, dot(u_mid, nabla_grad(v))) * dx

        # PSPG
        tau_pspg = tau_supg
        F_pspg = (1 / self.rho) * inner(tau_pspg * R, grad(q)) * dx

        # LSIC
        Re = (vnorm * h) / (2.0 * (self.mu / self.rho))
        z = conditional(le(Re, 3), Re / 3, 1.0)
        tau_lsic = (vnorm * h * z) / 2.0
        F_lsic = tau_lsic * inner(div(u_mid), self.rho * div(v)) * dx

        F += F_supg + F_lsic
        F += F_pspg
        self.F = F
        self._v_test = v

    def setup(
        self,
        bcu: list[BoundaryCondition],
        bcp: list[BoundaryCondition],
        facet_tags=None,
        tags=None,
    ) -> None:
        # 1. Outlet subdomain measure
        ds_out = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=facet_tags,
            subdomain_id=tags["outlet"],
        )

        n = FacetNormal(self.mesh)
        v = self._v_test
        u = self.u_mid

        # 2. Prescribed outlet pressure p = p_c
        #    From IBP: −∫(σ·n)·v dΓ = −∫(2με(u)·n)·v dΓ + ∫p(v·n) dΓ
        #    Replace p → p_c in the pressure boundary integral,
        #    keep viscous boundary integral as natural (involves unknown u):
        p_c = Constant(self.mesh, PETSc.ScalarType(self._p_outlet_val / 2))
        self.F += p_c * dot(v, n) * ds_out
        self.F -= dot(dot(2 * self.mu * self.epsilon(u), n), v) * ds_out

        # 3. Backflow stabilization (Moghadam et al. 2011, Eq. 10):
        #    -beta * <w, rho * (u·n)_- * u>_{Gamma_out}
        #    (u·n)_- = (u·n - |u·n|) / 2   (negative when backflow)
        #    Semi-implicit: use u_prev for (u·n)_- coefficient, u_mid for velocity.
        u_prev = self.u_prev
        un_prev = dot(u_prev, n)
        un_minus = 0.5 * (un_prev - abs(un_prev))  # <= 0 when backflow, 0 otherwise
        self.F -= self.beta_backflow * self.rho * un_minus * dot(u, v) * ds_out

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

        # 5. Dirichlet BCs (wall no-slip + inlet parabolic from scenario)
        self.bcu_d = [bc.getBC(self.V) for bc in bcu]
        self.bcp_d = []

        # 6. SNES + Schur fieldsplit
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
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
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

        opts = PETSc.Options()
        opts["nonlinear_snes_max_it"] = 100
        opts["nonlinear_snes_monitor"] = ""
        opts["nonlinear_ksp_max_it"] = 1000
        opts["nonlinear_ksp_gmres_restart"] = 200
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

    def _updateResidual(self) -> None:
        u_size_local = self.u_residual.x.petsc_vec.getLocalSize()
        start_u, end_u = self.u_residual.x.petsc_vec.getOwnershipRange()
        start_p, end_p = self.p_residual.x.petsc_vec.getOwnershipRange()

        self.u_residual.x.petsc_vec.setValues(
            range(start_u, end_u), self.b.array_r[:u_size_local]
        )
        self.p_residual.x.petsc_vec.setValues(
            range(start_p, end_p), self.b.array_r[u_size_local:]
        )
        self.u_residual.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.p_residual.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    def solveStep(self):
        if self.nullsp.test(self.A):
            self.A.setNullSpace(self.nullsp)

        self.nullsp.remove(self.x_n)

        self.solver.solve(None, self.x_n)
        self.updateSolution(self.x_n)
        self._updateResidual()

        its_snes = self.solver.getIterationNumber()
        its_ksp = self.solver.getLinearSolveIterations()
        PETSc.Sys.Print(
            f"Solver converged in {its_snes} nonlinear iterations"
            f" (with total number of {its_ksp} linear iterations)"
        )

        reason = self.solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"Did not converge, reason: {reason}.")
