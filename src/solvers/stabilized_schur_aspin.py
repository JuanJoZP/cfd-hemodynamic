# SUPG, PSPG, and LSIC stabilization, newton linealization, full schur preconditioning

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
    MixedFunctionSpace,
    TestFunctions,
    TrialFunctions,
    conditional,
    derivative,
    div,
    dot,
    ds,
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
    ):
        super().__init__(mesh, dt, rho, mu, f)

        super().initVelocitySpace(
            "Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,)
        )
        super().initPressureSpace("Lagrange", mesh.topology.cell_name(), 1)

        self.VQ = MixedFunctionSpace(self.V, self.Q)

        v, q = TestFunctions(self.VQ)

        if initial_velocity:
            self.u_prev.interpolate(initial_velocity)

        # weak form
        u_sol = self.u_sol
        p_sol = self.p_sol
        u_prev = self.u_prev
        u_mid = 0.5 * (u_sol + u_prev)
        n = FacetNormal(self.mesh)

        F = self.rho * inner(v, (u_sol - u_prev) / self.dt) * dx
        F += self.rho * dot(v, dot(u_mid, nabla_grad(u_mid))) * dx
        F -= inner(v, self.rho * self.f) * dx
        F += inner(self.epsilon(v), self.sigma(u_mid, p_sol, self.mu)) * dx
        # probar p_prev en vez de p_sol
        F += dot(p_sol * n, v) * ds - dot(mu * nabla_grad(u_mid) * n, v) * ds
        F += inner(q, div(u_mid)) * dx

        # stabilization terms
        V_dg0 = functionspace(mesh, ("DG", 0))
        h = Function(V_dg0)
        h.x.array[:] = mesh.h(
            mesh.topology.dim,
            np.arange(h.x.index_map.size_local + h.x.index_map.num_ghosts),
        )

        vnorm = sqrt(
            inner(u_prev, u_prev)
        )  # u_prev instead of u_sol to avoid nonlinearity after derivation

        R = self.rho * ((u_sol - u_prev) / self.dt + dot(u_mid, nabla_grad(u_mid)))
        R -= div(self.sigma(u_mid, p_sol, self.mu))
        R -= self.rho * self.f

        # SUPG
        eps = Constant(self.mesh, np.finfo(PETSc.ScalarType()).resolution)
        tau_supg1 = h / conditional(
            ge((2.0 * vnorm), eps), (2.0 * vnorm), eps
        )  # avoid division by zero
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

    def updateSolution(self, x: PETSc.Vec) -> None:
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
        snes: PETSc.SNES,
        x: PETSc.Vec,
        J_mat: PETSc.Mat,
        P_mat: PETSc.Mat,
        bcs: list[DirichletBC] = [],
    ) -> None:
        "Assembles the Jacobian matrix evaluated at u_sol and p_sol."
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

    def setup(self, bcu: list[BoundaryCondition], bcp: list[BoundaryCondition]) -> None:
        # create linealizated problem
        du, dp = TrialFunctions(self.VQ)

        J = derivative(self.F, (self.u_sol, self.p_sol), (du, dp))
        self.F_form = form(extract_blocks(self.F))
        self.J_form = form(extract_blocks(J))

        self.A = create_matrix_block(self.J_form)
        self.b = create_vector_block(self.F_form)
        self.x_n = self.b.duplicate()  # solution to the nth newton iteration
        self.offset = (
            self.V.dofmap.index_map.size_local + self.V.dofmap.index_map.num_ghosts
        ) * self.V.dofmap.index_map_bs  # after this index values of x correspond to pressure, before to velocity

        self.bcu_d = [bc.getBC(self.V) for bc in bcu]
        self.bcp_d = [bc.getBC(self.Q) for bc in bcp]

        # newton solver
        snes = PETSc.SNES().create(self.mesh.comm)
        snes.setOptionsPrefix("nonlinear_")

        snes.setType("aspin")

        snes.setFunction(
            self.assembleResidual, f=self.b, kargs={"bcs": [*self.bcu_d, *self.bcp_d]}
        )
        snes.setJacobian(
            self.assembleJacobian,
            J=self.A,
            P=None,
            kargs={"bcs": [*self.bcu_d, *self.bcp_d]},
        )

        # x is the initial guess for the newton iteration = solution at previous time step
        start, end = self.x_n.getOwnershipRange()
        u_size_local = self.u_prev.x.petsc_vec.getLocalSize()
        self.x_n.setValues(range(start, start + u_size_local), self.u_prev.x.petsc_vec)
        self.x_n.setValues(
            range(start + u_size_local, end),
            self.p_prev.x.petsc_vec,
        )
        self.x_n.assemble()

        # fgmres global solver with field split (schur) preconditioner
        ksp = snes.getKSP()
        ksp.setType("fgmres")
        snes.computeJacobian(self.x_n, self.A)  # asemble A in order to set up PC
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

        # set solvers for schur and pressure blocks
        ksp_u, ksp_p = pc.getFieldSplitSchurGetSubKSP()
        ksp_u.setType("gmres")
        ksp_u.getPC().setType("asm")
        ksp_p.setType("preonly")
        ksp_p.getPC().setType("asm")

        ksp_u.getPC().setUp()
        ksp_p.getPC().setUp()

        snes.setFromOptions()
        snes.setUp()
        self.solver = snes

        # constant pressure null space
        vec_const = self.A.createVecs()[0]
        vec_const.set(0.0)
        indices_p = is_p.getIndices()
        for i in indices_p:
            vec_const.setValue(i, 1.0)
        vec_const.assemble()

        norm = vec_const.norm(PETSc.NormType.NORM_2)
        vec_const.scale(1.0 / norm)  # normalize

        self.nullsp = PETSc.NullSpace().create(vectors=[vec_const], comm=self.mesh.comm)

    def solveStep(self):
        if self.nullsp.test(self.A):
            self.A.setNullSpace(
                self.nullsp
            )  # esto deberia moverse a setup pero da error

        self.nullsp.remove(self.x_n)  # creo que se puede quitar?

        self.solver.solve(None, self.x_n)
        self.updateSolution(self.x_n)

        reason = self.solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"Did not converge, reason: {reason}.")
