# SUPG, PSPG, and LSIC stabilization, newton linealization, full schur preconditioning

from typing import Callable
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from basix.ufl import element
from dolfinx.mesh import Mesh
from dolfinx.fem import (
    DirichletBC,
    Constant,
    Function,
    functionspace,
    form,
)
from ufl import (
    MixedFunctionSpace,
    TrialFunctions,
    derivative,
    dx,
    ds,
    dot,
    extract_blocks,
    inner,
    nabla_grad,
    grad,
    div,
    TestFunctions,
    sqrt,
    CellDiameter,
    conditional,
    le,
)
from dolfinx.fem.petsc import (
    create_matrix_nest,
    create_vector_nest,
    assemble_matrix_nest,
    assemble_vector_nest,
    apply_lifting_nest,
    set_bc_nest,
)

from src.solverBase import SolverBase


class SolverBase(SolverBase):
    MAX_ITER = 50
    ALPHA = 3  # regularization
    BETA = 0.5  # regularization

    def __init__(
        self,
        mesh: Mesh,
        dt: float,
        rho: float,
        mu: float,
        f: list,
        h: list = None,
        initial_velocity: Callable[[np.ndarray], np.ndarray] = None,
    ):
        self.mesh = mesh
        self.dt = Constant(mesh, PETSc.ScalarType(dt))
        self.rho = Constant(mesh, PETSc.ScalarType(rho))
        self.mu = Constant(mesh, PETSc.ScalarType(mu))
        self.f = Constant(mesh, PETSc.ScalarType(f))
        if h is None:
            self.h = Constant(mesh, PETSc.ScalarType([0.0] * mesh.geometry.dim))
        else:
            self.h = Constant(mesh, PETSc.ScalarType(h))

        element_velocity = element(
            "Lagrange",
            mesh.topology.cell_name(),
            1,
            shape=(mesh.geometry.dim,),
        )
        element_pressure = element("Lagrange", mesh.topology.cell_name(), 1)

        self.V = functionspace(mesh, element_velocity)
        self.Q = functionspace(mesh, element_pressure)
        self.VQ = MixedFunctionSpace(self.V, self.Q)

        v, q = TestFunctions(self.VQ)

        self.u_sol, self.p_sol = Function(self.V), Function(self.Q)
        self.u_prev, self.p_prev = Function(self.V), Function(self.Q)

        if initial_velocity:
            self.u_sol.interpolate(initial_velocity)
            self.u_prev.interpolate(initial_velocity)

        # weak form
        u_sol = self.u_sol
        p_sol = self.p_sol
        u_prev = self.u_prev
        u_mid = 0.5 * (u_sol + u_prev)

        F = self.rho * inner(v, (u_sol - u_prev) / self.dt) * dx
        F += self.rho * dot(v, dot(u_mid, nabla_grad(u_mid))) * dx
        F -= inner(v, self.rho * self.f) * dx
        F += inner(self.epsilon(v), self.sigma(u_mid, p_sol, self.mu)) * dx
        F -= inner(v, self.h) * ds
        F += inner(q, div(u_mid)) * dx

        # stabilization terms
        h = CellDiameter(mesh)
        vnorm = sqrt(inner(u_mid, u_mid))

        R = self.rho * ((u_sol - u_prev) / self.dt + dot(u_mid, nabla_grad(u_mid)))
        R -= div(self.sigma(u_mid, p_sol, self.mu))
        R -= self.rho * self.f

        # SUPG
        tau_supg1 = h / (2.0 * vnorm)
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

        self.F = F
        self.F_stab = F_supg + F_lsic + F_pspg

    def assembleJacobian(self, bcu: list[DirichletBC], bcp: list[DirichletBC]) -> None:
        "Assembles the Jacobian matrix evaluated at u_sol and p_sol."
        bcs = [*bcu, *bcp]

        self.A.zeroEntries()
        assemble_matrix_nest(self.A, self.J_form, bcs)
        self.A.assemble()

        self.A_stab.zeroEntries()
        assemble_matrix_nest(self.A_stab, self.J_form_stab, bcs)
        self.A_stab.assemble()

    def assembleResidual(self, bcu: list[DirichletBC], bcp: list[DirichletBC]) -> None:
        "Assembles the residual vector evaluated at u_sol and p_sol, applies lifting and set_bcs so that the constrained dofs are x_n - g."
        bcs = [*bcu, *bcp]

        for b_sub in self.b.getNestSubVecs():
            with b_sub.localForm() as b_local:
                b_local.set(0.0)
        assemble_vector_nest(self.b, self.F_form_stab)
        assemble_vector_nest(self.b, self.F_form)
        self.b.scale(-1.0)  # in this case RHS = - residual

        u_n, p_n = self.x_n.getNestSubVecs()

        self.u_sol.x.petsc_vec.copy(u_n)
        self.p_sol.x.petsc_vec.copy(p_n)

        apply_lifting_nest(self.b, self.J_form, bcs, x0=self.x_n, alpha=1.0)
        for b_sub in self.b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        apply_lifting_nest(self.b, self.J_form_stab, bcs, x0=self.x_n, alpha=1.0)
        for b_sub in self.b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        set_bc_nest(self.b, [bcu, bcp], x0=self.x_n, alpha=1.0)

    def assembleTimeIndependent(
        self, bcu: list[DirichletBC], bcp: list[DirichletBC]
    ) -> None:
        # create linealizated problem
        du, dp = TrialFunctions(self.VQ)

        J = derivative(self.F, (self.u_sol, self.p_sol), (du, dp))

        self.F_form = form(extract_blocks(self.F))
        self.J_form = form(extract_blocks(J))

        J_stab = derivative(self.F_stab, (self.u_sol, self.p_sol), (du, dp))

        self.F_form_stab = form(extract_blocks(self.F_stab))
        self.J_form_stab = form(extract_blocks(J_stab))

        self.A = create_matrix_nest(self.J_form)
        self.A_stab = create_matrix_nest(self.J_form_stab)

        self.b = create_vector_nest(self.F_form)

        self.x_n = self.b.duplicate()  # solution to the nth newton iteration

        # !!!!!
        # solve x_0 the initial guess in the first timestep
        # self.x_0 = self.b.duplicate()
        # self.x0.set ...
        # then make u_sol, p_sol = x_0 so it is assembled in the first iteration

        self.assembleJacobian(bcu, bcp)
        self.assembleResidual(bcu, bcp)

        # solver for dp
        ksp1 = PETSc.KSP().create(comm=self.mesh.comm)
        ksp1.setType("gmres")
        ksp1.getPC().setType("ilu")
        ksp1.setTolerances(rtol=1e-8)
        self.ksp1 = ksp1

        # solver for du
        ksp2 = PETSc.KSP().create(comm=self.mesh.comm)
        ksp2.setType("cg")
        ksp2.getPC().setType("hypre")
        ksp2.setTolerances(rtol=1e-8)
        self.ksp2 = ksp2

    def solveIteration(self, bcu: list[DirichletBC], bcp: list[DirichletBC]) -> float:
        "Solve one newton iteration with the current bcs. It assumes that the previous guess (or initial guess) is the one in u_sol and p_sol."
        self.assembleJacobian(bcu, bcp)
        self.assembleResidual(bcu, bcp)

        # !! VER SI ALGUNA DE LAS MATRICES NO ES DEPENDIENTE DEL TIEMPO

        # assemble aproximation to inverse of K
        diagA = self.A.getNestSubMatrix(0, 0).getDiagonal()
        diagA.reciprocal()
        self.K_tilde = self.A.getNestSubMatrix(0, 0).duplicate()
        self.K_tilde.setDiagonal(diagA)
        self.K_tilde.scale(1.0 + self.ALPHA)  # = inv((1 + alpha) * diag(K))

        # step 1
        OP1 = self.A_stab.getNestSubMatrix(1, 1).copy()  # = -C
        OP1.scale(-1.0)  # = C
        OP1.axpy(
            1.0,
            self.A.getNestSubMatrix(1, 0).matMatMult(
                self.K_tilde, self.A.getNestSubMatrix(0, 1)
            ),
        )  # C + GT * K_tilde * G

        self.ksp1.setOperators(OP1)

        RHS1 = self.x_n.getNestSubVecs()[1].duplicate()
        Ru, Rp = self.b.getNestSubVecs()  # -Ru, Rp
        Ru.scale(-1.0)  # = Ru
        self.A.getNestSubMatrix(1, 0).matMult(self.K_tilde).multAdd(
            Ru, Rp, RHS1
        )  # RHS1 = GT * K_tilde * Ru + Rp
        RHS1.scale(-1.0)  # = -Rp - GT * K_tilde * Ru

        self.ksp1.solve(RHS1, self.x_n.getNestSubVecs()[1])  # save dp in x_n[1]
        self.x_n.getNestSubVecs()[1].ghostUpdate(
            PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
        )

        # step 2
        OP2 = self.A.getNestSubMatrix(0, 0).copy()  # = K
        OP2.axpy(self.BETA, self.A_stab.getNestSubMatrix(0, 0))  # = K + beta * K_tau
        self.ksp2.setOperators(OP2)

        RHS2 = self.x_n.getNestSubVecs()[0].duplicate()
        A_full01 = self.A.getNestSubMatrix(0, 1)
        A_full01.axpy(1.0, self.A_stab.getNestSubMatrix(0, 1))  # = G + Du
        A_full01.multAdd(self.x_n.getNestSubVecs()[1], Ru, RHS2)
        RHS2.scale(-1.0)

        self.ksp2.solve(RHS2, self.x_n.getNestSubVecs()[0])  # save du in x_n[0]
        self.x_n.getNestSubVecs()[0].ghostUpdate(
            PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
        )

        # update u_sol and p_sol
        self.u_sol.x.petsc_vec.axpy(1.0, self.x_n.getNestSubVecs()[0])
        self.p_sol.x.petsc_vec.axpy(1.0, self.x_n.getNestSubVecs()[1])
        self.u_sol.x.scatter_forward()
        self.p_sol.x.scatter_forward()

        return self.x_n.norm()

    def solveStep(self, bcu: list[DirichletBC], bcp: list[DirichletBC]):
        # get x0 for this t: interpolation or oseen
        # self.x0.set ...
        # then set u_sol, p_sol to x0
        norm_dx = self.solveIteration(bcu, bcp)

        it = 1
        if norm_dx >= 1e-8:
            while it <= self.MAX_ITER:
                norm_dx = self.solveIteration(bcu, bcp)
                if norm_dx < 1e-8:
                    break

                it += 1

        if norm_dx < 1e-8:
            print(f"Converged after {it} iterations. |dx| = {norm_dx:.3e}")

            self.u_prev.x.array[:] = self.u_sol.x.array[:]
            self.p_prev.x.array[:] = self.p_sol.x.array[:]
        else:
            raise RuntimeError(
                f"Did not converge after {it} iterations. |dx| = {norm_dx:.3e}"
            )
