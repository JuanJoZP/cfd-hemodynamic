# SUPG, PSPG, and LSIC stabilization, newton linealization, full schur preconditioning

from typing import Callable
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from basix.ufl import element, mixed_element
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
    create_matrix_block,
    create_vector_block,
    assemble_matrix_block,
    assemble_vector_block,
)

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

        F += F_supg + F_lsic
        F += F_pspg
        self.F = F

    def assembleJacobian(self, bcs: list[DirichletBC]) -> None:
        "Assembles the Jacobian matrix evaluated at u_sol and p_sol."
        self.A.zeroEntries()
        assemble_matrix_block(self.A, self.J_form, bcs)
        self.A.assemble()

    def assembleResidual(self, bcs: list[DirichletBC]) -> None:
        "Assembles the residual vector evaluated at u_sol and p_sol, applies lifting and set_bcs so that the constrained dofs are x_n - g."
        with self.b.localForm() as b_local:
            b_local.set(0.0)

        # set x_n to the solution in previous newton iteration
        self.x_n.setValues(range(0, self.offset), self.u_sol.x.petsc_vec)
        self.x_n.setValues(
            range(
                self.offset,
                self.offset
                + self.Q.dofmap.index_map.size_local * self.Q.dofmap.index_map_bs,
            ),
            self.p_sol.x.petsc_vec,
        )
        self.x_n.assemble()

        assemble_vector_block(
            self.b, self.F_form, self.J_form, bcs=bcs, x0=self.x_n, alpha=-1.0
        )

    def assembleTimeIndependent(
        self, bcu: list[DirichletBC], bcp: list[DirichletBC]
    ) -> None:
        # create linealizated problem
        du, dp = TrialFunctions(self.VQ)

        J = derivative(self.F, (self.u_sol, self.p_sol), (du, dp))

        self.F_form = form(extract_blocks(self.F))
        self.J_form = form(extract_blocks(J))

        self.A = create_matrix_block(self.J_form)
        self.b = create_vector_block(self.F_form)
        self.x_n = self.b.duplicate()  # solution to the nth newton iteration
        self.offset = (
            self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        )  # after this index values of x correspond to pressure, before to velocity

        # !!!!!
        # solve x_0 the initial guess in the first timestep
        # self.x_0 = self.b.duplicate()
        # self.x0.set ...
        # then make u_sol, p_sol = x_0 so it is assembled in the first iteration

        self.assembleJacobian([*bcu, *bcp])
        self.assembleResidual([*bcu, *bcp])

        # gmres global solver with field split preconditioner
        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setType("gmres")
        ksp.setTolerances(rtol=1e-4)
        ksp.setOperators(self.A)

        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)  # temporal
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.FULL)  # temporal

        IS = PETSc.IS
        fields = self.V.dofmap.index_map, self.Q.dofmap.index_map
        u_size = fields[0].size_local * self.V.dofmap.index_map_bs
        p_size = fields[1].size_local * self.Q.dofmap.index_map_bs
        is_u = IS().createStride(u_size, 0, 1, self.mesh.comm)
        is_p = IS().createStride(p_size, u_size, 1, self.mesh.comm)
        pc.setFieldSplitIS(("u", is_u), ("p", is_p))

        # solve the schur block with gmres and the pressure block with cg
        pc.setUp()
        ksp_u, ksp_p = pc.getFieldSplitSchurGetSubKSP()
        ksp_u.setType("gmres")
        ksp_u.getPC().setType("ilu")
        ksp_p.setType("cg")
        ksp_p.getPC().setType("hypre")
        self.solver = ksp

    def solveIteration(self, bcu: list[DirichletBC], bcp: list[DirichletBC]) -> float:
        "Solve one newton iteration with the current bcs. It assumes that the previous guess (or initial guess) is the one in u_sol and p_sol."
        self.assembleJacobian(bcs=[*bcu, *bcp])
        self.assembleResidual(bcs=[*bcu, *bcp])

        solver = self.solver
        solver.setOperators(self.A)
        solver.solve(self.b, self.x_n)  # use x_n by now as dx

        x_u_array = self.x_n.array_r[: self.offset]
        x_p_array = self.x_n.array_r[self.offset :]

        self.u_sol.x.array[:] -= x_u_array
        self.p_sol.x.array[:] -= x_p_array
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
