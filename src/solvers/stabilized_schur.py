# SUPG, PSPG, and LSIC stabilization, newton linealization, full schur preconditioning

from typing import Callable
from petsc4py import PETSc
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
    FacetNormal,
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

from src.solvers.stokes import StokesSolver
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
        h: list = None,
        initial_velocity: Callable[[np.ndarray], np.ndarray] = None,
    ):
        self.mesh = mesh
        self.dt = Constant(mesh, PETSc.ScalarType(dt))
        self.rho = Constant(mesh, PETSc.ScalarType(rho))
        self.mu = Constant(mesh, PETSc.ScalarType(mu))
        self.f = Constant(mesh, PETSc.ScalarType(f))
        if h == None:
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

        self.u_sol: Function = Function(self.V)
        self.p_sol: Function = Function(self.Q)
        self.u_prev, self.p_prev = Function(self.V), Function(self.Q)

        if initial_velocity:
            self.u_sol.interpolate(initial_velocity)
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
        # F -= inner(v, self.h) * ds
        # probar p_prev en vez de p_sol
        F += dot(p_sol * n, v) * ds - dot(mu * nabla_grad(u_mid) * n, v) * ds
        F += inner(q, div(u_mid)) * dx

        # stabilization terms
        h = CellDiameter(mesh)
        vnorm = sqrt(
            inner(u_prev, u_prev)
        )  # u_prev instead of u_sol to avoid nonlinearity after derivation

        R = self.rho * ((u_sol - u_prev) / self.dt + dot(u_mid, nabla_grad(u_mid)))
        R -= div(self.sigma(u_mid, p_sol, self.mu))
        R -= self.rho * self.f

        # SUPG
        tau_supg1 = h / (
            (2.0 * vnorm) + Constant(self.mesh, PETSc.ScalarType(1e-10))
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

    def updateSolution(self, x) -> None:
        "Updates the solution functions u_sol and p_sol with the values in x."
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.u_sol.x.array[:] = x[: self.offset]
        self.p_sol.x.array[:] = x[self.offset :]
        self.u_sol.x.scatter_forward()
        self.p_sol.x.scatter_forward()

    def assembleJacobian(
        self, snes, x, J_mat, P_mat, bcs: list[DirichletBC] = []
    ) -> None:
        "Assembles the Jacobian matrix evaluated at u_sol and p_sol."
        J_mat.zeroEntries()
        assemble_matrix_block(J_mat, self.J_form, bcs)
        J_mat.assemble()

    def assembleResidual(self, snes, x, F_vec, bcs: list[DirichletBC] = []) -> None:
        "Assembles the residual vector evaluated at u_sol and p_sol, applies lifting and set_bcs so that the constrained dofs are x_n - g."
        self.updateSolution(x)
        with F_vec.localForm() as F_local:
            F_local.set(0.0)

        assemble_vector_block(F_vec, self.F_form, self.J_form, bcs=bcs, x0=x, alpha=-1.0)

    def assembleTimeIndependent(
        self, bcu: list[BoundaryCondition], bcp: list[BoundaryCondition]
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

        # guess for the soltion at t = 0
        # !!! solamente deberia si no se provee initial_solution
        stokes_solver = StokesSolver(self.mesh, self.rho, self.mu, self.f)
        bcu_stokes = [bc.getBC(stokes_solver.V) for bc in bcu]
        bcp_stokes = [bc.getBC(stokes_solver.Q) for bc in bcp]
        stokes_solver.solve([*bcu_stokes, *bcp_stokes])

        self.u_prev.interpolate(stokes_solver.u_sol)
        self.p_prev.interpolate(stokes_solver.p_sol)

        bcu_d = [bc.getBC(self.V) for bc in bcu]
        bcp_d = [bc.getBC(self.Q) for bc in bcp]

        # newton solver
        snes = PETSc.SNES().create(self.mesh.comm)
        snes.setOptionsPrefix("nonlinear_")
        snes.setType("newtonls")
        snes.setFunction(self.assembleResidual, f=self.b, kargs={"bcs": [*bcu_d, *bcp_d]})
        snes.setJacobian(
            self.assembleJacobian, J=self.A, P=None, kargs={"bcs": [*bcu_d, *bcp_d]}
        )

        # x is the initial guess for the newton iteration = solution at previous time step
        self.x_n.setValues(range(0, self.offset), self.u_prev.x.petsc_vec)
        self.x_n.setValues(
            range(
                self.offset,
                self.offset
                + self.Q.dofmap.index_map.size_local * self.Q.dofmap.index_map_bs,
            ),
            self.p_prev.x.petsc_vec,
        )
        self.x_n.assemble()

        # gmres global solver with field split preconditioner
        ksp = snes.getKSP()
        ksp.setType("gmres")
        snes.computeJacobian(self.x_n, self.A)  # asemble A in order to set up PC
        ksp.setOperators(self.A)

        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.DIAG)  # temporal
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)  # temporal

        IS = PETSc.IS
        fields = self.V.dofmap.index_map, self.Q.dofmap.index_map
        u_size = fields[0].size_local * self.V.dofmap.index_map_bs
        p_size = fields[1].size_local * self.Q.dofmap.index_map_bs
        is_u = IS().createStride(u_size, 0, 1, self.mesh.comm)
        is_p = IS().createStride(p_size, u_size, 1, self.mesh.comm)
        pc.setFieldSplitIS(("u", is_u), ("p", is_p))
        pc.setUp()

        # solve the schur block with gmres and the pressure block with cg
        ksp_u, ksp_p = pc.getFieldSplitSchurGetSubKSP()
        ksp_u.setType("gmres")
        ksp_u.getPC().setType("ilu")
        ksp_p.setType("cg")
        ksp_p.getPC().setType("ilu")

        snes.setMonitor(lambda snes, its, rnorm: print(f"Iter {its}: ||F(u)|| = {rnorm}"))
        snes.setFromOptions()
        snes.setUp()
        self.solver = snes

        # create pressure null space
        with self.A.createVecs()[0].localForm() as vec_const:
            vec_const.set(0.0)  # 0 on u part

            start, end = is_p.getIndices()[0], is_p.getIndices()[-1] + 1
            vec_const.array[start:end] = 1.0  # 1 on p part (constant)
            vec_const.assemble()

            norm = vec_const.norm(PETSc.NormType.NORM_2)
            vec_const.scale(1.0 / norm)  # normalize

            self.nullsp = PETSc.NullSpace().create(
                vectors=[vec_const], comm=self.mesh.comm
            )

    def solveStep(self, bcu: list[DirichletBC], bcp: list[DirichletBC]):
        if self.nullsp.test(self.A):
            self.A.setNullSpace(self.nullsp)

        self.nullsp.remove(self.x_n)  # creo que se puede quitar?

        self.solver.solve(None, self.x_n)
        self.updateSolution(self.x_n)

        reason = self.solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"Did not converge, reason: {reason}.")
        else:
            print(
                f"Converged after {self.solver.getIterationNumber()} iterations. residual: {self.solver.getFunctionNorm()}."
            )

            self.u_prev.x.array[:] = self.u_sol.x.array[:]
            self.p_prev.x.array[:] = self.p_sol.x.array[:]
